"""
Hand-only MANO clip loader for ManipTrans stage-1 (DexHandImitator).

Drop this file into  main/dataset/handonly_dataset_dexhand.py  and add to
ManipDataFactory.dataset_type (main/dataset/factory.py):

        elif type(index) == str and index.startswith("h"):
            dtype = "handonly"

Each clip is one .npz under  data/handonly/<side>/  named  h<idx>.npz  with:
    global_orient : [T, 3]   MANO root axis-angle (world frame, z-up, metres)
    hand_pose     : [T, 45]  MANO 15x3 axis-angle, flat_hand_mean=True, no PCA
    transl        : [T, 3]
    betas         : [10]     (optional, defaults to zeros)
    fps           : scalar   (must be 60 -> resample beforehand)

Usage in the env:  dataIndices=[h0]   (or  [h0,h1,h2]  for a multi-clip imitator)
"""

import os
from functools import lru_cache

import numpy as np
import torch
from manotorch.manolayer import ManoLayer

from main.dataset.transform import aa_to_rotmat
from .base import ManipData
from .decorators import register_manipdata

# MANO mesh vertex ids used by ManipTrans as fingertips (same topology for both hands)
TIP_VIDS = {"index_tip": 353, "middle_tip": 467, "pinky_tip": 695, "ring_tip": 576, "thumb_tip": 766}


class HandOnlyDexHand(ManipData):
    side = None  # "right" | "left"

    def __init__(
        self,
        *,
        data_dir: str = None,
        split: str = "all",
        skip: int = 2,  # only used by the base class as time_delta = 1/(120/skip) -> 1/60 s
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        **kwargs,
    ):
        data_dir = data_dir or f"data/handonly/{self.side}"
        super().__init__(
            data_dir=data_dir,
            split=split,
            skip=skip,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            **kwargs,
        )
        self.manolayer = ManoLayer(
            rot_mode="axisang",
            side=self.side,
            center_idx=None,
            mano_assets_root="data/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)

        # Your data is z-up world; the env's mujoco2gym_transf assumes y-up (MuJoCo) input,
        # so undo its rotation exactly like the GRAB loader does.
        transf_offset = np.eye(4)
        transf_offset[:3, :3] = aa_to_rotmat(np.array([-np.pi / 2, 0, 0])) @ aa_to_rotmat(np.array([0, 0, np.pi / 2]))
        self.transf_offset = torch.tensor(transf_offset, dtype=torch.float32, device=mujoco2gym_transf.device)
        self.mujoco2gym_transf = mujoco2gym_transf @ self.transf_offset

        self.data_pathes = sorted(
            os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".npz")
        )

    def _path(self, idx: str):
        p = os.path.join(self.data_dir, f"{idx}.npz")
        assert os.path.exists(p), p
        return p

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        assert isinstance(idx, str) and idx.startswith("h"), idx
        raw = np.load(self._path(idx))
        assert int(raw["fps"]) == 60, "ManipTrans expects 60 FPS; resample first"

        T = raw["global_orient"].shape[0]
        dev = self.device
        go = torch.tensor(raw["global_orient"], dtype=torch.float32, device=dev)
        hp = torch.tensor(raw["hand_pose"], dtype=torch.float32, device=dev)
        tr = torch.tensor(raw["transl"], dtype=torch.float32, device=dev)
        betas = torch.tensor(raw["betas"] if "betas" in raw else np.zeros(10), dtype=torch.float32, device=dev)
        betas = betas[None].expand(T, -1)

        out = self.manolayer(torch.cat([go, hp], dim=-1), betas)
        verts = out.verts + tr[:, None]  # [T, 778, 3]
        joints = torch.matmul(self.manolayer.th_J_regressor, verts)  # [T, 16, 3] native MANO order

        wrist_pos = joints[:, 0]
        middle_prox = joints[:, 4]
        wrist_pos = wrist_pos - (middle_prox - wrist_pos) * 0.25  # same wrist hack as GRAB/OakInk loaders

        mano_joints = {
            "index_proximal": joints[:, 1], "index_intermediate": joints[:, 2], "index_distal": joints[:, 3],
            "middle_proximal": joints[:, 4], "middle_intermediate": joints[:, 5], "middle_distal": joints[:, 6],
            "pinky_proximal": joints[:, 7], "pinky_intermediate": joints[:, 8], "pinky_distal": joints[:, 9],
            "ring_proximal": joints[:, 10], "ring_intermediate": joints[:, 11], "ring_distal": joints[:, 12],
            "thumb_proximal": joints[:, 13], "thumb_intermediate": joints[:, 14], "thumb_distal": joints[:, 15],
        }
        for k, vid in TIP_VIDS.items():
            mano_joints[k] = verts[:, vid]

        wrist_rot = aa_to_rotmat(go) @ torch.tensor(
            np.repeat(self.dexhand.relative_rotation[None], T, axis=0), dtype=torch.float32, device=dev
        )

        # ---- dummy object: the imitator env never simulates it, but base.process_data and
        # ---- the env's pack_data/BPS still read these keys.
        dummy_verts = torch.full((1000, 3), 10.0, device=dev)  # far away -> tips_distance is meaningless, unused
        obj_traj = torch.eye(4, device=dev)[None].repeat(T, 1, 1)

        data = {
            "data_path": self._path(idx),
            "obj_id": "-1",
            "obj_verts": dummy_verts,
            "obj_urdf_path": "",
            "obj_trajectory": obj_traj,
            "scene_objs": [],
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
        }
        self.process_data(data, idx, dummy_verts)
        # Stage-1 imitator does not use retargeted (opt_*) data; this just fills zeros without the warning.
        self.verbose = False
        self.load_retargeted_data(data, "/nonexistent")
        return data


@register_manipdata("handonly_rh")
class HandOnlyDexHandRH(HandOnlyDexHand):
    side = "right"


@register_manipdata("handonly_lh")
class HandOnlyDexHandLH(HandOnlyDexHand):
    side = "left"
