"""
Bimanual stage-1 imitator with hand-hand collision (task name: DexHandImitatorBiH, side=BiH).

One policy controls both hands ("united" bimanual_mode of the ManipTrans configs, which already doubles every
obs/network dim when side=BiH). Per env: right hand actor, left hand actor, table (no collision).
  * collision filters: right 0b01, left 0b10 -> the two hands collide with each other; self-collision of each
    hand is off (same as Arti-MANO single-hand runs); the table is in its own collision group.
  * obs  = [right | left] for proprioception, privileged (dq) and target (same per-hand layout as the RH imitator)
  * act  = [right (6|9 + ndof) | left (6|9 + ndof)]
  * reward = right + left (each the original ManipTrans imitation reward, max 6.4)
  * failure if either hand exceeds its thresholds; success if both reach the end.
  * RSI samples ONE start frame shared by both hands.
Data: dataIndices=[h_<m>_<clip>_<lab>] is loaded from data/handonly/right AND data/handonly/left.

Evaluation (test=true and EVAL_OUT set): first episode of every env, written to $EVAL_OUT.json/.npz/_traj.npz,
same keys as the single-hand evaluator plus inter-hand contact metrics:
  x_contact_frac      fraction of steps with inter-hand contact (net contact force > 0.1 N on any hand body)
  x_contact_force_N   mean over steps of the summed contact-force magnitude on both hands
  x_contact_force_max_N
(with per-hand self-collision off and the table filtered out, net contact force on a hand body can only come
 from the other hand).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_mul

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_quat, aa_to_rotmat, rotmat_to_quat, rot6d_to_aa
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from ...utils import torch_jit_utils
from ..core.config import ROBOT_HEIGHT
from ..core.vec_task import VecTask
from .dexhandimitator import DexHandImitatorRHEnv, compute_imitation_reward

SIDES = ("rh", "lh")
SIDE_NAME = {"rh": "right", "lh": "left"}
FILTER = {"rh": 0b01, "lh": 0b10}


class DexHandImitatorBiHEnv(DexHandImitatorRHEnv):
    side = "BiH"

    # ------------------------------------------------------------------ init
    def __init__(self, cfg, *, rl_device=0, sim_device=0, graphics_device_id=0, display=False, record=False,
                 headless=True):
        self._record = record
        self.cfg = cfg
        assert not cfg["env"]["useQuatRot"], "BiH imitator supports useQuatRot=False only"
        self.use_quat_rot = False
        self.max_episode_length = cfg["env"]["episodeLength"]
        self.action_scale = cfg["env"]["actionScale"]
        self.aggregate_mode = cfg["env"]["aggregateMode"]
        self.training = cfg["env"]["training"]

        self.hands = {s: DexHandFactory.create_hand(cfg["env"]["dexhand"], SIDE_NAME[s]) for s in SIDES}
        self.dexhand = self.hands["rh"]  # for code paths of the parent that only need a reference hand

        self.use_pid_control = cfg["env"]["usePIDControl"]
        if self.use_pid_control:
            h = self.hands["rh"]
            self.Kp_rot, self.Ki_rot, self.Kd_rot = h.Kp_rot, h.Ki_rot, h.Kd_rot
            self.Kp_pos, self.Ki_pos, self.Kd_pos = h.Kp_pos, h.Ki_pos, h.Kd_pos
        self.root_dim = 9 if self.use_pid_control else 6
        self.act_dim = {s: self.root_dim + self.hands[s].n_dofs for s in SIDES}
        cfg["env"]["numActions"] = sum(self.act_dim.values())

        self.act_moving_average = cfg["env"]["actionsMovingAverage"]
        self.translation_scale = cfg["env"]["translationScale"]
        self.orientation_scale = cfg["env"]["orientationScale"]
        self._prop_dump_info = cfg["env"]["propDumpInfo"]

        self.states = {s: {} for s in SIDES}
        self.objs_handles, self.objs_assets = {}, {}
        self.actions = None
        self.dataIndices = cfg["env"]["dataIndices"]
        self.obs_future_length = cfg["env"]["obsFutureLength"]
        self.rollout_state_init = cfg["env"]["rolloutStateInit"]
        self.random_state_init = cfg["env"]["randomStateInit"]
        self.tighten_method = cfg["env"]["tightenMethod"]
        self.tighten_factor = cfg["env"]["tightenFactor"]
        self.tighten_steps = cfg["env"]["tightenSteps"]
        self.sim_device = torch.device(sim_device)

        VecTask.__init__(self, config=cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, display=display, record=record, headless=headless)

        tdim = sum((3 + 3 + 3 + 4 + 4 + 3 + 3 + (self.hands[s].n_bodies - 1) * 9) * self.obs_future_length
                   for s in SIDES)
        self.obs_dict["target"] = torch.zeros((self.num_envs, tdim), device=self.device)
        obs_space = self.obs_space.spaces
        obs_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(tdim,))
        self.obs_space = spaces.Dict(obs_space)

        self.default_dof_pos = {}
        for s in SIDES:
            p = torch.ones(self.hands[s].n_dofs, device=self.device) * np.pi / 36
            if cfg["env"]["dexhand"] == "inspire":
                p[8], p[9] = 0.3, 0.01
            self.default_dof_pos[s] = p

        self._eval_init()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh()

    # ------------------------------------------------------------------ sim / envs
    def _load_hand_asset(self, s):
        o = gymapi.AssetOptions()
        o.thickness = 0.001
        o.angular_damping = 20
        o.linear_damping = 20
        o.max_linear_velocity = 50
        o.max_angular_velocity = 100
        o.fix_base_link = False
        o.disable_gravity = True
        o.flip_visual_attachments = False
        o.collapse_fixed_joints = False
        o.default_dof_drive_mode = gymapi.DOF_MODE_POS
        o.use_mesh_materials = True
        asset = self.gym.load_asset(self.sim, *os.path.split(self.hands[s].urdf_path), o)
        sp = self.gym.get_asset_rigid_shape_properties(asset)
        for e in sp:
            e.friction, e.rolling_friction, e.torsion_friction = 4.0, 0.01, 0.01
        self.gym.set_asset_rigid_shape_properties(asset, sp)
        props = self.gym.get_asset_dof_properties(asset)
        n = self.gym.get_asset_dof_count(asset)
        lo, hi, eff, vel = [], [], [], []
        for i in range(n):
            props["driveMode"][i] = gymapi.DOF_MODE_POS
            props["stiffness"][i] = 500
            props["damping"][i] = 30
            lo.append(props["lower"][i]); hi.append(props["upper"][i])
            eff.append(props["effort"][i]); vel.append(props["velocity"][i])
        T = lambda x: torch.tensor(x, device=self.sim_device, dtype=torch.float)
        self.dof_lower[s], self.dof_upper[s] = T(lo), T(hi)
        self.effort_limits[s], self.speed_limits[s] = T(eff), T(vel)
        return asset, props, n

    def _create_envs(self):
        spacing = 1.0
        env_lower, env_upper = gymapi.Vec3(-spacing, -spacing, 0.0), gymapi.Vec3(spacing, spacing, spacing)

        to = gymapi.AssetOptions(); to.fix_base_link = True
        table_width_offset = 0.2
        table_asset = self.gym.create_box(self.sim, 0.8 + table_width_offset, 1.6, 0.03, to)
        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        self._table_surface_z = table_pos.z + 0.015
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(-0.4, 0, self._table_surface_z + ROBOT_HEIGHT)
        hand_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, 0)

        m2g = np.eye(4)
        m2g[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
        m2g[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(m2g, device=self.sim_device, dtype=torch.float32)

        # ---- data: same index for both hands, loaded from data/handonly/{right,left}
        self.demo = {}
        for s in SIDES:
            dsets = {}
            for dtype in set(ManipDataFactory.dataset_type(i) for i in self.dataIndices):
                dsets[dtype] = ManipDataFactory.create_data(
                    manipdata_type=dtype, side=SIDE_NAME[s], device=self.sim_device,
                    mujoco2gym_transf=self.mujoco2gym_transf, max_seq_len=self.max_episode_length,
                    dexhand=self.hands[s], embodiment=self.cfg["env"]["dexhand"])
            raw = [dsets[ManipDataFactory.dataset_type(idx)][idx]
                   for idx in (self.dataIndices[k % len(self.dataIndices)] for k in range(self.num_envs))]
            self.dexhand = self.hands[s]           # pack_data of the parent reads self.dexhand
            self.demo[s] = self.pack_data(raw)
        self.dexhand = self.hands["rh"]
        self.seq_len = torch.minimum(self.demo["rh"]["seq_len"], self.demo["lh"]["seq_len"])
        self.demo_data = self.demo["rh"]           # parent helpers (camera etc.) expect demo_data

        self.dof_lower, self.dof_upper, self.effort_limits, self.speed_limits = {}, {}, {}, {}
        assets, dprops, ndof = {}, {}, {}
        for s in SIDES:
            assets[s], dprops[s], ndof[s] = self._load_hand_asset(s)
        self.ndof = ndof
        self.dof_slice = {"rh": slice(0, ndof["rh"]), "lh": slice(ndof["rh"], ndof["rh"] + ndof["lh"])}
        nb = sum(self.gym.get_asset_rigid_body_count(a) for a in assets.values()) + 1
        ns = sum(self.gym.get_asset_rigid_shape_count(a) for a in assets.values()) + 1

        self.envs = []
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, nb + 8, ns + 8, True)
            if self.camera_handlers is not None:
                self.camera_handlers.append(self.create_camera(env=env_ptr, isaac_gym=self.gym))
            for s in SIDES:                         # right hand first -> actor 0, left -> actor 1
                a = self.gym.create_actor(env_ptr, assets[s], hand_pose, f"dexhand_{s}", i, FILTER[s])
                self.gym.enable_actor_dof_force_sensors(env_ptr, a)
                self.gym.set_actor_dof_properties(env_ptr, a, dprops[s])
            tp = gymapi.Transform(); tp.p = table_pos
            t = self.gym.create_actor(env_ptr, table_asset, tp, "table", i + self.num_envs, 0b11)
            self.gym.set_rigid_body_color(env_ptr, t, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
        self.init_data()

    def create_camera(self, *, env, isaac_gym):
        cfg = gymapi.CameraProperties()
        cfg.enable_tensors = True
        cfg.width, cfg.height, cfg.horizontal_fov = (1280, 720, 69.4) if self._record else (320, 180, 69.4)
        cam = isaac_gym.create_camera_sensor(env, cfg)
        mid = 0.5 * (self.demo["rh"]["wrist_pos"][0, 0] + self.demo["lh"]["wrist_pos"][0, 0])
        mid = mid.detach().cpu().numpy() if torch.is_tensor(mid) else mid
        isaac_gym.set_camera_location(cam, env, gymapi.Vec3(mid[0] + 0.6, mid[1], mid[2] + 0.35),
                                      gymapi.Vec3(mid[0], mid[1], mid[2]))
        return cam

    def init_data(self):
        env_ptr = self.envs[0]
        self.actor = {s: self.gym.find_actor_handle(env_ptr, f"dexhand_{s}") for s in SIDES}
        self.handles = {s: {k: self.gym.find_actor_rigid_body_handle(env_ptr, self.actor[s], k)
                            for k in self.hands[s].body_names} for s in SIDES}
        self.wrist_idx = {s: self.handles[s][self.hands[s].to_dex("wrist")[0]] for s in SIDES}
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self._root_state = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(
            self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(
            self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(
            self.num_envs, -1, 13)
        self.net_cf = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(
            self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim)).view(self.num_envs, -1)
        self._q, self._qd = self._dof_state[..., 0], self._dof_state[..., 1]

        nrb = self._rigid_body_state.shape[1]
        self.apply_forces = torch.zeros((self.num_envs, nrb, 3), device=self.device)
        self.apply_torque = torch.zeros((self.num_envs, nrb, 3), device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.curr_targets = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        if self.use_pid_control:
            z = lambda: {s: torch.zeros((self.num_envs, 3), device=self.device) for s in SIDES}
            self.prev_pos_error, self.prev_rot_error = z(), z()
            self.pos_error_integral, self.rot_error_integral = z(), z()
        self._global_idx = {s: torch.tensor(
            [self.gym.find_actor_index(env, f"dexhand_{s}", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32, device=self.sim_device) for s in SIDES}
        self.hand_rb_idx = torch.tensor(sorted(set(v for s in SIDES for v in self.handles[s].values())),
                                        device=self.device, dtype=torch.long)

    def allocate_buffers(self):
        VecTask.allocate_buffers(self)

    # ------------------------------------------------------------------ states / obs
    def _update_states(self):
        for s in SIDES:
            q, qd = self._q[:, self.dof_slice[s]], self._qd[:, self.dof_slice[s]]
            self.states[s].update({"q": q, "cos_q": torch.cos(q), "sin_q": torch.sin(q), "dq": qd,
                                   "base_state": self._root_state[:, self.actor[s], :]})
            self.states[s]["joints_state"] = torch.stack(
                [self._rigid_body_state[:, self.handles[s][k], :][:, :10] for k in self.hands[s].body_names], dim=1)

    def _target_obs(self, s):
        D, st = self.demo[s], self.states[s]
        nE, nT = D["wrist_pos"].shape[:2]
        nF = self.obs_future_length
        idx = torch.clamp(self.progress_buf + 1, torch.zeros_like(self.seq_len), self.seq_len - 1)
        idx = torch.stack([idx + t for t in range(nF)], dim=-1)

        def ind(data):
            rest = data.shape[2:]
            e = idx
            for _ in rest:
                e = e.unsqueeze(-1)
            return torch.gather(data, 1, e.expand(-1, -1, *rest))

        bs = st["base_state"]
        o = {}
        tw = ind(D["wrist_pos"])
        o["delta_wrist_pos"] = (tw - bs[:, None, :3]).reshape(nE, -1)
        tv = ind(D["wrist_velocity"])
        o["wrist_vel"] = tv.reshape(nE, -1)
        o["delta_wrist_vel"] = (tv - bs[:, None, 7:10]).reshape(nE, -1)
        tq = aa_to_quat(ind(D["wrist_rot"]).reshape(nE * nF, -1))[:, [1, 2, 3, 0]]
        o["delta_wrist_quat"] = quat_mul(bs[:, None, 3:7].repeat(1, nF, 1).reshape(nE * nF, -1),
                                         quat_conjugate(tq)).reshape(nE, -1)
        o["wrist_quat"] = tq.reshape(nE, -1)
        ta = ind(D["wrist_angular_velocity"])
        o["wrist_ang_vel"] = ta.reshape(nE, -1)
        o["delta_wrist_ang_vel"] = (ta - bs[:, None, 10:13]).reshape(nE, -1)
        tj = ind(D["mano_joints"]).reshape(nE, nF, -1, 3)
        o["delta_joints_pos"] = (tj - st["joints_state"][:, None, 1:, :3]).reshape(nE, -1)
        tjv = ind(D["mano_joints_velocity"]).reshape(nE, nF, -1, 3)
        o["joints_vel"] = tjv.reshape(nE, -1)
        o["delta_joints_vel"] = (tjv - st["joints_state"][:, None, 1:, 7:10]).reshape(nE, -1)
        return torch.cat([o[k] for k in ("delta_wrist_pos", "wrist_vel", "delta_wrist_vel", "wrist_quat",
                                         "delta_wrist_quat", "wrist_ang_vel", "delta_wrist_ang_vel",
                                         "delta_joints_pos", "joints_vel", "delta_joints_vel")], dim=-1)

    def compute_observations(self):
        self._refresh()
        prop, priv, tgt = [], [], []
        for s in SIDES:
            st = self.states[s]
            for ob in self._obs_keys:
                if ob == "base_state":
                    prop.append(torch.cat([torch.zeros_like(st[ob][:, :3]), st[ob][:, 3:]], dim=-1))
                else:
                    prop.append(st[ob])
            for ob in self._privileged_obs_keys:
                priv.append(st[ob])
            tgt.append(self._target_obs(s))
        self.obs_dict["proprioception"][:] = torch.cat(prop, dim=-1)
        if len(priv) > 0:
            self.obs_dict["privileged"][:] = torch.cat(priv, dim=-1)
        self.obs_dict["target"][:] = torch.cat(tgt, dim=-1)
        return self.obs_dict

    # ------------------------------------------------------------------ reward
    def _scale_factor(self):
        if not self.training:
            return float(self.cfg["env"].get("testScaleFactor", 1.0))
        last = self.gym.get_frame_count(self.sim)
        if self.tighten_method == "exp_decay":
            return (np.e * 2) ** (-1 * last / self.tighten_steps) * (1 - self.tighten_factor) + self.tighten_factor
        if self.tighten_method == "linear_decay":
            return 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last, self.tighten_steps)
        if self.tighten_method == "const":
            return self.tighten_factor
        return 1.0

    def compute_reward(self, actions):
        ar = torch.arange(self.num_envs, device=self.device)
        cur = torch.minimum(self.progress_buf, self.seq_len - 1)
        max_length = torch.clip(self.seq_len, 0, self.max_episode_length).float()
        sf = self._scale_factor()
        out, self.target_state = {}, {}
        for s in SIDES:
            D, st = self.demo[s], self.states[s]
            ts = {"wrist_pos": D["wrist_pos"][ar, cur],
                  "wrist_quat": aa_to_quat(D["wrist_rot"][ar, cur])[:, [1, 2, 3, 0]],
                  "wrist_vel": D["wrist_velocity"][ar, cur],
                  "wrist_ang_vel": D["wrist_angular_velocity"][ar, cur],
                  "joints_pos": D["mano_joints"][ar, cur].reshape(self.num_envs, -1, 3),
                  "joints_vel": D["mano_joints_velocity"][ar, cur].reshape(self.num_envs, -1, 3)}
            ts["power"] = torch.abs(self.dof_force[:, self.dof_slice[s]] * st["dq"]).sum(-1)
            w = self.wrist_idx[s]
            ts["wrist_power"] = torch.abs((self.apply_forces[:, w] * st["base_state"][:, 7:10]).sum(-1)) + \
                torch.abs((self.apply_torque[:, w] * st["base_state"][:, 10:13]).sum(-1))
            self.target_state[s] = ts
            out[s] = compute_imitation_reward(self.reset_buf, self.progress_buf, self.running_progress_buf,
                                              self.curr_targets, st, ts, max_length, sf, self.hands[s].weight_idx)
        rew = out["rh"][0] + out["lh"][0]
        succ = out["rh"][2] & out["lh"][2]
        fail = out["rh"][3] | out["lh"][3]
        self.rew_buf[:] = rew
        self.success_buf[:] = succ
        self.failure_buf[:] = fail
        self.reset_buf[:] = torch.where(succ | fail, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reward_dict = {}
        for s in SIDES:
            for k, v in out[s][4].items():
                self.reward_dict[f"{s}_{k}"] = v
        cf = torch.norm(self.net_cf[:, self.hand_rb_idx], dim=-1).sum(-1)
        self.reward_dict["contact_force"] = cf
        self.total_rew_buf += self.rew_buf
        self._eval_step(cf)

    # ------------------------------------------------------------------ reset
    def _reset_default(self, env_ids):
        n = len(env_ids)
        if self.random_state_init:
            seq_idx = torch.floor(self.seq_len[env_ids] * 0.99 * torch.rand(n, device=self.device)).long()
        else:
            seq_idx = torch.zeros(n, dtype=torch.long, device=self.device)
        for s in SIDES:
            lo, hi = self.dof_lower[s], self.dof_upper[s]
            dp = self.default_dof_pos[s][None].repeat(n, 1)
            dp = torch.clamp(dp + torch.randn_like(dp) * ((hi - lo) / 8)[None], lo[None], hi[None])
            dv = torch.clamp(torch.randn_like(dp) * 0.1, -self.speed_limits[s][None], self.speed_limits[s][None])
            D = self.demo[s]
            wp = D["wrist_pos"][env_ids, seq_idx]
            wp = wp + torch.randn_like(wp) * 0.01
            wr = aa_to_rotmat(D["wrist_rot"][env_ids, seq_idx])
            nr = torch.rand(n, 3, device=self.device)
            nr = aa_to_rotmat(nr / torch.norm(nr, dim=-1, keepdim=True) * torch.randn(n, 1, device=self.device)
                              * (np.pi / 18))
            wq = rotmat_to_quat(nr @ wr)[:, [1, 2, 3, 0]]
            wv = D["wrist_velocity"][env_ids, seq_idx]
            wv = wv + torch.randn_like(wv) * 0.01
            wa = D["wrist_angular_velocity"][env_ids, seq_idx]
            wa = wa + torch.randn_like(wa) * 0.01
            self._root_state[env_ids, self.actor[s], :] = torch.cat([wp, wq, wv, wa], dim=-1)
            sl = self.dof_slice[s]
            self._q[env_ids, sl] = dp
            self._qd[env_ids, sl] = dv
            self._pos_control[env_ids, sl] = dp
            if self.use_pid_control:
                for b in (self.prev_pos_error, self.prev_rot_error, self.pos_error_integral, self.rot_error_integral):
                    b[s][env_ids] = 0
        ids = torch.cat([self._global_idx[s][env_ids] for s in SIDES]).to(torch.int32)
        ui = gymtorch.unwrap_tensor(ids)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state), ui, len(ids))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_state), ui, len(ids))
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._pos_control), ui,
                                                        len(ids))
        self.progress_buf[env_ids] = seq_idx
        self.running_progress_buf[env_ids] = 0
        for b in (self.reset_buf, self.success_buf, self.failure_buf, self.error_buf, self.total_rew_buf):
            b[env_ids] = 0
        self.apply_forces[env_ids] = 0
        self.apply_torque[env_ids] = 0
        self.curr_targets[env_ids] = 0
        self.prev_targets[env_ids] = 0

    # ------------------------------------------------------------------ actions
    def pre_physics_step(self, actions):
        ema = self.act_moving_average
        off = 0
        for s in SIDES:
            a = actions[:, off: off + self.act_dim[s]]
            off += self.act_dim[s]
            sl, rd = self.dof_slice[s], self.root_dim
            tgt = torch_jit_utils.scale(torch.clamp(a[:, rd: rd + self.ndof[s]], -1, 1),
                                        self.dof_lower[s], self.dof_upper[s])
            tgt = ema * tgt + (1.0 - ema) * self.prev_targets[:, sl]
            self.curr_targets[:, sl] = torch_jit_utils.tensor_clamp(tgt, self.dof_lower[s], self.dof_upper[s])
            w = self.wrist_idx[s]
            if self.use_pid_control:
                pe = a[:, :3]
                self.pos_error_integral[s] = torch.clamp(self.pos_error_integral[s] + pe * self.dt, -1, 1)
                f = self.Kp_pos * pe + self.Ki_pos * self.pos_error_integral[s] + \
                    self.Kd_pos * (pe - self.prev_pos_error[s]) / self.dt
                self.prev_pos_error[s] = pe
                re = rot6d_to_aa(a[:, 3:9])
                self.rot_error_integral[s] = torch.clamp(self.rot_error_integral[s] + re * self.dt, -1, 1)
                tq = self.Kp_rot * re + self.Ki_rot * self.rot_error_integral[s] + \
                    self.Kd_rot * (re - self.prev_rot_error[s]) / self.dt
                self.prev_rot_error[s] = re
            else:
                f = a[:, 0:3] * self.dt * self.translation_scale * 500
                tq = a[:, 3:6] * self.dt * self.orientation_scale * 200
            self.apply_forces[:, w, :] = ema * f + (1.0 - ema) * self.apply_forces[:, w, :]
            self.apply_torque[:, w, :] = ema * tq + (1.0 - ema) * self.apply_torque[:, w, :]
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
                                                gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)
        self.prev_targets[:] = self.curr_targets[:]
        self._pos_control[:] = self.prev_targets[:]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    # ------------------------------------------------------------------ evaluation (EVAL_OUT)
    def _eval_init(self):
        self._eval_out = os.environ.get("EVAL_OUT", "") if not self.training else ""
        if not self._eval_out:
            return
        n, dev = self.num_envs, self.device
        z = lambda: torch.zeros(n, device=dev)
        self._ev = {"done": torch.zeros(n, dtype=torch.bool, device=dev),
                    "success": torch.zeros(n, dtype=torch.bool, device=dev),
                    "end_frame": torch.zeros(n, dtype=torch.long, device=dev),
                    "steps": z(), "rew": z(), "jerk_n": z()}
        self._ex = {k: z() for k in ("contact_frac", "contact_force_N", "contact_force_max_N")}
        for s in SIDES:
            for k in ("power_joint_W", "power_wrist_W", "err_tip_mm", "err_kp_mm", "err_wrist_pos_mm",
                      "err_wrist_rot_deg", "jerk_tip", "jerk_wrist"):
                self._ex[f"{s}_{k}"] = z()
        self._ev_terms, self._ev_hist = {}, []
        self._ev_K = min(int(os.environ.get("EVAL_TRAJ_ENVS", "4")), n)
        self._ev_traj = {k: [] for k in ("rb", "kp_rh", "kp_lh", "kp_ref_rh", "kp_ref_lh", "contact", "alive")}
        # per-frame recorder for the scale sweep (EVAL_FRAMES=1, see sweep_scale.py)
        self._ev_frames = {} if os.environ.get("EVAL_FRAMES", "0") == "1" else None
        # camera recording of env 0 (EVAL_RENDER=1, needs capture_video=true; see render_eval.sh)
        self._ev_render = (os.environ.get("EVAL_RENDER", "0") == "1"
                           and getattr(self, "camera_obs", None) is not None and len(self.camera_obs) > 0)
        self._ev_rframes = []
        if self._ev_render:
            import math
            T0 = int(self.seq_len[0])
            wp = torch.cat([self.demo[s]["wrist_pos"][0][:T0] for s in SIDES]).float().cpu().numpy()
            lo, hi = wp.min(0), wp.max(0)
            ctr = (lo + hi) / 2
            ext = float(np.linalg.norm(hi - lo)) + 0.35
            az, el = [math.radians(float(v)) for v in os.environ.get("EVAL_CAM", "0,30").split(",")]
            vfov = 2 * math.atan(math.tan(math.radians(69.4) / 2) * 720 / 1280)
            dist = ext / 2 / math.tan(vfov / 2) * 1.05
            eye = ctr + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            self.gym.set_camera_location(self.camera_handlers[0], self.envs[0],
                                         gymapi.Vec3(*[float(v) for v in eye]), gymapi.Vec3(*[float(v) for v in ctr]))
            print(f"[eval] recording env 0 camera, center {ctr.round(3)}, distance {dist:.2f} m")
            # hide the table (it is in its own collision group, so this does not change the physics);
            # EVAL_RENDER_TABLE=1 keeps it
            if os.environ.get("EVAL_RENDER_TABLE", "0") != "1":
                ids = []
                for ei, env in enumerate(self.envs):
                    h = self.gym.find_actor_handle(env, "table")
                    if h >= 0:
                        self._root_state[ei, h, 2] = -10.0
                        ids.append(ei)
                if ids:
                    # full (non-indexed) write: an indexed call here would replace the pending indexed root-state write
                    # of the initial reset_idx (IsaacGym applies only the last indexed call before a sim step)
                    self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))
        print(f"[eval] BiH first-episode evaluator on, {n} envs -> {self._eval_out}")

    @torch.no_grad()
    def _eval_step(self, cf):
        if not self._eval_out:
            return
        E, X = self._ev, self._ex
        act = ~E["done"]
        af = act.float()
        E["steps"] += af
        E["rew"] += self.rew_buf * af
        for k, v in self.reward_dict.items():
            self._ev_terms.setdefault(k, torch.zeros(self.num_envs, device=self.device))
            self._ev_terms[k] += v.detach() * af
        X["contact_frac"] += (cf > 0.1).float() * af
        X["contact_force_N"] += cf * af
        X["contact_force_max_N"] = torch.where(act, torch.maximum(X["contact_force_max_N"], cf),
                                               X["contact_force_max_N"])
        cdt = self.dt * getattr(self, "control_freq_inv", 1)
        snap = {}
        for s in SIDES:
            st, ts, h = self.states[s], self.target_state[s], self.hands[s]
            bs = st["base_state"]
            kp, kref = st["joints_state"][:, 1:, :3], ts["joints_pos"]
            dist = torch.norm(kp - kref, dim=-1)
            tip = [i - 1 for t in ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
                   for i in h.weight_idx[t]]
            dq_ = quat_mul(ts["wrist_quat"], quat_conjugate(bs[:, 3:7]))
            X[f"{s}_power_joint_W"] += ts["power"] * af
            X[f"{s}_power_wrist_W"] += ts["wrist_power"] * af
            X[f"{s}_err_tip_mm"] += 1000 * dist[:, tip].mean(-1) * af
            X[f"{s}_err_kp_mm"] += 1000 * dist.mean(-1) * af
            X[f"{s}_err_wrist_pos_mm"] += 1000 * torch.norm(ts["wrist_pos"] - bs[:, :3], dim=-1) * af
            X[f"{s}_err_wrist_rot_deg"] += torch.rad2deg(2 * torch.acos(torch.clamp(dq_[:, 3].abs(), max=1.0))) * af
            snap[s] = (kp[:, tip].clone(), bs[:, :3].clone())
            if self._ev_frames is not None:
                keys = ("thumb_tip", "index_tip", "middle_tip", "pinky_tip", "ring_tip",
                        "level_1_joints", "level_2_joints")
                base = (0.04, 0.045, 0.05, 0.06, 0.06, 0.07, 0.08)          # thresholds = base / 0.7 * scale
                F = self._ev_frames
                F.setdefault(f"ratio_{s}", []).append(torch.stack(
                    [dist[:, [i - 1 for i in h.weight_idx[k]]].mean(-1) / (b / 0.7) for k, b in zip(keys, base)],
                    -1).cpu())
                for k, v in (("power_joint_W", ts["power"]), ("power_wrist_W", ts["wrist_power"]),
                             ("err_tip_mm", 1000 * dist[:, tip].mean(-1)), ("err_kp_mm", 1000 * dist.mean(-1)),
                             ("err_wrist_pos_mm", 1000 * torch.norm(ts["wrist_pos"] - bs[:, :3], dim=-1)),
                             ("err_wrist_rot_deg",
                              torch.rad2deg(2 * torch.acos(torch.clamp(dq_[:, 3].abs(), max=1.0))))):
                    F.setdefault(f"m_{s}_{k}", []).append(v.detach().cpu())
        jerk = {}
        if len(self._ev_hist) == 3:
            for s in SIDES:
                (t3, w3), (t2, w2), (t1, w1), (t0, w0) = [h[s] for h in self._ev_hist] + [snap[s]]
                jerk[f"{s}_jerk_tip"] = torch.norm(t0 - 3 * t1 + 3 * t2 - t3, dim=-1).mean(-1) / cdt ** 3
                jerk[f"{s}_jerk_wrist"] = torch.norm(w0 - 3 * w1 + 3 * w2 - w3, dim=-1) / cdt ** 3
                X[f"{s}_jerk_tip"] += jerk[f"{s}_jerk_tip"] * af
                X[f"{s}_jerk_wrist"] += jerk[f"{s}_jerk_wrist"] * af
            E["jerk_n"] += af
        if self._ev_frames is not None:
            F, nan = self._ev_frames, torch.full_like(cf, float("nan"))
            for s in SIDES:
                for k in ("jerk_tip", "jerk_wrist"):
                    F.setdefault(f"m_{s}_{k}", []).append(jerk.get(f"{s}_{k}", nan).detach().cpu())
            F.setdefault("m_contact_force_N", []).append(cf.detach().cpu())
            F.setdefault("run", []).append(self.running_progress_buf.clone().cpu())
            F.setdefault("alive", []).append(act.clone().cpu())
        self._ev_hist = (self._ev_hist + [snap])[-3:]
        K = self._ev_K
        if K > 0:
            T = self._ev_traj
            T["rb"].append(self._rigid_body_state[:K].cpu()); T["contact"].append(cf[:K].cpu())
            T["alive"].append(act[:K].cpu())
            for s in SIDES:
                T[f"kp_{s}"].append(self.states[s]["joints_state"][:K, :, :3].cpu())
                T[f"kp_ref_{s}"].append(self.target_state[s]["joints_pos"][:K].cpu())
        if self._ev_render and bool(act[0]):
            self._ev_rframes.append(self.camera_obs[0][..., :3].clone().cpu().numpy())
        fin = act & self.reset_buf.bool()
        E["success"][fin] = self.success_buf[fin].bool()
        E["end_frame"][fin] = self.progress_buf[fin]
        E["done"] |= fin
        if bool(E["done"].all()):
            self._eval_write()

    def _eval_write(self):
        E, X = self._ev, self._ex
        steps = E["steps"].clamp(min=1)
        jn = E["jerk_n"].clamp(min=1)
        T = int(self.seq_len.max())
        res = {"num_envs": int(self.num_envs), "clip_len": T,
               "success_rate": float(E["success"].float().mean()),
               "frac_tracked_mean": float((E["end_frame"].float() / (T - 4)).clamp(max=1).mean()),
               "end_frame_mean": float(E["end_frame"].float().mean()),
               "reward_per_step_mean": float((E["rew"] / steps).mean()),
               "x_contact_force_max_N": float(X["contact_force_max_N"].mean())}
        for k, v in X.items():
            if k == "contact_force_max_N":
                continue
            res[f"x_{k}"] = float((v / (jn if "jerk" in k else steps)).mean())
        for k in ("power_joint_W", "power_wrist_W", "err_tip_mm", "err_kp_mm", "jerk_tip", "jerk_wrist"):
            # both-hand summary: power summed over the two hands, errors / jerk averaged
            res[f"x_{k}"] = (res[f"x_rh_{k}"] + res[f"x_lh_{k}"]) * (1.0 if "power" in k else 0.5)
        res.update({f"{k}_per_step_mean": float((v / steps).mean()) for k, v in self._ev_terms.items()})
        os.makedirs(os.path.dirname(self._eval_out) or ".", exist_ok=True)
        np.savez(self._eval_out + ".npz", success=E["success"].cpu().numpy(), end_frame=E["end_frame"].cpu().numpy(),
                 steps=E["steps"].cpu().numpy(), **{k: v.cpu().numpy() for k, v in X.items()})
        if self._ev_K > 0:
            np.savez(self._eval_out + "_traj.npz", fps=60,
                     **{k: torch.stack(v, 0).numpy() for k, v in self._ev_traj.items()},
                     **{f"body_names_{s}": np.array(self.hands[s].body_names) for s in SIDES},
                     **{f"body_rb_idx_{s}": np.array([self.handles[s][b] for b in self.hands[s].body_names])
                        for s in SIDES})
        if self._ev_render and len(self._ev_rframes) > 0:
            import imageio
            imageio.mimsave(self._eval_out + "_render.mp4", self._ev_rframes, fps=60, macro_block_size=1)
            print(f"[eval] wrote {self._eval_out}_render.mp4 ({len(self._ev_rframes)} frames)")
        if self._ev_frames is not None:
            np.savez_compressed(self._eval_out + "_frames.npz", success=E["success"].cpu().numpy(),
                                end_frame=E["end_frame"].cpu().numpy(),
                                **{k: torch.stack(v, 0).numpy() for k, v in self._ev_frames.items()})
        with open(self._eval_out + ".json", "w") as f:
            json.dump(res, f, indent=2)
        print("[eval] " + json.dumps(res))
        os._exit(0)