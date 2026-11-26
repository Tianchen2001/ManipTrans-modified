import gym
import torch
import wandb
from rl_games.common.algo_observer import AlgoObserver

from lib.utils.utils import retry
from lib.utils.reformat import omegaconf_to_dict

import os, imageio


class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb

        wandb_unique_id = f"uid_{experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")

        cfg = self.cfg

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method="fork"),
            )

            if cfg.wandb_logcode_dir:
                wandb.run.log_code(root=cfg.wandb_logcode_dir)
                print("wandb running directory........", wandb.run.dir)

        print("Initializing WandB...")
        try:
            init_wandb()
        except Exception as exc:
            print(f"Could not initialize WandB! {exc}")

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)

    def after_init(self, algo):
        """Mirror every '<tag>/iter' scalar (and add_scalars dict) to wandb live; x-axis = training iteration."""
        import atexit
        import numpy as np
        import wandb

        if wandb.run is None or getattr(algo, "writer", None) is None:
            return
        wandb.define_metric("iter")
        wandb.define_metric("*", step_metric="iter")
        w, buf, cur = algo.writer, {}, [None]

        def flush():
            if buf and wandb.run is not None:
                wandb.log({**buf, "iter": cur[0]})
                buf.clear()

        def push(it, key, val):
            if cur[0] is not None and it != cur[0]:
                flush()
            cur[0] = it
            try:
                if hasattr(val, "detach"):
                    val = val.detach().cpu().numpy()
                buf[key] = float(np.asarray(val, dtype=float).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                pass

        orig_scalar, orig_scalars = w.add_scalar, w.add_scalars

        def add_scalar(tag, value, global_step=None, *args, **kwargs):
            orig_scalar(tag, value, global_step, *args, **kwargs)
            if tag.endswith("/iter") and global_step is not None:
                push(int(global_step), tag[: -len("/iter")], value)

        def add_scalars(main_tag, values, global_step=None, *args, **kwargs):
            orig_scalars(main_tag, values, global_step, *args, **kwargs)
            if main_tag.endswith("/iter") and global_step is not None:
                for k, v in values.items():
                    push(int(global_step), f"{main_tag[: -len('/iter')]}/{k}", v)

        w.add_scalar, w.add_scalars = add_scalar, add_scalars
        atexit.register(flush)


class WandbVideoCaptureWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        n_parallel_recorders: int = 1,
        n_successful_videos_to_record: int = 50,
        out_dir="videos",
    ):
        super().__init__(env)
        n_parallel_recorders = min(n_parallel_recorders, env.num_envs)
        self._n_recorders = n_parallel_recorders
        self._videos = [[] for _ in range(n_parallel_recorders)]
        self._rcd_idxs = [i for i in range(env.num_envs) if i % (env.num_envs // n_parallel_recorders) == 0][
            :n_parallel_recorders
        ]
        self._n_video_saved = 0
        self._n_successful_video_saved = 0
        self._n_successful_videos_to_record = n_successful_videos_to_record

        self._out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def reset(self, **kwargs):
        self._videos = [[] for _ in range(self._n_recorders)]
        return super().reset(**kwargs)

    # def step(self, action):
    #     obs, reward, done, info = super().step(action)
    #     for i, idx in enumerate(self._rcd_idxs):
    #         self._videos[i].append(self.env.camera_obs[idx].clone())
    #     if torch.any(done):
    #         for i, idx in enumerate(self._rcd_idxs):
    #             if done[idx]:
    #                 video = torch.stack(self._videos[i])[..., :-1]  # (T, H, W, C), RGBA -> RGB
    #                 video = video.to(dtype=torch.uint8)
    #                 video = video.permute(0, 3, 1, 2).detach().cpu().numpy()  # (T, C, H, W)
    #                 video = wandb.Video(video, fps=10, format="mp4")
    #                 succeeded = self.env.success_buf
    #                 failed = self.env.failure_buf
    #                 status = "timeout"
    #                 if succeeded[idx]:
    #                     status = "success"
    #                     self._n_successful_video_saved += 1
    #                 elif failed[idx]:
    #                     status = "failure"
    #                 wandb.log({f"test_video/video-{self._n_video_saved}_{status}": video})
    #                 self._n_video_saved += 1
    #                 self._videos[i] = []
    #                 if self._n_successful_video_saved >= self._n_successful_videos_to_record:
    #                     exit()
    #     return obs, reward, done, info
    def step(self, action):
        obs, reward, done, info = super().step(action)
        for i, idx in enumerate(self._rcd_idxs):
            self._videos[i].append(self.env.camera_obs[idx][..., :3].clone().cpu())
        if torch.any(done):
            for i, idx in enumerate(self._rcd_idxs):
                if not done[idx]:
                    continue
                status = "success" if self.env.success_buf[idx] else ("failure" if self.env.failure_buf[idx] else "timeout")
                frames = torch.stack(self._videos[i]).to(torch.uint8).numpy()        # (T, H, W, 3)
                path = os.path.join(self._out_dir, f"video-{self._n_video_saved:03d}_{status}_T{len(frames)}.mp4")
                imageio.mimsave(path, frames, fps=60, macro_block_size=1)
                print(f"saved {path}")
                if wandb.run is not None:
                    wandb.log({f"test_video/video-{self._n_video_saved}_{status}": wandb.Video(path, format="mp4")})
                if status == "success":
                    self._n_successful_video_saved += 1
                self._n_video_saved += 1
                self._videos[i] = []
                if self._n_successful_video_saved >= self._n_successful_videos_to_record:
                    exit()
        return obs, reward, done, info
