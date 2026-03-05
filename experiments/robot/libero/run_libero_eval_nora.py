"""
run_libero_eval_nora.py

Runs Nora-1.5 in a LIBERO simulation environment.
"""

import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import draccus
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import trange
import wandb

script_dir = Path(__file__).parent
openvla_root = script_dir.parent.parent.parent
sys.path.append(str(openvla_root))
sys.path.append(str(openvla_root / "LIBERO"))
sys.path.append(str(script_dir))

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video_given_path,
)
from experiments.robot.libero.inference.modelling_expert import VLAWithExpert
from experiments.robot.robot_utils import (
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.viz_utils import PoseCumulator


@dataclass
class GenerateConfig:
    # Model
    model_family: str = "nora"
    pretrained_checkpoint: str = "declare-lab/nora-1.5-libero"
    nora_num_steps: int = 10

    # LIBERO
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    task_start_index: Optional[int] = None
    task_end_index: Optional[int] = None
    resume: bool = False

    # Action post-processing
    normalize_gripper: bool = False
    invert_gripper: bool = False
    clip_actions: bool = False
    action_clip_value: float = 1.0

    # Logging
    run_id_note: Optional[str] = "nora"
    save_root: str = "./rollouts_nora"
    save_logs: bool = True
    use_wandb: bool = False
    wandb_project: str = "nora-libero"
    wandb_entity: str = "anonymous"
    wandb_dir: Optional[str] = None
    seed: int = 7


def _select_action(raw_action: np.ndarray) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32)
    if action.ndim == 1 and action.shape[0] == 7:
        return action
    if action.ndim >= 2 and action.shape[-1] == 7:
        return action.reshape(-1, 7)[0]
    raise ValueError(f"Unexpected Nora action shape: {action.shape}")


def _build_observation(obs, img):
    return {
        "full_image": img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }


@draccus.wrap()
def eval_libero_nora(cfg: GenerateConfig) -> None:
    set_seed_everywhere(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VLAWithExpert.from_pretrained(cfg.pretrained_checkpoint, device=device)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    run_group = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}"
    if cfg.run_id_note:
        run_group += f"--{cfg.run_id_note}"

    save_folder = Path(cfg.save_root) / f"{cfg.run_id_note}" / f"{cfg.task_suite_name}"
    save_folder.mkdir(parents=True, exist_ok=True)

    total_episodes, total_successes = 0, 0
    task_start_index = cfg.task_start_index if cfg.task_start_index is not None else 0
    task_end_index = cfg.task_end_index + 1 if cfg.task_end_index is not None else num_tasks_in_suite

    for task_id in trange(task_start_index, task_end_index):
        if cfg.resume:
            existing_results = list(save_folder.glob(f"task{task_id}--ep*--succ*.csv"))
            max_existing_episode = (
                max([int(str(p).split("--ep")[1].split("--")[0]) for p in existing_results]) if existing_results else -1
            )
            if max_existing_episode + 1 >= cfg.num_trials_per_task:
                continue
            start_episode = max_existing_episode + 1
        else:
            start_episode = 0

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in trange(start_episode, cfg.num_trials_per_task):
            if cfg.use_wandb:
                wandb.init(
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    group=run_group,
                    name=f"task-{task_id}--episode-{episode_idx}",
                    dir=cfg.wandb_dir,
                    config=asdict(cfg),
                )

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            if "libero_spatial" in cfg.task_suite_name:
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400
            else:
                raise ValueError(f"Unsupported task suite: {cfg.task_suite_name}")

            logs = defaultdict(list)
            logs_cum = defaultdict(float)
            pose_cumulator = PoseCumulator()
            logs_to_dump = []
            action_episode = []
            probs_episode = []
            logits_episode = []
            hidden_states_episode = []
            done = False
            reward = 0.0

            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                img = get_libero_image(obs, resize_size)
                observation = _build_observation(obs, img)
                pil_image = Image.fromarray(observation["full_image"]).convert("RGB")
                raw_output = model.sample_actions(
                    pil_image, task_description,
                    num_steps=cfg.nora_num_steps,
                    return_hidden_states=True,
                )
                raw_action, extra_outputs = raw_output
                action = _select_action(raw_action)

                # Collect VLM hidden states (last token of last layer)
                if extra_outputs["hidden_states"] is not None:
                    hidden_states_episode.append(extra_outputs["hidden_states"][-1, :])  # (hidden_dim,)

                # Collect VLM logits (last token)
                if extra_outputs["logits"] is not None:
                    logits_episode.append(extra_outputs["logits"][-1, :].numpy())  # (vocab_size,)

                # Collect velocity norms as proxy for "probs" (denoising confidence)
                # velocity_norms shape: (num_steps, action_chunk_length)
                probs_episode.append(extra_outputs["velocity_norms"])

                if cfg.clip_actions:
                    action = np.clip(action, -cfg.action_clip_value, cfg.action_clip_value)
                if cfg.normalize_gripper:
                    action = normalize_gripper_action(action, binarize=True)
                if cfg.invert_gripper:
                    action = invert_gripper_action(action)

                # Compute action norms
                dpos = float(np.linalg.norm(action[:3]))
                drot = float(np.linalg.norm(action[3:6]))
                logs_cum["dpos"] += dpos
                logs_cum["drot"] += drot

                # Track end-effector pose
                pose_cumulator.update(obs["robot0_eef_pos"], obs["robot0_eef_quat"])

                to_be_logged = {
                    "action/timestep": t,
                    "action/dx": float(action[0]),
                    "action/dy": float(action[1]),
                    "action/dz": float(action[2]),
                    "action/droll": float(action[3]),
                    "action/dpitch": float(action[4]),
                    "action/dyaw": float(action[5]),
                    "action/dgripper": float(action[6]),
                    "action/dpos": dpos,
                    "action/drot": drot,
                    "action/cum_dpos": logs_cum["dpos"],
                    "action/cum_drot": logs_cum["drot"],
                    "pose/cum_pos": pose_cumulator.cum_pos,
                    "pose/cum_rot": pose_cumulator.cum_rot,
                }

                # Convert numpy values to float
                for k, v in to_be_logged.items():
                    if isinstance(v, np.floating):
                        to_be_logged[k] = float(v)

                logs_to_dump.append(to_be_logged)
                if cfg.use_wandb:
                    wandb.log(to_be_logged)

                replay_images.append(img)
                action_episode.append(action.copy())
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    break
                t += 1

            if cfg.use_wandb:
                wandb.log({"ep/success": float(done)})
                wandb.finish(quiet=True)

            task_episodes += 1
            total_episodes += 1
            task_successes += int(done)
            total_successes += int(done)

            mp4_path = save_folder / f"task{task_id}--ep{episode_idx}--succ{int(done)}.mp4"
            save_rollout_video_given_path(replay_images, mp4_path)

            # Always save episode data as pkl
            save_dict = {
                "hidden_states": torch.stack(hidden_states_episode, dim=0) if hidden_states_episode else torch.zeros((0, 1)),  # (T, hidden_dim)
                "action": np.stack(action_episode) if action_episode else np.zeros((0, 7), dtype=np.float32),  # (T, 7)
                "probs": np.stack(probs_episode) if probs_episode else None,  # (T, num_steps, action_chunk_length) — velocity norms
                "logits": np.stack(logits_episode) if logits_episode else None,  # (T, vocab_size) — VLM logits at last token
                "task_suite_name": cfg.task_suite_name,
                "task_id": task_id,
                "task_description": task_description,
                "eposide_idx": episode_idx,
                "episode_success": done,
                "mp4_path": str(mp4_path),
            }
            pkl_path = mp4_path.with_suffix(".pkl")
            pickle.dump(save_dict, open(pkl_path, "wb"))
            print(f"Saved episode data at {pkl_path}")

            if cfg.save_logs:
                pd.DataFrame(logs_to_dump).to_csv(mp4_path.with_suffix(".csv"), index=False)

        if task_episodes > 0:
            print(f"Task {task_id} success rate: {task_successes / task_episodes:.3f}")
            print(f"Total success rate: {total_successes / total_episodes:.3f}")

        env.close()


if __name__ == "__main__":
    eval_libero_nora()
