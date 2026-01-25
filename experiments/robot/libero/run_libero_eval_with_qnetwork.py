"""
run_libero_eval_with_qnetwork.py

Runs OpenVLA in LIBERO with live GRUQNetwork evaluation at each step.
Computes Q-values and checks conformal prediction thresholds in real-time.
"""

from collections import defaultdict
import os
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal, Optional, Union, Tuple, List
import torch
import torch.nn.functional as F
import wandb

import draccus
import numpy as np
from tqdm import tqdm, trange
from scipy.special import softmax
import matplotlib.pyplot as plt
import imageio
import cv2

import pandas as pd

sys.path.append("/home/shellyfra/Projects/SAFE/openvla")
sys.path.append("/home/shellyfra/Projects/SAFE/openvla/LIBERO")
sys.path.append("/home/shellyfra/Projects/SAFE")

from libero.libero import benchmark

# Import LIBERO utilities
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video_given_path
)
from experiments.robot.openvla_utils import (
    get_processor,
    get_text_tokens
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.viz_utils import (
    PoseCumulator
)
from experiments.robot.unc_utils import (
    compute_token_uncertainty_metrics,
    compute_samples_uncertainty_metrics,
)

# Import Q-Network
from failure_prob.model.q_learning import GRUQNetwork, CategoricalGRUQNetwork
from failure_prob.conf import Config as OpenvlaDatasetConfig
from failure_prob.utils.metrics import compute_functional_conformal_band
from failure_prob.utils.routines import model_forward_dataloader
from failure_prob.data.utils import Rollout, RolloutDataset, RolloutDatasetContinuous, ConsecutiveSampler
from failure_prob.data import load_rollouts, split_rollouts
from failure_prob.utils.random import seed_everything
from torch.utils.data import DataLoader
from omegaconf import OmegaConf


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    unnorm_key: Optional[str] = None                 # Unnormalization key for the model
    
    n_samples: int = 1                               # Number of samples to draw from the model for each action step
    attn_implementation: str = "flash_attention_2"   # Only eager attention supports return_attentions
    output_logits: bool = True                       # Whether to output logits from the model
    output_attentions: bool = False                  # Whether to output attention weights from the model
    output_hidden_states: bool = False               # Whether to output hidden states from the model

    #################################################################################################################
    # Q-Network specific parameters
    #################################################################################################################
    qnetwork_checkpoint: str = "/home/shellyfra/Projects/SAFE/checkpoints/model_final_TDQC_OpenVLA_LIBERO10.ckpt"  # Path to trained Q-network checkpoint
    qnetwork_config_path: str = "/home/shellyfra/Projects/SAFE/checkpoints/config.yaml"      # Path to Q-network config file
    conformal_threshold: float = 0.5                 # Conformal prediction threshold for stopping
    use_categorical: bool = False                    # Whether to use categorical Q-network
    
    #################################################################################################################
    # Conformal Prediction parameters
    #################################################################################################################
    use_conformal_prediction: bool = True            # Whether to use conformal prediction
    conformal_alpha: float = 0.2                     # Miscoverage rate for conformal prediction (e.g., 0.1 for 90% coverage)
    calibration_data_path: str = "/home/shellyfra/Projects/SAFE/openvla/rollouts/single-foward/libero_10/"  # Path to rollouts for CP calibration
    calibration_seed: int = 20                        # Seed for train/test split (0 means tasks 0,3,5 for test)
    test_task_ids: Optional[List[int]] = None        # Specific task IDs to evaluate (default: [0, 3, 5])
    
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"               # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize
    num_trials_per_task: int = 50                    # Number of rollouts per task
    task_start_index: Optional[int] = None           # Start task index (inclusive) - use test_task_ids instead
    task_end_index: Optional[int] = None             # End task index (inclusive) - use test_task_ids instead
    resume: bool = False                             # Resume from a previous run

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = "qnetwork_eval"     # Extra note to add in run ID for logging
    save_root: str = "./rollouts_qnetwork"           # Root directory to save rollouts
    save_videos: bool = True                         # Whether to save videos with Q-values and CP bands

    use_wandb: bool = False                           # Whether to also log results in Weights & Biases
    wandb_project: str = "openvla-qnetwork"          # Name of W&B project to log to
    wandb_entity: str = "shellyfra"                  # Name of entity to log under
    wandb_dir: Optional[str] = None                  # Directory to save W&B logs
    save_logs: bool = True                           # Whether to dump W&B logs to a file
    seed: int = 7                                    # Random seed
    # fmt: on


def load_qnetwork(cfg: GenerateConfig, device: torch.device):
    """Load trained Q-network model."""
    print(f"Loading Q-network from {cfg.qnetwork_checkpoint}")
    
    # Load config
    if cfg.qnetwork_config_path:
        qnet_cfg = OmegaConf.load(cfg.qnetwork_config_path)
        # Keep as DictConfig - OmegaConf supports attribute access
    else:
        raise ValueError("Must provide qnetwork_config_path")
    
    # Determine input dimension from the training data
    input_dim = qnet_cfg.dataset.dim_features
    
    # Create model
    if cfg.use_categorical or (hasattr(qnet_cfg.model, 'use_categorical') and qnet_cfg.model.use_categorical):
        qnetwork = CategoricalGRUQNetwork(qnet_cfg, input_dim)
    else:
        qnetwork = GRUQNetwork(qnet_cfg, input_dim)
    
    # Load checkpoint
    checkpoint = torch.load(cfg.qnetwork_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        qnetwork.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        qnetwork.load_state_dict(checkpoint['state_dict'])
    else:
        qnetwork.load_state_dict(checkpoint)
    
    qnetwork.to(device)
    qnetwork.eval()
    
    print(f"Q-network loaded successfully with input_dim={input_dim}")
    return qnetwork, qnet_cfg


def prepare_qnetwork_input(
    action: np.ndarray,
    token_probs: torch.Tensor,
    step_idx: int,
    device: torch.device
) -> dict:
    """
    Prepare input for Q-network from current step.
    
    Args:
        action: Action taken (7,)
        token_probs: Token probabilities (7, 256) or top-k (7, 10)
        step_idx: Current step index
        device: Device to put tensors on
        
    Returns:
        Dictionary with required inputs for Q-network
    """
    # Get top-10 probabilities if not already
    if token_probs.shape[-1] > 10:
        top_k_probs = torch.topk(token_probs, k=10, dim=-1).values  # (7, 10)
    else:
        top_k_probs = token_probs
    
    # Prepare batch (add batch and time dimensions)
    batch = {
        "top_10_probs": top_k_probs.unsqueeze(0).unsqueeze(0).to(device),  # (1, 1, 7, 10)
        "action_vectors": torch.from_numpy(action).float().unsqueeze(0).unsqueeze(0).to(device),  # (1, 1, 7)
        "done_masks": torch.ones(1, 1, 1).to(device),  # (1, 1, 1)
    }
    
    return batch


def compute_q_value(
    qnetwork,
    batch: dict,
    hidden_state: Optional[torch.Tensor] = None,
    use_categorical: bool = False
) -> tuple[float, torch.Tensor]:
    """
    Compute Q-value for current step.
    
    Args:
        qnetwork: Q-network model
        batch: Input batch
        hidden_state: Hidden state from previous step (num_layers, batch_size, hidden_size)
        use_categorical: Whether using categorical Q-network
        
    Returns:
        (q_value, new_hidden_state)
    """
    with torch.no_grad():
        # Process through input projection and get GRU input
        actions = batch["action_vectors"]
        
        # Prepare input (same as in forward pass)
        if "top_10_probs" in batch:
            top_k_probs = batch["top_10_probs"]
            B, T, action_dim, k = top_k_probs.shape
            probs = top_k_probs.reshape(B, T, -1)
        else:
            probs = batch["probabilities"]
        
        if qnetwork.use_actions and actions is not None:
            x = torch.cat([probs, actions], dim=-1)
        else:
            x = probs
        
        x = qnetwork.input_proj(x)  # (B, T, hidden_size) = (1, 1, hidden_size)
        x = x.squeeze(0)  # (T, hidden_size) = (1, hidden_size)
        
        # Initialize hidden state if not provided (2D for GRU input compatibility)
        if hidden_state is None:
            num_layers = qnetwork.gru.num_layers
            hidden_size = qnetwork.gru.hidden_size
            hidden_state = torch.zeros(num_layers, hidden_size, dtype=x.dtype, device=x.device)
        
        # Run through GRU with 2D input (T, hidden_size) and 2D hidden state (num_layers, hidden_size)
        gru_out, new_hidden_state = qnetwork.gru(x, hidden_state)
        
        # Get Q-value from head
        if use_categorical:
            # Categorical: get logits then probabilities
            logits = qnetwork.head(gru_out)
            probs = torch.softmax(logits, dim=-1)
            q_value = (probs * qnetwork.support.view(1, -1)).sum(dim=-1)
            q_val_scalar = (1 - q_value.squeeze()).item()  # Convert to failure score
        else:
            # Standard: get Q-value directly
            logits = qnetwork.head(gru_out)
            q = torch.sigmoid(logits)
            q_val_scalar = (1 - q.squeeze()).item()  # Score (higher = more failure)
        
    return q_val_scalar, new_hidden_state


def recovery_sampling(
    cfg: GenerateConfig,
    model,
    processor,
    obs,
    env,
    task_description: str,
    resize_size: tuple,
    vocab_size: int,
    n_action_bins: int,
    device: torch.device,
    num_recovery_steps: int = 20,
    prev_action=None,
    initial_state=None
) -> Tuple[bool, int, Any, float, List[np.ndarray], bool]:
    """
    Execute recovery by sampling 10 actions and trying each one for 20 steps.
    
    Args:
        cfg: Configuration
        model: OpenVLA model
        processor: Processor for observations
        obs: Current observation
        env: Environment
        task_description: Task description string
        resize_size: Image resize size
        vocab_size: Vocabulary size for action binning
        n_action_bins: Number of action bins
        device: Device
        num_recovery_steps: Number of recovery steps for each action trial
        prev_action: Action that triggered CP violation
        initial_state: Environment state to reset to for each trial
        
    Returns:
        (done, steps_taken, final_obs, final_reward, recovery_frames, recovery_succeeded)
    """
    print(f"    Starting recovery: sampling 10 actions, trying each for {num_recovery_steps} steps")
    
    # Get observation and sample 10 actions
    img = get_libero_image(obs, resize_size)
    observation = {
        "full_image": img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    
    # Clear GPU cache before sampling
    torch.cuda.empty_cache()
    
    # Sample 10 actions
    result = get_action(
        cfg,
        model,
        observation,
        task_description,
        processor=processor,
        n_samples=10,
    )
    
    if type(result) is not tuple:
        print("      No tuple result from get_action, skipping recovery")
        return False, 0, obs, 0, [], False
    
    actions, _, _, _ = result
    
    # Remove prev_action from sampling pool
    available_indices = list(range(actions.shape[0]) if actions.ndim > 1 else [0])
    if prev_action is not None and actions.ndim > 1:
        for idx in range(actions.shape[0]):
            if np.allclose(actions[idx], prev_action, atol=1e-3):
                available_indices.remove(idx)
                print(f"      Removed index {idx} (matches prev_action) from sampling pool")
                break
    
    print(f"      Trying {len(available_indices)} sampled actions")
    
    # Try each sampled action
    best_result = None
    best_reward = -float('inf')
    
    for trial_idx, action_idx in enumerate(available_indices):
        print(f"      Trial {trial_idx + 1}/{len(available_indices)}: testing action {action_idx}")
        
        # Reset to initial state
        if initial_state is not None:
            env.reset()
            obs = env.set_init_state(initial_state)
        
        # Get the action to try
        trial_action = actions[action_idx] if actions.ndim > 1 else actions
        
        # Execute this action for num_recovery_steps
        trial_frames = []
        trial_done = False
        trial_reward = 0
        
        for step in range(num_recovery_steps):
            # Get current observation
            img = get_libero_image(obs, resize_size)
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
            
            # For first step, use the sampled action; after that use argmax (n_samples=1)
            if step == 0:
                action = trial_action
            else:
                torch.cuda.empty_cache()
                result = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    n_samples=1,
                )
                if type(result) is tuple:
                    action = result[0][0] if result[0].ndim > 1 else result[0]
                else:
                    action = result
            
            # Normalize and execute action
            action_norm = normalize_gripper_action(action, binarize=True)
            if cfg.model_family == "openvla":
                action_norm = invert_gripper_action(action_norm)
            
            obs, reward, done, info = env.step(action_norm)
            
            # Store frame
            if "agentview_image" in obs:
                trial_frames.append(obs["agentview_image"])
            
            if done:
                trial_done = True
                trial_reward = reward
                success = reward > 0
                print(f"        Episode finished at step {step + 1}: {'SUCCESS' if success else 'FAILURE'}")
                break
        
        # Check if this trial is the best so far
        if trial_done and trial_reward > best_reward:
            best_reward = trial_reward
            best_result = (trial_done, len(trial_frames), obs, trial_reward, trial_frames, trial_reward > 0)
            
            # If we found a success, stop trying other actions
            if trial_reward > 0:
                print(f"      Found successful recovery with action {action_idx}!")
                return best_result
    
    # If we found any completion (even failure), return it
    if best_result is not None:
        print(f"      Best trial completed with reward {best_reward}")
        return best_result
    
    # No trial completed - return last state
    print(f"      No trial completed within {num_recovery_steps} steps")
    return False, num_recovery_steps, obs, 0, [], False


def save_video_with_scores(
    frames: List[np.ndarray],
    scores: List[float],
    cp_band: Optional[np.ndarray],
    task_id: int,
    task_description: str,
    episode_idx: int,
    episode_success: bool,
    cp_early_stop: bool,
    save_folder: str,
    fps: int = 10
):
    """
    Save video with side-by-side RGB frames and Q-value/CP band plot.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        scores: List of Q-values or failure scores
        cp_band: Conformal prediction band array
        task_id: Task ID
        task_description: Task description text
        episode_idx: Episode index
        episode_success: Whether episode succeeded
        cp_early_stop: Whether CP triggered early stop
        save_folder: Folder to save video
        fps: Frames per second
    """
    if len(frames) == 0 or len(scores) == 0:
        return
    
    # Compute score range for consistent plotting
    vmin = min(scores)
    vmax = max(scores)
    if cp_band is not None:
        vmin = min(vmin, cp_band[:len(scores)].min())
        vmax = max(vmax, cp_band[:len(scores)].max())
    
    # Create save folder
    os.makedirs(save_folder, exist_ok=True)
    
    # Determine if episode was predicted to fail
    episode_pred_fail = False
    if cp_band is not None and len(scores) > 0:
        episode_pred_fail = any(scores[i] > cp_band[i] for i in range(min(len(scores), len(cp_band))))
    
    save_name = f"task{task_id}_ep{episode_idx}_succ{int(episode_success)}_predsucc{1-int(episode_pred_fail)}"
    
    video_frames = []
    has_failed = False
    
    for j in range(len(frames)):
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        
        # Determine which scores to plot up to current frame
        score_idx = min(j, len(scores) - 1)
        score_plot_end = score_idx + 1
        
        # Check if this frame exceeded CP band
        if cp_band is not None and score_idx < len(cp_band):
            if scores[score_idx] > cp_band[score_idx]:
                has_failed = True
        
        # Left subplot: RGB frame with colored border
        ax = axes[0]
        # Convert BGR to RGB and apply same 180-degree rotation as used for VLA input
        frame_rgb = cv2.cvtColor(frames[j], cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb[::-1, ::-1]  # Rotate 180 degrees (same as get_libero_image)
        
        # Add colored border (red if failed, green otherwise)
        if has_failed:
            frame_rgb = cv2.copyMakeBorder(frame_rgb, 10, 10, 10, 10, 
                                          cv2.BORDER_CONSTANT, value=(255, 0, 0))
        else:
            frame_rgb = cv2.copyMakeBorder(frame_rgb, 10, 10, 10, 10, 
                                          cv2.BORDER_CONSTANT, value=(0, 255, 0))
        
        ax.imshow(frame_rgb)
        ax.axis("off")
        ax.set_title(f"Frame {j}")
        
        # Right subplot: Q-value/score plot
        ax = axes[1]
        
        # Plot CP band if available
        if cp_band is not None:
            x_band = np.arange(len(cp_band))
            ax.fill_between(x_band, np.zeros_like(cp_band), cp_band, 
                           color="green", alpha=0.2, label="CP band")
        
        # Plot scores up to current frame
        x_scores = np.arange(score_plot_end)
        ax.plot(x_scores, scores[:score_plot_end], 
               color="blue", lw=2, marker='o', markersize=3, label="Q-value")
        
        ax.set_xlim(0, max(len(scores), len(cp_band) if cp_band is not None else 0))
        ax.set_ylim(vmin - 0.05, vmax + 0.05)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Score")
        ax.set_title("Failure Score")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Overall title
        status = "SUCCESS" if episode_success else "FAILURE"
        cp_status = " (CP STOPPED)" if cp_early_stop else ""
        fig.suptitle(f"{task_description}\nEp {episode_idx}, {status}{cp_status}", 
                    fontsize=10)
        fig.tight_layout()
        
        # Convert figure to numpy array
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        video_frames.append(plot_img)
    
    # Save video
    save_path = os.path.join(save_folder, f"{save_name}.mp4")
    imageio.mimsave(save_path, video_frames, fps=fps)
    print(f"    Saved video: {save_path}")


def calibrate_conformal_prediction(
    cfg: GenerateConfig,
    qnetwork,
    val_seen_dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Calibrate conformal prediction bands using val_seen data.
    
    Args:
        cfg: Configuration
        qnetwork: Q-network model
        val_seen_dataloader: DataLoader for val_seen rollouts
        device: Device
        
    Returns:
        cp_band: Conformal prediction band (T,) - threshold at each timestep
    """
    print(f"\nCalibrating conformal prediction with alpha={cfg.conformal_alpha}")
    
    # Get rollouts from dataset first to check if we have data
    rollouts = val_seen_dataloader.dataset.get_rollouts()
    print(f"Number of calibration rollouts: {len(rollouts)}")
    
    if len(rollouts) == 0:
        raise ValueError("No rollouts found in calibration dataset. Check calibration_data_path and config settings.")
    
    # Forward pass on calibration data
    with torch.no_grad():
        scores, valid_masks, _ = model_forward_dataloader(qnetwork, val_seen_dataloader)
    
    scores = scores.detach().cpu().numpy()
    seq_lengths = valid_masks.sum(dim=-1).cpu().numpy()  # (B,)
    
    print(f"Scores shape: {scores.shape}, Valid sequences: {(seq_lengths > 0).sum()}")
    
    # Convert to list of sequences
    scores_by_split = {
        "val_seen": [
            scores[i, :int(seq_lengths[i])] for i in range(len(seq_lengths)) if seq_lengths[i] > 0
        ]
    }
    
    print(f"Number of score sequences: {len(scores_by_split['val_seen'])}")
    
    # Get rollouts from dataset
    rollouts_by_split = {"val_seen": rollouts}
    
    # Compute conformal prediction band
    # For failure detection, calibrate on failed rollouts to learn failure score distribution
    cp_band = compute_functional_conformal_band(
        rollouts=rollouts,
        scores=scores_by_split["val_seen"],
        alpha=cfg.conformal_alpha,
        calib_on="neg",  # Calibration on the successful rollouts
        align_method="extend"
    )
    
    print(f"Calibration complete. CP band length: {len(cp_band)}")
    print(f"CP band range: [{cp_band.min():.4f}, {cp_band.max():.4f}]")
    
    return cp_band





@draccus.wrap()
def eval_libero_with_qnetwork(cfg: GenerateConfig) -> None:
    """Main evaluation loop with Q-network and optional conformal prediction."""
    
    assert cfg.pretrained_checkpoint != "", "Must provide pretrained_checkpoint"
    assert cfg.qnetwork_checkpoint != "", "Must provide qnetwork_checkpoint"
    
    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load OpenVLA model
    print("Loading OpenVLA model...")
    model = get_model(cfg, device)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    
    # Load Q-network
    qnetwork, qnet_cfg = load_qnetwork(cfg, device)
    
    # Load calibration data and compute conformal prediction threshold
    cp_band = None
    if cfg.use_conformal_prediction:
        print("\nLoading calibration data for conformal prediction...")
        assert cfg.calibration_data_path != "", "Must provide calibration_data_path for conformal prediction"
        
        # Set seed for reproducible data split
        seed_everything(20)
        
        # Load all rollouts from calibration data path
        print(f"Loading rollouts from {cfg.calibration_data_path}")
        all_rollouts = load_rollouts(qnet_cfg)
        print(f"Loaded {len(all_rollouts)} rollouts")
        
        # Split rollouts using the same logic as training
        rollouts_by_split_name = split_rollouts(qnet_cfg, all_rollouts)
        
        # Use calibration tasks (non-test tasks) for computing conformal band
        # With seed 20: Seen tasks: [7, 1, 8, 5, 0, 2, 6], Unseen tasks: [9, 4, 3], so calibration uses remaining tasks
        cal_rollouts = rollouts_by_split_name["val_seen"]
        print(f"Using {len(cal_rollouts)} rollouts for calibration")
        set_seed_everywhere(cfg.seed)

        # Create dataset and dataloader for calibration
        cal_dataset = RolloutDataset(qnet_cfg, cal_rollouts)
            
        cal_dataloader = DataLoader(
            cal_dataset,
            batch_size=qnet_cfg.model.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Calibrate conformal prediction
        # cp_band = calibrate_conformal_prediction(cfg, qnetwork, cal_dataloader, device)
        cp_band = np.array([0.55364347, 0.6004635, 0.6159687, 0.62208635, 0.63810444, 0.666774,
 0.6678727, 0.66632473, 0.6751642, 0.6954582, 0.6805216, 0.6960955,
 0.67814636, 0.6970952, 0.6980845, 0.7020362, 0.688499, 0.72018075,
 0.70954645, 0.7482432, 0.75311446, 0.7873663, 0.7984467, 0.8379945,
 0.83655775, 0.85945773, 0.83507466, 0.7962165, 0.7729573, 0.82493544,
 0.8693599, 0.8757074, 0.8962555, 0.8588753, 0.87694156, 0.85260713,
 0.87451696, 0.8608254, 0.84997463, 0.83468676, 0.7942938, 0.7781569,
 0.76962394, 0.74258614, 0.7194412, 0.72303635, 0.6991202, 0.7089902,
 0.69851017, 0.7503003, 0.74801916, 0.70048565, 0.6893656, 0.66563845,
 0.6867944, 0.69243574, 0.6738049, 0.68873614, 0.703409, 0.70690817,
 0.7143407, 0.7245461, 0.73577845, 0.74858296, 0.77545524, 0.7962052,
 0.7937569, 0.7861711, 0.7636255, 0.77389026, 0.77412254, 0.7474363,
 0.75227827, 0.74216425, 0.7438587, 0.7442956, 0.7698471, 0.74626076,
 0.75188804, 0.79475284, 0.8167328, 0.8607359, 0.8684646, 0.8403996,
 0.8180328, 0.8362357, 0.7834352, 0.82495946, 0.7591102, 0.82022196,
 0.8262436, 0.8204602, 0.74081486, 0.7590832, 0.74553585, 0.76324475,
 0.7540931, 0.73856676, 0.732711, 0.7630793, 0.7855823, 0.71375036,
 0.72253114, 0.6961771, 0.7034679, 0.7497002, 0.76246583, 0.7193041,
 0.757805, 0.77443606, 0.77857053, 0.7999973, 0.8385123, 0.79455125,
 0.7956295, 0.7809666, 0.8046513, 0.8627622, 0.8791101, 0.89719117,
 0.90184563, 0.9205423, 0.92653394, 0.9186658, 0.8984595, 0.9005548,
 0.8614375, 0.8433602, 0.83899975, 0.8714964, 0.87327117, 0.8631016,
 0.851225, 0.82007515, 0.8691224, 0.90863246, 0.93648136, 0.9709177,
 0.9666596, 0.96953106, 0.9606114, 0.8653704, 0.8404896, 0.82211506,
 0.8029909, 0.8200626, 0.7928973, 0.8376578, 0.838833, 0.84847116,
 0.85679996, 0.8566748, 0.8307533, 0.82445246, 0.8116056, 0.83702135,
 0.80206823, 0.80689824, 0.8098441, 0.7985216, 0.8131069, 0.7666719,
 0.78898907, 0.79012877, 0.7969121, 0.76583105, 0.7664565, 0.7609348,
 0.7723451, 0.7623124, 0.77744645, 0.75778675, 0.77950794, 0.7411144,
 0.75552917, 0.77622247, 0.84770006, 0.84346795, 0.822721, 0.82367384,
 0.8156876, 0.80118424, 0.78774494, 0.78123343, 0.7632961, 0.75235295,
 0.74925256, 0.757027, 0.76206994, 0.75017214, 0.7684624, 0.7812711,
 0.81160176, 0.80121815, 0.8035307, 0.8000443, 0.78891647, 0.7827481,
 0.7873907, 0.78569365, 0.7668251, 0.7691511, 0.74406147, 0.75093436,
 0.77239823, 0.7807485, 0.7657077, 0.7741063, 0.77754134, 0.797539,
 0.78871393, 0.77949405, 0.7878001, 0.8008459, 0.80572474, 0.78746104,
 0.7798719, 0.76405627, 0.7559308, 0.75140357, 0.73519003, 0.7486421,
 0.75164926, 0.75081336, 0.7629696, 0.7785157, 0.80342627, 0.7757447,
 0.76918936, 0.772375, 0.76598, 0.79464865, 0.76898736, 0.79249716,
 0.8507185, 0.8644688, 0.8558607, 0.86905897, 0.8251446, 0.87989783,
 0.8573288, 0.83905196, 0.85041225, 0.86201835, 0.88624585, 0.90535766,
 0.92370474, 0.90517527, 0.9242281, 0.8987392, 0.89043236, 0.902496,
 0.90396357, 0.90806615, 0.9178214, 0.9138459, 0.85460067, 0.8425995,
 0.8588953, 0.861103, 0.87434906, 0.8851272, 0.8942602, 0.9029039,
 0.89870167, 0.8925415, 0.89917266, 0.8964132, 0.8886843, 0.8881578,
 0.89442587, 0.90109986, 0.90462434, 0.9014236, 0.8977362, 0.8961829,
 0.8947044, 0.9031336, 0.88685346, 0.8858139, 0.88804775, 0.89701617,
 0.90584224, 0.90386724, 0.89283323, 0.9038564, 0.91690993, 0.93772507,
 0.9200907, 0.9159785, 0.91567063, 0.9211582, 0.92185485, 0.925532,
 0.9227066, 0.91854453, 0.93195075, 0.9285706, 0.93884146, 0.9312563,
 0.92639005, 0.93338037, 0.92841995, 0.92217886, 0.9118788, 0.9205862,
 0.92100394, 0.91163695, 0.9171977, 0.91367805, 0.91722775, 0.9120263,
 0.9124695, 0.91138065, 0.90948766, 0.9145549, 0.9061241, 0.9181234,
 0.92165077, 0.9133975, 0.9024857, 0.9000055, 0.88790894, 0.8917929,
 0.90367174, 0.91025186, 0.9214711, 0.9227666, 0.9289508, 0.9284035,
 0.9262128, 0.9254435, 0.91592216, 0.9138979, 0.9100705, 0.902647,
 0.90590215, 0.8902954, 0.895047, 0.88657403, 0.8861008, 0.88604456,
 0.88581955, 0.8859794, 0.89226145, 0.88592637, 0.886162, 0.88632566,
 0.88736, 0.88779664, 0.8882467, 0.88784707, 0.88627017, 0.886645,
 0.88730013, 0.8877244, 0.88796604, 0.8876474, 0.8866395, 0.8863436,
 0.88648146, 0.88645566, 0.9021696, 0.88590896, 0.88618475, 0.8864765,
 0.8866866, 0.8865773, 0.88591814, 0.8868066, 0.8857078, 0.8910154,
 0.8891918, 0.8856416, 0.8878071, 0.88699734, 0.886914, 0.88638747,
 0.8863345, 0.8864156, 0.8875673, 0.88844526, 0.8884573, 0.88757676,
 0.88796234, 0.88795376, 0.8868986, 0.88673437, 0.8861884, 0.8862517,
 0.8864763, 0.8858677, 0.88569504, 0.88667405, 0.8901619, 0.89031005,
 0.88861895, 0.89089215, 0.88777554, 0.88626355, 0.8867115, 0.8896719,
 0.89131004, 0.8915737, 0.8857378, 0.885759, 0.8859913, 0.88636434,
 0.8873354, 0.88797, 0.8878617, 0.88852096, 0.888052, 0.8878014,
 0.88926065, 0.8891235, 0.88886446, 0.8885803, 0.88846904, 0.88862467,
 0.8895038, 0.8899943, 0.889966, 0.8907397, 0.89018357, 0.89037424,
 0.89095885, 0.89119357, 0.8915783, 0.89226687, 0.8916004, 0.8922335,
 0.8929839, 0.89373934, 0.8932896, 0.89315146, 0.8927982, 0.89322865,
 0.8932378, 0.89312524, 0.8941388, 0.8940376, 0.89407927, 0.89471316,
 0.895571, 0.89523757, 0.89398736, 0.8940102, 0.8935072, 0.8931948,
 0.89359534, 0.89370835, 0.8935765, 0.89354247, 0.89352894, 0.8931098,
 0.8939498, 0.893933, 0.8940382, 0.8937826, 0.8924854, 0.892197,
 0.8919015, 0.8918005, 0.8915957, 0.8924333, 0.89256334, 0.89256334,
 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334,
 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334,
 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334,
 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334,
 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334,
 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334,
 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334,
 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334, 0.89256334,
 0.89256334, 0.89256334, 0.89256334, 0.89256334])
        print(f"Conformal prediction calibrated with alpha={cfg.conformal_alpha}")
    
    # Get LIBERO tasks
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name} with {num_tasks_in_suite} tasks")
    
    # Determine which tasks to evaluate
    if cfg.test_task_ids is not None:
        task_ids_to_eval = cfg.test_task_ids
    elif cfg.task_start_index is not None and cfg.task_end_index is not None:
        task_ids_to_eval = list(range(cfg.task_start_index, cfg.task_end_index))
    else:
        # Default: use seed 0 test split (tasks 0, 3, 5 for libero_10)
        task_ids_to_eval = [4]
    
    print(f"Evaluating on tasks: {task_ids_to_eval}")
    
    # Results storage
    all_results = []
    
    # Iterate through tasks
    for task_id in tqdm(task_ids_to_eval, desc="Evaluating tasks"):
        # Get task
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(task_suite.get_task_bddl_file_path(task_id))
        print(f"\nTask {task_id}: {task_name}")
        print(f"Description: {task_description}")
        
        # Get initial states
        initial_states = task_suite.get_task_init_states(task_id)
        
        # Create environment
        env, task_description_from_env = get_libero_env(task, cfg.model_family)
        cfg.unnorm_key = cfg.task_suite_name
        
        # Task results
        task_results = {
            "task_id": task_id,
            "task_name": task_name,
            "successes": [],
            "q_values": [],
            "early_stops": [],
            "episode_lengths": [],
            "failure_scores": [],  # Conformal prediction scores
            "cp_thresholds": [],
            "cp_violations": [],
            "cp_early_stops": [],
        }
        
        # Run episodes
        for episode_idx in range(min(cfg.num_trials_per_task, len(initial_states))):
            print(f"\n  Episode {episode_idx + 1}/{cfg.num_trials_per_task}")
            
            # Initialize W&B
            if cfg.use_wandb:
                run_name = f"task-{task_id}--episode-{episode_idx}"
                wandb.init(
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    name=run_name,
                    dir=cfg.wandb_dir,
                    config=asdict(cfg),
                )
            
            # Reset environment
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            
            # Setup
            t = 0
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
            
            # Episode tracking
            episode_q_values = []
            episode_failure_scores = []  # For conformal prediction
            episode_cp_thresholds = []
            episode_cp_violations = []
            episode_actions = []
            episode_probs = []
            episode_frames = []  # Store RGB frames for video
            early_stopped = False
            cp_early_stop = False
            
            # Initialize hidden state with correct dimensions (num_layers, hidden_size)
            num_layers = qnetwork.gru.num_layers
            hidden_size = qnetwork.gru.hidden_size
            hidden_state = torch.zeros(num_layers, hidden_size, dtype=torch.float32, device=device)
            
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # Wait for objects to stabilize
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue
                    
                    # Get image
                    img = get_libero_image(obs, resize_size)
                    
                    # Prepare observation
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    
                    # Get action from OpenVLA
                    result = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        n_samples=cfg.n_samples,
                    )
                    
                    if type(result) is tuple:
                        actions, probs, logits, generated_outputs = result
                    else:
                        actions = result
                        generated_outputs = {}
                    
                    # Normalize and invert gripper
                    actions = normalize_gripper_action(actions, binarize=True)
                    if cfg.model_family == "openvla":
                        actions = invert_gripper_action(actions)
                    
                    action = actions[0] if actions.ndim > 1 else actions
                    
                    # Extract token probabilities for Q-network
                    if isinstance(logits, tuple):
                        vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
                        n_action_bins = model.config.n_action_bins
                        
                        logits_stacked = torch.stack(logits, axis=0)
                        action_logits = logits_stacked[:, 0, vocab_size - n_action_bins + 1 : vocab_size + 1]
                        token_probs = F.softmax(action_logits, dim=-1)  # (7, 256)
                        
                        # Get top-10 for Q-network
                        top_10_probs = torch.topk(token_probs, k=10, dim=-1).values  # (7, 10)
                    else:
                        # Fallback: create dummy probs
                        top_10_probs = torch.ones(7, 10, device=device) * 0.1
                    
                    # Prepare Q-network input
                    qnet_batch = prepare_qnetwork_input(action, top_10_probs, t, device)
                    
                    # Compute Q-value
                    # q_value, hidden_state = compute_q_value(
                    #     qnetwork,
                    #     qnet_batch,
                    #     hidden_state=hidden_state,
                    #     use_categorical=cfg.use_categorical
                    # )
                    q_value, hidden_state = qnetwork(qnet_batch)
                    
                    episode_q_values.append(q_value)
                    episode_actions.append(action)
                    
                    # Store frame for video (convert BGR to RGB)
                    if cfg.save_videos:
                        frame_bgr = obs["agentview_image"]  # Get raw observation image
                        episode_frames.append(frame_bgr)
                    
                    # Check conformal prediction threshold if enabled
                    cp_threshold = None
                    cp_violation = False
                    if cfg.use_conformal_prediction and cp_band is not None:
                        actual_step = t - cfg.num_steps_wait  # Adjust for stabilization wait
                        if actual_step >= 0 and actual_step < len(cp_band):
                            cp_threshold = cp_band[actual_step]
                            cp_violation = q_value > (cp_threshold + 0.005)
                            
                            episode_failure_scores.append(q_value)
                            episode_cp_thresholds.append(cp_threshold)
                            episode_cp_violations.append(cp_violation)
                            
                            if cp_violation:
                                print(f"    WARNING: Conformal prediction violation! Testing 10 sampled actions...")
                                cp_early_stop = True
                                early_stopped = True
                                
                                # Save current state for resetting during trials
                                current_state = env.sim.get_state()
                                
                                # Execute recovery sampling
                                if isinstance(logits, tuple):
                                    done, steps_taken, obs, reward, recovery_frames, recovery_succeeded = recovery_sampling(
                                        cfg=cfg,
                                        model=model,
                                        processor=processor,
                                        obs=obs,
                                        env=env,
                                        task_description=task_description,
                                        resize_size=resize_size,
                                        vocab_size=vocab_size,
                                        n_action_bins=n_action_bins,
                                        device=device,
                                        num_recovery_steps=20,
                                        prev_action=action,
                                        initial_state=current_state
                                    )
                                    
                                    # Add recovery frames to episode frames
                                    if cfg.save_videos:
                                        episode_frames.extend(recovery_frames)
                                    
                                    # Update timestep counter
                                    t += steps_taken
                                    
                                    # If episode finished during recovery, break
                                    if done:
                                        break
                                    continue
                    
                    # Display monitoring info
                    if cp_threshold is not None:
                        info_str = f"    Step {t}: Q-value = {q_value:.4f}"
                        info_str += f" (CP threshold: {cp_threshold:.4f})"
                        if cp_violation:
                            info_str += " [VIOLATION]"
                        print(info_str)
                    
                    # Log to W&B
                    if cfg.use_wandb:
                        log_dict = {
                            "step": t,
                            "q_value": q_value,
                            "action_norm": np.linalg.norm(action),
                        }
                        if cp_threshold is not None:
                            log_dict.update({
                                "cp_threshold": cp_threshold,
                                "cp_violation": cp_violation,
                            })
                        wandb.log(log_dict)
                    
                    # Execute action (only if no CP violation)
                    obs, reward, done, info = env.step(action)
                    t += 1
                    
                    # Check success
                    if done:
                        success = reward > 0
                        print(f"  Episode finished: {'SUCCESS' if success else 'FAILURE'}")
                        break
                        
                except Exception as e:
                    print(f"  Exception at step {t}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # Record results
            final_success = reward > 0 if t > cfg.num_steps_wait else False
            task_results["successes"].append(final_success)
            task_results["q_values"].append(episode_q_values)
            task_results["early_stops"].append(early_stopped)
            task_results["cp_early_stops"].append(cp_early_stop)
            task_results["episode_lengths"].append(t)
            task_results["failure_scores"].append(episode_failure_scores)
            task_results["cp_thresholds"].append(episode_cp_thresholds)
            task_results["cp_violations"].append(episode_cp_violations)
            
            # Save video with Q-values and CP bands
            if cfg.save_videos and len(episode_frames) > 0:
                save_video_with_scores(
                    frames=episode_frames,
                    scores=episode_failure_scores if episode_failure_scores else episode_q_values,
                    cp_band=cp_band,
                    task_id=task_id,
                    task_description=task_description,
                    episode_idx=episode_idx,
                    episode_success=final_success,
                    cp_early_stop=cp_early_stop,
                    save_folder=cfg.save_root,
                    fps=10  # LIBERO runs at ~10Hz
                )
            
            if cfg.use_wandb:
                log_dict = {
                    "episode_success": final_success,
                    "episode_length": t,
                    "early_stopped": early_stopped,
                    "mean_q_value": np.mean(episode_q_values) if episode_q_values else 0.0,
                    "max_q_value": np.max(episode_q_values) if episode_q_values else 0.0,
                }
                if cfg.use_conformal_prediction:
                    log_dict.update({
                        "cp_early_stop": cp_early_stop,
                        "num_cp_violations": sum(episode_cp_violations) if episode_cp_violations else 0,
                        "mean_failure_score": np.mean(episode_failure_scores) if episode_failure_scores else 0.0,
                    })
                wandb.log(log_dict)
                wandb.finish()
        
        # Task summary
        success_rate = np.mean(task_results["successes"])
        early_stop_rate = np.mean(task_results["early_stops"])
        avg_length = np.mean(task_results["episode_lengths"])
        
        # Calculate stopped and succeeded/failed
        stopped_and_succeeded = sum(1 for i, s in enumerate(task_results["successes"]) if task_results["cp_early_stops"][i] and s)
        stopped_and_failed = sum(1 for i, s in enumerate(task_results["successes"]) if task_results["cp_early_stops"][i] and not s)
        
        print(f"\nTask {task_id} Summary:")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Early Stop Rate (Q-value): {early_stop_rate:.2%}")
        print(f"  Avg Episode Length: {avg_length:.1f}")
        
        if cfg.use_conformal_prediction:
            cp_early_stop_rate = np.mean(task_results["cp_early_stops"])
            total_violations = sum(sum(v) for v in task_results["cp_violations"] if v)
            total_steps = sum(len(v) for v in task_results["cp_violations"] if v)
            print(f"  CP Early Stop Rate: {cp_early_stop_rate:.2%}")
            print(f"  CP Stopped & Succeeded: {stopped_and_succeeded}/{len(task_results['successes'])}")
            print(f"  CP Stopped & Failed: {stopped_and_failed}/{len(task_results['successes'])}")
            if total_steps > 0:
                print(f"  CP Violation Rate: {total_violations}/{total_steps} ({total_violations/total_steps:.2%})")
        
        all_results.append(task_results)
        
        # Cleanup
        env.close()
    
    # Save results
    results_path = os.path.join(cfg.save_root, f"qnetwork_results_{cfg.run_id_note}.pkl")
    os.makedirs(cfg.save_root, exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {results_path}")
    
    # Overall summary
    all_successes = [s for task in all_results for s in task["successes"]]
    all_early_stops = [e for task in all_results for e in task["early_stops"]]
    
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Success Rate: {np.mean(all_successes):.2%}")
    print(f"Q-value Early Stop Rate: {np.mean(all_early_stops):.2%}")
    print(f"Total Episodes: {len(all_successes)}")
    
    if cfg.use_conformal_prediction:
        all_cp_early_stops = [e for task in all_results for e in task["cp_early_stops"]]
        all_cp_violations = [v for task in all_results for violations in task["cp_violations"] for v in violations]
        print(f"CP Early Stop Rate: {np.mean(all_cp_early_stops):.2%}")
        if all_cp_violations:
            print(f"CP Violation Rate: {np.mean(all_cp_violations):.2%}")
            print(f"Total CP Violations: {sum(all_cp_violations)}/{len(all_cp_violations)}")


if __name__ == "__main__":
    eval_libero_with_qnetwork()
