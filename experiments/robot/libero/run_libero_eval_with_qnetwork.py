"""
run_libero_eval_with_qnetwork.py

Runs OpenVLA in LIBERO with live GRUQNetwork evaluation at each step.
Computes Q-values and checks conformal prediction thresholds in real-time.
"""

# CRITICAL: Set rendering backend BEFORE any other imports
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# Disable display requirement for headless rendering
os.environ['EGL_DEVICE_ID'] = '0'

from collections import defaultdict
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal, Optional, Union, Tuple, List
import torch
import torch.nn.functional as F
import wandb
import json

import draccus
import numpy as np
from tqdm import tqdm, trange
from scipy.special import softmax
import matplotlib.pyplot as plt
import imageio
import cv2

import pandas as pd

script_dir = Path(__file__).parent
openvla_root = script_dir.parent.parent.parent
repo_root = openvla_root.parent
sys.path.append(str(openvla_root))
sys.path.append(str(openvla_root / "LIBERO"))
sys.path.append(str(repo_root))

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
)
from experiments.robot.viz_utils import (
    PoseCumulator
)
from experiments.robot.unc_utils import (
    compute_token_uncertainty_metrics,
    compute_samples_uncertainty_metrics,
)

# Import Q-Network
from failure_prob.model.q_learning import GRUQNetwork
from failure_prob.conf import Config as OpenvlaDatasetConfig
from failure_prob.utils.metrics import compute_functional_conformal_band, eval_functional_conformal
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
    
    temperature: float = 1.5                         # Temperature for action sampling in CP-guided selection
    n_action_candidates: int = 10                    # Number of action candidates to sample for CP-guided selection
    use_cp_guided_selection: bool = True             # Whether to use CP-guided action selection

    #################################################################################################################
    # Q-Network specific parameters
    #################################################################################################################
    qnetwork_checkpoint: str = "./checkpoints/model_final_TDQC_OpenVLA_LIBERO10.ckpt"  # Path to trained Q-network checkpoint
    qnetwork_config_path: str = "./checkpoints/config_TDQC_OpenVLA_LIBERO10.yaml"      # Path to Q-network config file
    conformal_threshold: float = 0.5                 # Conformal prediction threshold for stopping
    
    #################################################################################################################
    # Conformal Prediction parameters
    #################################################################################################################
    use_conformal_prediction: bool = False            # Whether to use conformal prediction
    conformal_alpha: float = 0.2                     # Miscoverage rate for conformal prediction (e.g., 0.1 for 90% coverage)
    calibration_data_path: str = "./openvla/rollouts/single-foward/libero_10/"  # Path to rollouts for CP calibration
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
    save_videos: bool = False                         # Whether to save videos with Q-values and CP bands

    use_wandb: bool = False                           # Whether to also log results in Weights & Biases
    wandb_project: str = "openvla-qnetwork"          # Name of W&B project to log to
    wandb_entity: str = "anonymous"                  # Name of entity to log under
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
    # IMPORTANT: Cast to float32 to match training data dtype
    batch = {
        "top_10_probs": top_k_probs.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32),  # (1, 1, 7, 10)
        "action_vectors": torch.from_numpy(action).float().unsqueeze(0).unsqueeze(0).to(device),  # (1, 1, 7)
        "done_masks": torch.ones(1, 1, 1, dtype=torch.float32, device=device),  # (1, 1, 1)
    }
    
    return batch


def compute_q_value(
    qnetwork,
    batch: dict,
    hidden_state: Optional[torch.Tensor] = None,
) -> tuple[float, torch.Tensor]:
    """
    Compute Q-value for current step using manual GRU processing.
    Bypasses the forward() method to avoid train/test mode complications.
    
    Args:
        qnetwork: Q-network model
        batch: Input batch with single timestep (B=1, T=1)
        hidden_state: Hidden state from previous step (num_layers, hidden_size)
        
    Returns:
        (q_value, new_hidden_state)
    """

    with torch.inference_mode():
        # Ensure model is in eval mode for consistent LayerNorm and other layers
        qnetwork.eval()
        
        # Extract and process input
        actions = batch["action_vectors"]
        
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
        
        # Project input: (1, 1, input_dim) -> (1, 1, hidden_size)
        x = qnetwork.input_proj(x)
        
        # Reshape for GRU in unbatched mode: (1, 1, hidden_size) -> (1, hidden_size)
        # This matches the test mode: x[i:i+1, j:j+1, :].squeeze(0)
        x = x.squeeze(0).squeeze(0)  # Remove batch and time dims -> (hidden_size,)
        x = x.unsqueeze(0)  # Add seq_len dim -> (1, hidden_size)
        
        # Initialize hidden state if needed (num_layers, hidden_size) for unbatched mode
        if hidden_state is None:
            num_layers = qnetwork.gru.num_layers
            hidden_size = qnetwork.gru.hidden_size
            hidden_state = torch.zeros(num_layers, hidden_size, dtype=x.dtype, device=x.device)
        
        # Pass through GRU in unbatched mode: 
        # Input: (seq_len=1, hidden_size)
        # Hidden: (num_layers, hidden_size)
        # Output: (seq_len=1, hidden_size), (num_layers, hidden_size)
        gru_out, new_hidden_state = qnetwork.gru(x, hidden_state)
        
        # Get Q-value from head
        # gru_out shape: (1, hidden_size)
        logits = qnetwork.head(gru_out)  # (1, 1)
        q_value = torch.sigmoid(logits)
        q_val_scalar = (1 - q_value.squeeze()).item()  # Convert to failure score
        
    return q_val_scalar, new_hidden_state

def _get_all_object_positions(obs: dict) -> dict:
    """Extract all non-robot object positions from observation."""
    exclude_keys = {'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 
                    'robot0_joint_pos', 'agentview_image', 'robot0_eye_in_hand_image'}
    
    object_positions = {}
    for key, value in obs.items():
        if key.endswith('_pos') and key not in exclude_keys:
            if isinstance(value, np.ndarray) and value.shape == (3,):
                object_positions[key.replace('_pos', '')] = value
    
    return object_positions


def _compute_task_4_metrics(obs: dict) -> dict:
    """Task 4 (libero_10): Put white mug on left plate and yellow-white mug on right plate."""
    # Plates (targets)
    plate_1_pos = obs['plate_1_pos']
    plate_2_pos = obs['plate_2_pos']
    
    # Goal: mugs should be at plate positions (+ small height offset)
    goal_positions = {
        'porcelain_mug_1': plate_1_pos + np.array([0, 0, 0.02]),      # White mug on left plate
        'white_yellow_mug_1': plate_2_pos + np.array([0, 0, 0.02]),   # Yellow-white mug on right plate
    }
    
    # Current mug positions
    current_positions = {
        'porcelain_mug_1': obs['porcelain_mug_1_pos'],
        'white_yellow_mug_1': obs['white_yellow_mug_1_pos'],
    }
    
    return goal_positions, current_positions


def _compute_task_3_metrics(obs: dict) -> dict:
    """Task 3 (libero_10): Put black bowl in bottom drawer and close it."""
    # Get drawer position - the drawer is part of white_cabinet_1
    # When drawer is open, we need to check if bowl is inside
    drawer_pos = obs.get('white_cabinet_1_bottom_region_pos', obs.get('white_cabinet_1_pos', np.array([0, 0, 0])))
    
    # Goal: black bowl should be in the drawer region (with some tolerance for inside)
    goal_positions = {
        'akita_black_bowl_1': drawer_pos + np.array([0, 0, 0.05]),  # Inside drawer, slight height offset
    }
    
    # Current bowl position
    current_positions = {
        'akita_black_bowl_1': obs['akita_black_bowl_1_pos'],
    }
    
    return goal_positions, current_positions


def _compute_task_9_metrics(obs: dict) -> dict:
    """Task 9 (libero_10): Put yellow and white mug in microwave and close it."""
    # Get microwave position
    microwave_pos = obs.get('microwave_1_pos', obs.get('microwave_pos', np.array([0, 0, 0])))
    
    # Goal: both white mug and yellow-white mug should be inside the microwave
    goal_positions = {
        'porcelain_mug_1': microwave_pos + np.array([0, 0, 0.05]),      # White mug in microwave
        'white_yellow_mug_1': microwave_pos + np.array([0, 0, 0.05]),   # Yellow-white mug in microwave
    }
    
    # Current mug positions
    current_positions = {
        'porcelain_mug_1': obs['porcelain_mug_1_pos'],
        'white_yellow_mug_1': obs['white_yellow_mug_1_pos'],
    }
    
    return goal_positions, current_positions


def _compute_generic_metrics(obs: dict) -> dict:
    """Generic fallback: just record all object positions without goal inference."""
    object_positions = _get_all_object_positions(obs)
    
    # No goal positions - just track current positions
    return {}, object_positions


def compute_and_save_final_distances(
    obs: dict,
    task_id: int,
    episode_idx: int,
    save_root: str,
    task_description: str = ""
) -> dict:
    """
    Compute final object-to-goal distances and save metrics when episode is done.
    Versatile function that handles different task types.
    
    Args:
        obs: Final observation dictionary
        task_id: Task ID
        episode_idx: Episode index
        save_root: Root directory to save results
        task_description: Optional task description for context
        
    Returns:
        Dictionary with distance metrics
    """
    # Task-specific handlers (for libero_10 suite)
    task_handlers = {
        3: _compute_task_3_metrics,  # Black bowl in drawer
        4: _compute_task_4_metrics,  # Mugs on plates
        9: _compute_task_9_metrics,  # Yellow-white mug in microwave
        # Add more tasks as needed
    }
    
    # Get goal and current positions using task-specific handler
    if task_id in task_handlers:
        goal_positions, current_positions = task_handlers[task_id](obs)
        has_goal_positions = len(goal_positions) > 0
    else:
        print(f"  No specific handler for task {task_id}, using generic position tracking")
        goal_positions, current_positions = _compute_generic_metrics(obs)
        has_goal_positions = False
    
    # Compute distances if we have goal positions
    distances = {}
    if has_goal_positions:
        for obj_name in goal_positions:
            if obj_name in current_positions:
                distance = np.linalg.norm(current_positions[obj_name] - goal_positions[obj_name])
                distances[obj_name] = float(distance)
                # print(f"  {obj_name}: {distance:.3f}m from goal")
    
    # Compute summary metrics
    if distances:
        avg_distance = np.mean(list(distances.values()))
        max_distance = np.max(list(distances.values()))
        objects_at_goal = sum(1 for d in distances.values() if d < 0.03)  # Within 3cm
    else:
        avg_distance = None
        max_distance = None
        objects_at_goal = None
    
    # Create metrics dictionary
    metrics = {
        'task_id': task_id,
        'episode_idx': episode_idx,
        'task_description': task_description,
        'distances': distances,
        'avg_distance': float(avg_distance) if avg_distance is not None else None,
        'max_distance': float(max_distance) if max_distance is not None else None,
        'objects_at_goal': objects_at_goal,
        'total_objects': len(goal_positions) if has_goal_positions else len(current_positions),
        'current_positions': {k: v.tolist() for k, v in current_positions.items()},
    }
    
    # Only include goal_positions if we have them
    if has_goal_positions:
        metrics['goal_positions'] = {k: v.tolist() for k, v in goal_positions.items()}
    
    # Save to JSON file
    os.makedirs(save_root, exist_ok=True)
    metrics_file = os.path.join(save_root, f"task{task_id}_ep{episode_idx}_final_distances.json")
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # print(f"  Saved distance metrics to {metrics_file}")
    return metrics

def cp_guided_action_selection(
    cfg: GenerateConfig,
    model,
    processor,
    observation: dict,
    task_description: str,
    qnetwork,
    hidden_state: torch.Tensor,
    device: torch.device,
    step_idx: int,
    cp_threshold: Optional[float] = None,
    env = None,
    resize_size: tuple = None,
) -> Tuple[np.ndarray, float, torch.Tensor, dict]:
    """
    Sample multiple actions and select the safest one according to Q-network.
    
    Args:
        cfg: Configuration
        model: OpenVLA model
        processor: Processor for observations
        observation: Current observation dict
        task_description: Task description string
        qnetwork: Q-network model
        hidden_state: Current hidden state
        device: Device
        step_idx: Current step index
        cp_threshold: Optional CP threshold for filtering
        
    Returns:
        (selected_action, q_value, updated_hidden_state, result_dict)
    """
    n_candidates = cfg.n_action_candidates
    
    # Sample multiple actions with temperature
    result = get_action(
        cfg,
        model,
        observation,
        task_description,
        processor=processor,
        n_samples=n_candidates,
        do_sample=True,
        temperature=cfg.temperature,
    )
    
    if type(result) is tuple:
        actions, probs, logits, generated_outputs = result
    else:
        actions = result
        logits = None
    
    # Normalize actions
    actions = normalize_gripper_action(actions, binarize=True)
    if cfg.model_family == "openvla":
        actions = invert_gripper_action(actions)
    
    # Get unique actions to reduce redundant evaluations
    unique_actions = np.unique(actions, axis=0)
    # print(f"        Evaluating {len(unique_actions)} unique actions out of {len(actions)} samples")
    
    # Save current environment state for rollouts
    if env is not None:
        initial_state = env.sim.get_state()
    
    # Evaluate each unique action with 1-step lookahead
    q_values = []
    hidden_states = []
    lookahead_actions = []
    
    for i in range(len(unique_actions)):
        action = unique_actions[i]
        
        if env is not None and resize_size is not None:
            # Restore environment to initial state
            env.sim.set_state(initial_state)
            env.sim.forward()
            
            # Execute candidate action
            try:
                next_obs, reward, done, info = env.step(action)
            except ValueError as e:
                if "terminated episode" in str(e):
                    print(f"        Warning: Action {i} caused termination, skipping")
                    q_value = 1.0
                    q_values.append(q_value)
                    hidden_states.append(hidden_state.clone() if hidden_state is not None else None)
                    lookahead_actions.append(action)
                    continue
                else:
                    raise
            
            # Get next observation for VLA
            next_img = get_libero_image(next_obs, resize_size)
            next_observation = {
                "full_image": next_img,
                "state": np.concatenate(
                    (next_obs["robot0_eef_pos"], quat2axisangle(next_obs["robot0_eef_quat"]), next_obs["robot0_gripper_qpos"])
                ),
            }
            
            # Query VLA for next action and probabilities
            next_result = get_action(
                cfg,
                model,
                next_observation,
                task_description,
                processor=processor,
                n_samples=1,
                do_sample=False,
            )
            
            if type(next_result) is tuple:
                next_actions, _, next_logits, next_gen_outputs = next_result
            else:
                next_actions = next_result
                next_logits = None
            
            # Process next action
            next_actions = normalize_gripper_action(next_actions, binarize=True)
            if cfg.model_family == "openvla":
                next_actions = invert_gripper_action(next_actions)
            next_action = next_actions[0] if next_actions.ndim > 1 else next_actions
            lookahead_actions.append(next_action)
            
            # Get probabilities for next action
            if isinstance(next_logits, tuple):
                predicted_token_ids = next_gen_outputs['sequences'][0, -model.get_action_dim(cfg.unnorm_key):].cpu().numpy()
                
                all_probs = []
                for dof in range(len(next_logits)):
                    logits_dof = next_logits[dof][0].cpu().numpy()  # (vocab_size,)
                    probs_dof = softmax(logits_dof)  # (vocab_size,)
                    all_probs.append(probs_dof)
                
                all_probs = np.array(all_probs)  # (7, vocab_size)
                sorted_probs = np.sort(all_probs, axis=1)[..., ::-1]  # Sort descending
                top_10_probs = sorted_probs[:, :10]  # (7, 10)
                top_10_probs = torch.from_numpy(top_10_probs.copy()).to(device=device, dtype=torch.float32)
            else:
                top_10_probs = torch.ones(7, 10, device=device) * 0.1
        else:
            # Fallback: use current action's probabilities if no env provided
            # Find corresponding index in original actions
            action_idx = np.where((actions == action).all(axis=1))[0][0]
            
            if isinstance(logits, tuple):
                predicted_token_ids = generated_outputs['sequences'][action_idx, -model.get_action_dim(cfg.unnorm_key):].cpu().numpy()
                
                all_probs = []
                for dof in range(len(logits)):
                    logits_dof = logits[dof][action_idx].cpu().numpy()  # (vocab_size,)
                    probs_dof = softmax(logits_dof)  # (vocab_size,)
                    all_probs.append(probs_dof)
                
                all_probs = np.array(all_probs)  # (7, vocab_size)
                sorted_probs = np.sort(all_probs, axis=1)[..., ::-1]  # Sort descending
                top_10_probs = sorted_probs[:, :10]  # (7, 10)
                top_10_probs = torch.from_numpy(top_10_probs.copy()).to(device=device, dtype=torch.float32)
            else:
                top_10_probs = torch.ones(7, 10, device=device) * 0.1
            lookahead_actions.append(action)
        
        # Evaluate action with Q-network
        qnet_batch = prepare_qnetwork_input(
            lookahead_actions[-1] if env is not None else action,
            top_10_probs,
            step_idx,
            device
        )
        
        # Compute Q-value (clone hidden state to avoid modifying original)
        q_value, new_hidden = compute_q_value(
            qnetwork,
            qnet_batch,
            hidden_state=hidden_state.clone() if hidden_state is not None else None,
        )
        
        q_values.append(q_value)
        hidden_states.append(new_hidden)
    
    # Restore environment to original state
    if env is not None:
        env.sim.set_state(initial_state)
        env.sim.forward()
    
    # Select action with lowest failure probability (from unique actions)
    safest_idx = np.argmin(q_values)
    selected_action = unique_actions[safest_idx]
    selected_q_value = q_values[safest_idx]
    selected_hidden_state = hidden_states[safest_idx]
    
    # Build result dict with selected action's info
    # For probabilities, compute from the selected action
    if isinstance(logits, tuple):
        # Find corresponding index in original sampled actions
        orig_action_idx = np.where((actions == selected_action).all(axis=1))[0][0]
        selected_token_ids = generated_outputs['sequences'][orig_action_idx, -model.get_action_dim(cfg.unnorm_key):].cpu().numpy()
        
        # Get probabilities for the selected tokens (the actual predicted tokens)
        selected_top_10_probs_list = []
        for dof in range(len(logits)):
            logits_dof = logits[dof][orig_action_idx].cpu().numpy()  # (vocab_size,)
            probs_dof = softmax(logits_dof)  # (vocab_size,)
            
            # Get top-10 indices (will include the selected token if it's in top-10)
            top_10_indices = np.argsort(probs_dof)[::-1][:10]
            top_10_probs_dof = probs_dof[top_10_indices]  # (10,)
            
            selected_top_10_probs_list.append(top_10_probs_dof)
        
        # Stack to (7, 10)
        selected_top_10_probs = np.array(selected_top_10_probs_list)
        selected_top_10_probs = torch.from_numpy(selected_top_10_probs.copy()).to(device=device, dtype=torch.float32)
    else:
        selected_top_10_probs = torch.ones(7, 10, device=device) * 0.1
    
    result_dict = {
        'top_10_probs': selected_top_10_probs,
        'all_q_values': q_values,
        'safest_idx': safest_idx,
        'q_value_range': (min(q_values), max(q_values)),
        'n_unique_actions': len(unique_actions),
    }
    
    print(f"step {step_idx}   CP-guided selection with 1-step lookahead: Q-values range [{min(q_values):.4f}, {max(q_values):.4f}], selected #{safest_idx}/{len(unique_actions)} with Q={selected_q_value:.4f}")
    
    return selected_action, selected_q_value, selected_hidden_state, result_dict


def recovery_sampling(
    cfg: GenerateConfig,
    model,
    processor,
    obs,
    env,
    task_description: str,
    resize_size: tuple,
    num_recovery_steps: int = 520,
    initial_state=None,
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
        current_state: Current environment state (flattened) at CP violation
        
    Returns:
        (done, steps_taken, final_obs, final_reward, recovery_frames, recovery_succeeded)
    """
    print(f"    Starting recovery: sampling 10 actions, trying each for {num_recovery_steps} steps")
    
    recovery_prompt = f"ATTENTION: {task_description}. Previous attempt failed. Try a different approach."
    # recovery_prompt = f'ATTENTION: execute {task_description}. Do it slow and safe.'

    best_result = None
    best_reward = -float('inf')
    
    trial_frames = []
    trial_done = False
    trial_reward = 0
    torch.cuda.empty_cache()
    env.reset()
    obs = env.set_init_state(initial_state)
    for step in range(num_recovery_steps): # - len(actions_history)
        # Wait for objects to stabilize
        if step < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            continue
        img = get_libero_image(obs, resize_size)
        observation = {
            "full_image": img,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }
            
        torch.cuda.empty_cache()
        result = get_action(
            cfg,
            model,
            observation,
            recovery_prompt,
            processor=processor,
            n_samples=1,
            do_sample=False,
        )
        if type(result) is tuple:
            action = result[0][0] if result[0].ndim > 1 else result[0]
        else:
            action = result
        
        action_norm = normalize_gripper_action(action, binarize=True)
        if cfg.model_family == "openvla":
            action_norm = invert_gripper_action(action_norm)
        
        obs, reward, done, info = env.step(action_norm)
        if step % 10 == 0 or done:
            print(f"        Step {step}: Done: {done}")
        
        if "agentview_image" in obs:
            trial_frames.append(obs["agentview_image"])
        
        if done:
            trial_done = True
            trial_reward = reward
            success = reward > 0
            print(f"        Episode finished at step {step + 1}: {'SUCCESS' if success else 'FAILURE'}")
            break
        
    if trial_done and trial_reward > best_reward:
        best_reward = trial_reward
        best_result = (trial_done, len(trial_frames), obs, trial_reward, trial_frames, trial_reward > 0)
        
    if trial_reward > 0:
        print(f"      Found successful recovery with action !")
        return best_result
    
    if best_result is not None:
        print(f"      Best trial completed with reward {best_reward}")
        return best_result
    
    print(f"      No trial completed within {num_recovery_steps} steps")
    return False, num_recovery_steps, obs, 0, trial_frames, False


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
    rollouts_by_split_name: dict = None,
    calib_split_names: list = ["val_seen"],
    test_split_names: list = ["val_unseen"],
) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    """
    Calibrate conformal prediction bands using val_seen data.
    
    Args:
        cfg: Configuration
        qnetwork: Q-network model
        val_seen_dataloader: DataLoader for val_seen rollouts
        device: Device
        rollouts_by_split_name: Dictionary mapping split names to rollouts (optional)
        calib_split_names: List of calibration split names
        test_split_names: List of test split names
        
    Returns:
        cp_band: Conformal prediction band (T,) - threshold at each timestep
        df: DataFrame with evaluation metrics
        cp_bands_by_alpha: Dictionary mapping alpha values to CP bands
    """
    print(f"\nCalibrating conformal prediction with alpha={cfg.conformal_alpha}")
    
    # Get rollouts from dataset first to check if we have data
    rollouts = val_seen_dataloader.dataset.get_rollouts()
    print(f"Number of calibration rollouts: {len(rollouts)}")
    
    if len(rollouts) == 0:
        raise ValueError("No rollouts found in calibration dataset. Check calibration_data_path and config settings.")
    
    # Forward pass on calibration data - PROCESS ONE STEP AT A TIME like live inference
    print("Processing calibration data step-by-step to match live inference behavior...")
    scores_list = []
    
    with torch.no_grad():
        for rollout in tqdm(rollouts, desc="Calibrating"):
            episode_scores = []
            
            # Initialize hidden state for this episode
            num_layers = qnetwork.gru.num_layers
            hidden_size = qnetwork.gru.hidden_size
            hidden_state = torch.zeros(num_layers, hidden_size, dtype=torch.float32, device=device)
            
            # Process each timestep sequentially
            T = rollout.top_10_probs.shape[0]  # Sequence length
            for t in range(T):
                # Prepare single-step batch (same as live inference)
                # IMPORTANT: Cast to float32 to match training data dtype
                batch = {




                    "top_10_probs": rollout.top_10_probs[t:t+1].unsqueeze(0).to(device=device, dtype=torch.float32),  # (1, 1, 7, 10)
                    "action_vectors": rollout.action_vectors[t:t+1].unsqueeze(0).to(device=device, dtype=torch.float32),  # (1, 1, 7)
                    "done_masks": torch.ones(1, 1, 1, dtype=torch.float32, device=device),  # (1, 1, 1)
                }
                
                # Compute Q-value using same function as live inference
                q_value, hidden_state = compute_q_value(
                    qnetwork,
                    batch,
                    hidden_state=hidden_state,
                )
                
                episode_scores.append(q_value)
            
            scores_list.append(np.array(episode_scores))
    
    print(f"Calibration scores computed for {len(scores_list)} episodes")
    
    # For compatibility with downstream code, create dummy seq_lengths
    seq_lengths = np.array([len(s) for s in scores_list])
    
    print(f"Calibration scores computed for {len(scores_list)} episodes")
    
    # For compatibility with downstream code, create dummy seq_lengths
    seq_lengths = np.array([len(s) for s in scores_list])
    
    print(f"Scores shape: {len(scores_list)} sequences, Valid sequences: {(seq_lengths > 0).sum()}")
    
    # Filter out invalid sequences
    valid_indices = [i for i in range(len(seq_lengths)) if seq_lengths[i] > 0]
    scores_list = [scores_list[i] for i in valid_indices]
    filtered_rollouts = [rollouts[i] for i in valid_indices]
    
    print(f"Number of score sequences: {len(scores_list)}")
    print(f"Number of filtered rollouts: {len(filtered_rollouts)}")
    
    # Build rollouts_by_split_name and scores_by_split_name
    # Always use filtered_rollouts for val_seen to ensure alignment with scores
    if rollouts_by_split_name is None:
        rollouts_by_split_name = {}
    
    # Override val_seen with filtered rollouts to match scores
    rollouts_by_split_name["val_seen"] = filtered_rollouts
    scores_by_split_name = {"val_seen": scores_list}
    
    # If we have test splits in rollouts_by_split_name, we need to compute their scores too
    # For now, use the same data for both calibration and test if test splits not provided
    for split_name in test_split_names:
        if split_name not in rollouts_by_split_name:
            rollouts_by_split_name[split_name] = filtered_rollouts
        if split_name not in scores_by_split_name:
            scores_by_split_name[split_name] = scores_list
    
    # Use eval_functional_conformal to get consistent output
    df, cp_bands_by_alpha = eval_functional_conformal(
        rollouts_by_split_name, scores_by_split_name, "model",
        calib_split_names=calib_split_names, test_split_names=test_split_names
    )
    
    # Retrieve the CP band for the given alpha
    cp_band = cp_bands_by_alpha[cfg.conformal_alpha][0]  # Shape: (T,)
    
    print(f"Calibration complete. CP band length: {len(cp_band)}")
    print(f"CP band range: [{cp_band.min():.4f}, {cp_band.max():.4f}]")
    
    return cp_band, df, cp_bands_by_alpha


@draccus.wrap()
def eval_libero_with_qnetwork(cfg: GenerateConfig) -> None:
    """Main evaluation loop with Q-network and optional conformal prediction."""
    
    assert cfg.pretrained_checkpoint != "", "Must provide pretrained_checkpoint"
    assert cfg.qnetwork_checkpoint != "", "Must provide qnetwork_checkpoint"
    
    # Set random seed
    seed_everything(cfg.seed)
    
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
    print("\nLoading calibration data for conformal prediction...")
    assert cfg.calibration_data_path != "", "Must provide calibration_data_path for conformal prediction"
    
    # Set seed for reproducible data split
    seed_everything(0)
    # Load or calibrate conformal prediction
    cp_band_path = f"./checkpoints/cp_band_alpha{cfg.conformal_alpha}.npy"
    if cp_band_path is not None and os.path.exists(cp_band_path):
        print(f"Loading CP band from {cp_band_path}")
        cp_band = np.load(cp_band_path)
        cp_bands_by_alpha = {cfg.conformal_alpha: np.expand_dims(cp_band, axis=0)}
        df = None
        print(f"CP band loaded. Length: {len(cp_band)}, Range: [{cp_band.min():.4f}, {cp_band.max():.4f}]")
    else:
        print("CP band file not found. Computing from scratch...")
                # Load all rollouts from calibration data path
        print(f"Loading rollouts from {cfg.calibration_data_path}")
        all_rollouts = load_rollouts(qnet_cfg)
        print(f"Loaded {len(all_rollouts)} rollouts")
        
        # Split rollouts using the same logic as training
        seed_everything(cfg.calibration_seed)
        rollouts_by_split_name = split_rollouts(qnet_cfg, all_rollouts)
        
        # Use calibration tasks (non-test tasks) for computing conformal band
        # With seed 20: Seen tasks: [7, 1, 8, 5, 0, 2, 6], Unseen tasks: [9, 4, 3], so calibration uses remaining tasks
        cal_rollouts = rollouts_by_split_name["val_seen"]
        print(f"Using {len(cal_rollouts)} rollouts for calibration")

        # Create dataset and dataloader for calibration
        cal_dataset = RolloutDataset(qnet_cfg, cal_rollouts)
            
        cal_dataloader = DataLoader(
            cal_dataset,
            batch_size=qnet_cfg.model.batch_size,
            shuffle=False,
            num_workers=0
        )
        cp_band, df, cp_bands_by_alpha = calibrate_conformal_prediction(
            cfg, qnetwork, cal_dataloader, device,
            rollouts_by_split_name=None,  # Let it build from dataloader
            calib_split_names=["val_seen"],
            test_split_names=["val_unseen"]
        )

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
        # Default: use seed 4 test split (Seen tasks: [7, 1, 8, 5, 0, 2, 6], Unseen tasks: [9, 4, 3])
        task_ids_to_eval = [3, 9]
    
    print(f"Evaluating on tasks: {task_ids_to_eval}")
    
    # Results storage
    all_results = []

    
    # Iterate through tasks
    seed_everything(cfg.seed)
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
            
            # Initialize hidden state for GRU - unbatched mode: (num_layers, hidden_size)
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
                    
                    # Use CP-guided action selection if enabled
                    if cfg.use_cp_guided_selection:
                        action, q_value, hidden_state, cp_result = cp_guided_action_selection(
                            cfg=cfg,
                            model=model,
                            processor=processor,
                            observation=observation,
                            task_description=task_description,
                            qnetwork=qnetwork,
                            hidden_state=hidden_state,
                            device=device,
                            step_idx=t,
                            cp_threshold=cp_band[t - cfg.num_steps_wait] if (cfg.use_conformal_prediction and cp_band is not None and (t - cfg.num_steps_wait) >= 0 and (t - cfg.num_steps_wait) < len(cp_band)) else None,
                        )
                        top_10_probs = cp_result['top_10_probs']
                        logits = None  # Not used when CP-guided selection is enabled
                    else:
                        # Standard action selection
                        result = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            n_samples=cfg.n_samples,
                            do_sample=False,
                        )
                        
                        logits = None
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
                        
                        if isinstance(logits, tuple):
                            logits_cpu = tuple(l.cpu().numpy() for l in logits)
                            logits_probs = np.array([softmax(logits_cpu[i]) for i in range(len(logits_cpu))])
                            probs_dof = logits_probs.squeeze(axis=1)
                            sorted_probs = np.sort(probs_dof, axis=1)[..., ::-1]
                            sorted_probs = sorted_probs[:, :10]
                            top_10_probs = torch.from_numpy(sorted_probs.copy()).to(device=device, dtype=torch.float32)
                        else:
                            top_10_probs = torch.ones(7, 10, device=device) * 0.1
                        
                        # Prepare Q-network input
                        qnet_batch = prepare_qnetwork_input(action, top_10_probs, t, device)
                        
                        # Compute Q-value
                        q_value, hidden_state = compute_q_value(
                            qnetwork,
                            qnet_batch,
                            hidden_state=hidden_state,
                        )
                    
                    if t == cfg.num_steps_wait:
                        print(f"[DEBUG] First Q-value at step {t}: {q_value:.7f}")

                    episode_q_values.append(q_value)
                    episode_actions.append(action)
                    
                    if cfg.save_videos:
                        frame_bgr = obs["agentview_image"]
                        episode_frames.append(frame_bgr)
                    
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

                                #  # Save current state for resetting during trials
                                # current_state = env.sim.get_state()
                                
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
                                        initial_state=initial_states[episode_idx],
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
            
            # Compute and save final distance metrics
            distance_metrics = compute_and_save_final_distances(
                obs=obs,
                task_id=task_id,
                episode_idx=episode_idx,
                save_root=cfg.save_root,
                task_description=task_description
            )
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
            if cfg.save_videos and len(episode_frames) > 0: # and not (final_success and not cp_early_stop)
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
