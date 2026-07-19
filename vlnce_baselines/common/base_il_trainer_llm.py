import json
import sys
import jsonlines
import os
import time
import warnings
from pathlib import Path
import imageio

# Resolve project root for shared model paths (cross-platform)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from shared.evaluation_selection import filter_ids_by_cross_floor
from shared.results import aggregate_numeric_metrics
from shared.resume_utils import load_episode_metrics
from collections import defaultdict
from typing import Dict, List
from PIL import Image
import requests
from openai import OpenAI
import cv2
import base64
import io
import numpy as np

# for navigator      
from vlnce_baselines.common.navigator.spatialNavigator import *
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy

import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *

from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

from ..utils import get_camera_orientations
from ..models.utils import (
    length2mask, dir_angle_feature, dir_angle_feature_with_ele,
)
from shared.eval_metrics import format_episode_metric
from shared.ssa import (
    SSAController,
    execute_oracle_expert_replay,
    execute_ssa_takeover,
    expert_actions_for_segment,
    expert_record_for_episode,
)
from shared.ssa.oracle import proposal_oracle_segment
from shared.ssa.trajectory import save_trajectory_debug
from shared.navigation import select_executable_candidate
from shared.trajectory_metrics import metric_positions
from shared.visualization import EpisodeGifRecorder


def _ssa_front_view(images_dict):
    return images_dict.get("0")


def _ssa_view_yaw_deg(angle_value):
    angle_deg = float(np.rad2deg(angle_value))
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _ssa_stair_subtask_active(current_action, current_landmarks):
    stair_keywords = ("stair", "stairs", "step", "steps", "staircase", "upstairs", "downstairs")
    text_parts = [str(current_action or "")]
    if isinstance(current_landmarks, list):
        text_parts.extend(str(item or "") for item in current_landmarks)
    elif current_landmarks:
        text_parts.append(str(current_landmarks))
    text = " ".join(text_parts).lower()
    return any(keyword in text for keyword in stair_keywords)


def _new_ssa_episode_trace():
    return {
        "proposal_seen": False,
        "delegated": False,
        "takeover_success": False,
        "takeover_reason": "",
        "available_steps": [],
        "rejection_reasons": [],
        "proposal_estimates": [],
        "delegate_declined_steps": [],
    }


def _resolve_valid_viewpoint(
    predicted_vp,
    candidate_dict,
    logger,
    *,
    context: str,
):
    predicted_key = str(predicted_vp)
    if predicted_key in candidate_dict:
        return predicted_key

    candidate_keys = list(candidate_dict.keys())
    if not candidate_keys:
        raise KeyError(f"No candidate viewpoints available during {context}.")

    fallback_key = "0" if "0" in candidate_dict else candidate_keys[0]
    logger.warning(
        f"Invalid predicted viewpoint '{predicted_key}' during {context}. "
        f"Falling back to valid candidate '{fallback_key}'. "
        f"Available candidates: {candidate_keys}"
    )
    return fallback_key

def image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    if isinstance(image_array, np.ndarray):
        # Convert to PIL Image
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # RGB image
            pil_image = Image.fromarray(image_array.astype(np.uint8), mode='RGB')
        elif len(image_array.shape) == 2:
            # Grayscale image
            pil_image = Image.fromarray(image_array.astype(np.uint8), mode='L')
        else:
            # Other formats, convert to RGB
            pil_image = Image.fromarray(image_array.astype(np.uint8))
        
        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    return None


def _save_episode_rgb_gif(gif_dir, episode_id, frames, nav_logger, max_width=640, duration=0.4):
    if not frames:
        return
    recorder = EpisodeGifRecorder(gif_dir, enabled=True, max_width=max_width, duration=duration, annotate=False)
    recorder.extend_frames(frames)
    output_path = recorder.save(episode_id)
    nav_logger.info(f"Saved RGB GIF for episode {episode_id} to {output_path}")


def _extract_low_level_rgb(observation):
    rgb = observation.get("rgb")
    if rgb is None:
        return None
    frame = np.asarray(rgb)
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        return frame[..., :3].astype(np.uint8).copy()
    return None

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import tensorflow as tf  # noqa: F401

class BaseVLNCETrainerLLM(BaseILTrainer):
    r"""A base trainer for VLN-CE imitation learning."""
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from waypoint_prediction.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        self.waypoint_predictor.load_state_dict(
            torch.load(
                os.path.join(PROJECT_ROOT, "models", "waypoint_prediction", "checkpoints", "check_val_best_avg_wayscore"),
                map_location=torch.device('cpu'),
                weights_only=False,
            )['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad = False

  
        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        logger.info("Finished setting up waypoint_predictor.")

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, weights_only=False, *args, **kwargs)

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
        )
        
    def generate_input(self, observations):
        instruction = observations['instruction']['text']
        image_dict = {}
        rgb_image_dict = {}
        depth_image_dict = {}
        rgb_index = 0
        depth_index = 0
        for key in observations.keys():
            if 'rgb' in key:
                # Convert numpy array to PIL Image for rgb images
                rgb_image_dict[str(rgb_index)] = Image.fromarray(observations[key], mode="RGB")
                rgb_index += 1
            if 'depth' in key:
                # Process depth images
                if observations[key].ndim == 3 and observations[key].shape[-1] == 1:
                    depth_map = observations[key].squeeze(-1)
                else:
                    depth_map = observations[key]
                depth_img = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))).astype(np.uint8)
                depth_image_dict[str(depth_index)] = Image.fromarray(depth_img)
                depth_index += 1
        for index in rgb_image_dict:
            image_dict[index] = {
                'rgb': rgb_image_dict[index],
                'depth': depth_image_dict[index],
                'base64': image_to_base64(np.array(rgb_image_dict[index]))
            }

        return instruction, image_dict
    
    def construct_image_dicts(self, batch_distance, batch_angles, image_dict):
        waypoint_distances = {}
        waypoint_radius = {}
        waypoint_images = {}
        angles = batch_angles[-1]
        for angle_idx in range(len(angles)):
            angle = angles[angle_idx]
            angle_deg = np.rad2deg(angle)
            if 0 < angle_deg <= 30:
                waypoint_images['1'] = image_dict['1']
                waypoint_distances['1'] = batch_distance[angle_idx]
                waypoint_radius['1'] = angles[angle_idx]
            elif 30 < angle_deg <= 60:
                waypoint_images['2'] = image_dict['2']
                waypoint_distances['2'] = batch_distance[angle_idx]
                waypoint_radius['2'] = angles[angle_idx]
            elif 60 < angle_deg <= 90:
                waypoint_images['3'] = image_dict['3']
                waypoint_distances['3'] = batch_distance[angle_idx]
                waypoint_radius['3'] = angles[angle_idx]
            elif 90 < angle_deg <= 120:
                waypoint_images['4'] = image_dict['4']
                waypoint_distances['4'] = batch_distance[angle_idx]
                waypoint_radius['4'] = angles[angle_idx]
            elif 120 < angle_deg <= 150:
                waypoint_images['5'] = image_dict['5']
                waypoint_distances['5'] = batch_distance[angle_idx]
                waypoint_radius['5'] = angles[angle_idx]
            elif 150 < angle_deg <= 180:
                waypoint_images['6'] = image_dict['6']
                waypoint_distances['6'] = batch_distance[angle_idx]
                waypoint_radius['6'] = angles[angle_idx]
            elif 180 < angle_deg <= 210:
                waypoint_images['7'] = image_dict['7']
                waypoint_distances['7'] = batch_distance[angle_idx]
                waypoint_radius['7'] = angles[angle_idx]
            elif 210 < angle_deg <= 240:
                waypoint_images['8'] = image_dict['8']
                waypoint_distances['8'] = batch_distance[angle_idx]
                waypoint_radius['8'] = angles[angle_idx]
            elif 240 < angle_deg <= 270:
                waypoint_images['9'] = image_dict['9']
                waypoint_distances['9'] = batch_distance[angle_idx]
                waypoint_radius['9'] = angles[angle_idx]
            elif 270 < angle_deg <= 300:
                waypoint_images['10'] = image_dict['10']
                waypoint_distances['10'] = batch_distance[angle_idx]
                waypoint_radius['10'] = angles[angle_idx]
            elif 300 < angle_deg <= 330:
                waypoint_images['11'] = image_dict['11']
                waypoint_distances['11'] = batch_distance[angle_idx]
                waypoint_radius['11'] = angles[angle_idx]
            else:
                waypoint_images['0'] = image_dict['0']  
                waypoint_distances['0'] = batch_distance[angle_idx]
                waypoint_radius['0'] = angles[angle_idx]
                
        return waypoint_images, waypoint_radius, waypoint_distances
    
    def _create_reverse_action(self, action):
        """
        Create a reverse action to undo the given action.
        Args:
            action: The original action to reverse
        Returns:
            The reverse action
        """
        if action['action']['action'] == 4:  # Move action
            # For move actions, reverse the angle by adding 180 degrees (π radians)
            original_angle = action['action']['action_args']['angle']
            original_distance = action['action']['action_args']['distance']
            
            # Reverse the angle (add π radians)
            reverse_angle = original_angle + math.pi
            # Normalize to [-π, π]
            reverse_angle = (reverse_angle + math.pi) % (2 * math.pi) - math.pi
            
            return {
                'action': {
                    'action': 4,
                    'action_args': {
                        'angle': reverse_angle,
                        'distance': original_distance,
                    }
                }
            }
        else:
            # For other actions, just return the same action (no reverse needed)
            return action
    

    def _eval_llm(
        self,
    ) -> None:
        r"""Evaluation.

        Args:
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint

        Returns:
            None
        """
        config = self.config.clone()


        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                if config.EVAL.OVERWRITE_RESULTS:
                    print("Overwriting previous results...")
                else:
                    print(f"skipping -- evaluation exists. File path: {fname}")
                    user_input = input("Do you want to overwrite the results? (yes/no): ").strip().lower()
                    if user_input != "yes":
                        print("Skipping evaluation.")
                        return
                    else:
                        print("Overwriting previous results...")
                

        envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.traj
        ) 

        #envs.number_of_episodes = [1] # set the number of episodes
        dataset_length = sum(envs.number_of_episodes) 
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(config) 
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            config,
            load_from_ckpt=False,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )
        self.policy.eval() 
        self.waypoint_predictor.eval()
        observations = envs.reset()
        
        instruction, images_list = self.generate_input(observations[-1])
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        ) 
        batch = batch_obs(observations, self.device) 
        batch = apply_obs_transforms_batch(batch, obs_transforms) 

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        ) 

        stats_episodes = {}
        save_episode_gif = bool(getattr(config, "SAVE_EPISODE_GIF", True))
        gif_max_width = int(getattr(config, "EPISODE_GIF_MAX_WIDTH", 640))
        gif_duration = float(getattr(config, "EPISODE_GIF_DURATION", 0.4))
        high_rgb_gif_dir = os.path.join(config.RESULTS_DIR, "rgb_gifs_high")
        low_rgb_gif_dir = os.path.join(config.RESULTS_DIR, "episode_gifs")
        if save_episode_gif:
            os.makedirs(high_rgb_gif_dir, exist_ok=True)
            os.makedirs(low_rgb_gif_dir, exist_ok=True)

        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)
            )

        resume_enabled = bool(getattr(config, "RESUME", False))

        # ========== Resume from checkpoint: Load completed episodes ==========
        completed_episode_ids = set()
        episode_results_file = os.path.join(
            config.RESULTS_DIR,
            f"episode_results_{config.TASK_CONFIG.DATASET.SPLIT}_r{self.local_rank}_w{self.world_size}.json"
        )
        if resume_enabled and os.path.exists(episode_results_file):
            try:
                with open(episode_results_file, "r") as f:
                    existing_results = json.load(f)
                completed_episode_ids = set(existing_results.keys())
                # Convert to int if needed (episode_ids might be stored as strings)
                completed_episode_ids = {int(ep_id) if ep_id.isdigit() else ep_id for ep_id in completed_episode_ids}
                print(f"[Resume] Found {len(completed_episode_ids)} completed episodes: {sorted(completed_episode_ids)}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"[Resume] Could not load existing results: {e}")
                completed_episode_ids = set()

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        # Update progress bar if resuming
        if pbar is not None and len(completed_episode_ids) > 0:
            pbar.update(len(completed_episode_ids))
        log_str = (
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        # set up the logger
        log_file = f"./navigator_{config.LOG_FILE}"
        if os.path.exists(log_file): os.remove(log_file)
        import logging
        logging.basicConfig(
            format='%(asctime)s - %(filename)s/%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            level=os.environ.get("LOGLEVEL", "INFO").upper(),
            stream=sys.stdout,
            filemode="a"
        )
        nav_logger = logging.getLogger("vln_logger")
        nav_logger.addHandler(logging.FileHandler(filename=log_file))
        
        dataset_name = "R2R"
        if not os.path.exists(f"cache_files/{dataset_name}"):
            os.makedirs(f"cache_files/{dataset_name}")

        actions_cache_path = f"./cache_files/{dataset_name}/actions_cache_3step-nav.json"
        if os.path.exists(actions_cache_path): 
            with open(actions_cache_path, "r", encoding="utf-8") as file:
                actions_cache = json.load(file)
        else:
            actions_cache = {}
        
        navigator = Open_Nav(self.device,config.LLM, config.API_KEY)
        ssa_controller = SSAController(
            enabled=getattr(config, "SSA_GUIDANCE", False),
            workspace_root=Path(PROJECT_ROOT),
            checkpoint_path=getattr(config, "SSA_CHECKPOINT", ""),
            detect_threshold=float(getattr(config, "SSA_DETECT_THRESHOLD", 0.5)),
            detector_model_source=getattr(config, "SSA_DETECTOR_MODEL_SOURCE", None),
            filter_behind=getattr(config, "SSA_FILTER_BEHIND", False),
            oracle_exit_enabled=getattr(config, "SSA_ORACLE_EXIT_ENABLE", False),
            oracle_entry_gate_enabled=getattr(config, "SSA_ORACLE_ENTRY_GATE_ENABLE", True),
            oracle_entry_radius_m=getattr(config, "SSA_ORACLE_ENTRY_RADIUS", 1.5),
            max_takeovers_per_episode=int(getattr(config, "SSA_MAX_TAKEOVERS_PER_EPISODE", 1)),
            oracle_expert_replay=bool(getattr(config, "SSA_ORACLE_EXPERT_REPLAY", False)),
        )
        ssa_enabled = bool(getattr(ssa_controller, "enabled", False))
        current_step = 0
        current_action_idx = 0
        nav_history = []
        error_number = 0
        chosen_images = []
        low_level_rgb_frames = []
        chosen_images_descriptions = []  # Descriptions for each image
        env_actions_history = []  # Record environment actions for backtracking
        previous_position = None  # Track previous position to detect if stuck
        stuck_directions = set()  # Track directions that caused the agent to get stuck
        last_chosen_vp = None  # Track the last chosen viewpoint

        # Step-level statistics for current episode
        episode_step_latencies = []  # List of latencies for each step in current episode
        episode_step_input_tokens = []  # List of input tokens for each step
        episode_step_output_tokens = []  # List of output tokens for each step
        step_start_time = None  # Track start time of current step

        # Global statistics across all episodes (for computing overall average)
        global_total_latency = 0.0
        global_total_input_tokens = 0
        global_total_output_tokens = 0
        global_total_steps = 0

        # ========== Resume: Load existing global statistics ==========
        if resume_enabled and len(completed_episode_ids) > 0:
            running_stats_file = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{config.TASK_CONFIG.DATASET.SPLIT}_running.json"
            )
            if os.path.exists(running_stats_file):
                try:
                    with open(running_stats_file, "r") as f:
                        running_stats = json.load(f)
                    # Recover global statistics from running stats
                    if 'global_total_steps' in running_stats and running_stats['global_total_steps'] > 0:
                        global_total_steps = running_stats['global_total_steps']
                        global_total_latency = running_stats.get('global_avg_latency_per_step', 0) * global_total_steps
                        global_total_input_tokens = int(running_stats.get('global_avg_input_tokens_per_step', 0) * global_total_steps)
                        global_total_output_tokens = int(running_stats.get('global_avg_output_tokens_per_step', 0) * global_total_steps)
                        print(f"[Resume] Loaded global stats: {global_total_steps} steps, {global_total_latency:.2f}s total latency")
                except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                    print(f"[Resume] Could not load global statistics: {e}")
        
        # Initialize debug info for visualization
        debug_info = {
            "experiment_name": config.EVAL.SPLIT,
            "episodes": []
        }
        
        # Add the initial forward-looking image (Direction 0)
        if '0' in images_list:
            chosen_images.append(images_list['0']['rgb'].copy())
            chosen_images_descriptions.append("Initial position: Agent standing at start point looking forward")
            nav_logger.info("Added initial forward-looking image to sequence")
        initial_low_rgb = _extract_low_level_rgb(observations[-1])
        if initial_low_rgb is not None:
            low_level_rgb_frames.append(initial_low_rgb)

        episode_ssa_trace = _new_ssa_episode_trace()
        
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()

            # ========== Resume: Skip already completed episodes ==========
            current_ep_id = current_episodes[0].episode_id
            if resume_enabled and (
                current_ep_id in completed_episode_ids
                or str(current_ep_id) in completed_episode_ids
            ):
                print(f"[Resume] Skipping already completed episode {current_ep_id}")
                # Mark as done and reset to next episode
                stats_episodes[current_ep_id] = None  # Placeholder to track skipped episodes
                observations[0] = envs.reset_at(0)[0]
                instruction, images_list = self.generate_input(observations[0])
                observations = extract_instruction_tokens(
                    observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)

                # Reset episode-level variables
                current_step = 0
                current_action_idx = 0
                nav_history = []
                chosen_images = []
                low_level_rgb_frames = []
                chosen_images_descriptions = []
                env_actions_history = []
                previous_position = None
                stuck_directions = set()
                last_chosen_vp = None
                episode_step_latencies = []
                episode_step_input_tokens = []
                episode_step_output_tokens = []
                ssa_controller.reset()
                episode_ssa_trace = _new_ssa_episode_trace()

                # Add initial image for next episode
                if '0' in images_list:
                    chosen_images.append(images_list['0']['rgb'].copy())
                    chosen_images_descriptions.append("Initial position: Agent standing at start point looking forward")
                reset_low_rgb = _extract_low_level_rgb(observations[0])
                if reset_low_rgb is not None:
                    low_level_rgb_frames.append(reset_low_rgb)
                continue

            positions = []; headings = []
            for ob_i in range(len(current_episodes)):
                agent_state_i = envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])
            # ==========Navigator start==========
            nav_logger.info(f"==================== The current episode id is {current_episodes[0].episode_id} ====================")
            nav_logger.info("Instruction: "+instruction)
            
            # Collect episode info for debug.json
            # Always create a new entry for each navigation attempt to avoid mixing steps from different runs
            episode_id = current_episodes[0].episode_id
            
            # Check if this is a continuation of the current episode or a new attempt
            # If it's the first step (current_step == 0), create a new episode entry
            if current_step == 0:
                episode_info = {
                    "episode_id": episode_id,
                    "scene_id": current_episodes[0].scene_id,
                    "instruction": instruction,
                    "steps": []  # Will store step-by-step data
                }
                debug_info["episodes"].append(episode_info)
                episode_ssa_trace = _new_ssa_episode_trace()
            # Otherwise, use the last episode entry (which should be the current one)
            else:
                episode_info = debug_info["episodes"][-1] if debug_info["episodes"] else None
                # Safety check: create new entry if somehow episode_info is None
                if episode_info is None:
                    episode_info = {
                        "episode_id": episode_id,
                        "scene_id": current_episodes[0].scene_id,
                        "instruction": instruction,
                        "steps": []
                    }
                    debug_info["episodes"].append(episode_info)
                    episode_ssa_trace = _new_ssa_episode_trace()
            actions, landmarks, landmark_list = "", "", []
            if instruction not in actions_cache.keys():
                nav_logger.info("[Cache MISS] Calling LLM to decompose instruction...")
                actions = navigator.get_actions(instruction)
                landmarks = navigator.get_landmarks(actions)
                action_list = actions.split("\n")
                # Extract landmarks for each action individually
                landmark_list = []
                for i, action in enumerate(action_list):
                    if action.strip():  # Skip empty actions
                        landmarks = navigator.get_landmarks(action)
                        landmark_list.append(landmarks.replace("- ", "").split("\n"))
                    else:
                        landmark_list.append("")
                        nav_logger.info(f"Action {i} is empty, skipping landmark extraction")
                actions_cache[instruction] = {
                    "actions": actions,
                    "landmarks": landmarks,
                    "landmark_list": landmark_list,
                }
                with open(actions_cache_path, "w", encoding="utf-8") as f2:
                    json.dump(actions_cache, f2, indent=2)
                nav_logger.info("[Cache SAVED] Instruction cached to disk")
            else:
                nav_logger.info("[Cache HIT] Reusing cached instruction decomposition")
                actions = actions_cache[instruction]["actions"]
                landmarks = actions_cache[instruction].get("landmarks", "")
                landmark_list = actions_cache[instruction].get("landmark_list", [])
                if not landmarks:
                    if ssa_enabled and landmark_list:
                        landmarks = "\n".join(
                            ", ".join(item) if isinstance(item, list) else str(item)
                            for item in landmark_list
                        )
                    else:
                        landmarks = navigator.get_landmarks(actions)
                        actions_cache[instruction]["landmarks"] = landmarks
                        with open(actions_cache_path, "w", encoding="utf-8") as f2:
                            json.dump(actions_cache, f2, indent=2)
                if not landmark_list and ssa_enabled:
                    landmark_list = []
                    for action in actions.split("\n"):
                        if action.strip():
                            action_landmarks = navigator.get_landmarks(action)
                            landmark_list.append(action_landmarks.replace("- ", "").split("\n"))
                        else:
                            landmark_list.append("")
                    actions_cache[instruction]["landmark_list"] = landmark_list
                    with open(actions_cache_path, "w", encoding="utf-8") as f2:
                        json.dump(actions_cache, f2, indent=2)

            action_list = actions.split("\n")
            if not landmark_list:
                landmark_list = [landmarks for _ in action_list]

            # Store sub-instructions and landmarks in episode_info for the first step
            if current_step == 0:  # First step after initialization
                nav_logger.info("Sub-instructions: "+str(action_list))
                nav_logger.info("Landmarks: " + (str(landmark_list) if ssa_enabled else landmarks))

                episode_info["sub_instructions"] = action_list
                episode_info["landmarks"] = landmark_list if ssa_enabled else landmarks

            # Preserve the original 3-step stopping budget when SSA is disabled.
            step_length = (
                self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
                if ssa_enabled
                else (6 if len(action_list) <= 6 else 8)
            )

            stop_flag = False
            backtrack_flag = False
            current_step += 1

            # Record step start time and reset LLM token accumulator
            step_start_time = time.time()
            navigator.llm.reset_step_tokens()

            nav_logger.info(f"-------------------- Step {current_step} --------------------")
            with torch.no_grad():
                # candidate waypoints prediction
                cand_rgb, cand_depth, \
                cand_direction, cand_mask, candidate_lengths, \
                batch_angles, batch_distances = self.policy.net( 
                    mode = "waypoint",
                    waypoint_predictor = self.waypoint_predictor,
                    observations = batch,
                    in_train = False,
                )
            
            images_dict, radius_dict, distance_dict = self.construct_image_dicts(batch_distances[-1], batch_angles, images_list)
            if current_action_idx < len(action_list):
                nav_logger.info("Current sub-instruction: "+action_list[current_action_idx]
                                +"Current landmarks: "+str(landmark_list[current_action_idx]))

            nav_logger.info("========== Get Observation ==========")
            observation, observe_dict = navigator.observe_environment(nav_logger, current_step, images_dict)
            
            # Prepare step data for debug.json
            current_landmarks = landmark_list[current_action_idx] if current_action_idx < len(landmark_list) else []
            step_data = {
                "step_index": current_step,
                "viewpoints": {},
                "current_action": action_list[current_action_idx] if current_action_idx < len(action_list) else "Finishing",
                "current_landmarks": current_landmarks
            }
            
            # Store all viewpoint images with base64 encoding and mark candidates
            for vp_id, vp_data in images_dict.items():
                # Get base64 data that's already encoded in generate_input
                rgb_base64 = vp_data.get('base64') if isinstance(vp_data, dict) else None

                step_data["viewpoints"][vp_id] = {
                    "rgb_base64": rgb_base64,
                    "is_chosen": False  # Will be updated after selection
                }
            
            if len(nav_history) == 0:
                history_traj = "Step 0 start position. "

            current_action = action_list[current_action_idx] if current_action_idx < len(action_list) else ""
            current_landmarks = landmark_list[current_action_idx] if current_action_idx < len(landmark_list) else []
            ssa_subtask_active = ssa_enabled and _ssa_stair_subtask_active(current_action, current_landmarks)
            ssa_takeover_requested = False
            ssa_takeover_direction = "unknown"
            ssa_pre_align_yaw_rad = None

            if not stop_flag:                               
                nav_logger.info("========== Next Action Prediction ==========")
                if current_action_idx + 1 < len(action_list):
                    next_instruction = action_list[current_action_idx + 1]
                else:
                    next_instruction = 'Stop.'

                # Stuck filtering is part of the extended 3-step controller; keep the
                # no-SSA path aligned with the original global multi-decision flow.
                if ssa_enabled:
                    filtered_observe_dict = {k: v for k, v in observe_dict.items() if k not in stuck_directions}
                    filtered_images_dict = {k: v for k, v in images_dict.items() if k not in stuck_directions}
                else:
                    filtered_observe_dict = observe_dict
                    filtered_images_dict = images_dict

                if len(filtered_observe_dict) == 0:
                    nav_logger.error("All directions are stuck! Using original observations.")
                    filtered_observe_dict = observe_dict
                    filtered_images_dict = images_dict
                else:
                    nav_logger.info(f"Filtered out {len(stuck_directions)} stuck directions: {stuck_directions}")

                if ssa_enabled:
                    next_vp, thought, completion_estimation, gpt_interaction = navigator.move_to_next_vp_single(
                        nav_logger,
                        action_list[current_action_idx],
                        landmark_list[current_action_idx],
                        history_traj,
                        observation,
                        filtered_observe_dict,
                        filtered_images_dict,
                        next_instruction,
                    )
                else:
                    nav_logger.info("========== Estimate Completion Progress ==========")
                    completion_estimation = navigator.estimate_completion(nav_logger, actions, landmarks, history_traj)
                    predictions, thoughts, completion_estimations, break_flag = navigator.move_to_next_vp(
                        nav_logger,
                        instruction,
                        landmarks,
                        history_traj,
                        observation,
                        filtered_observe_dict,
                        filtered_images_dict,
                        next_instruction=next_instruction,
                    )
                    nav_logger.info("========== Thought ==========")
                    fused_pred_thought = navigator.thought_fusion(nav_logger, predictions, thoughts)
                    nav_logger.info("========== Test Decision ==========")
                    next_vp, thought, error_number = navigator.test_decisions(
                        nav_logger,
                        fused_pred_thought,
                        observation,
                        instruction,
                        error_number,
                        filtered_observe_dict,
                    )
                    gpt_interaction = {
                        "mode": "original_global_navigation",
                        "completion_estimation": completion_estimation,
                        "completion_estimations": completion_estimations,
                    }
                requested_vp = next_vp
                next_vp, remapped = select_executable_candidate(
                    next_vp,
                    radius=radius_dict,
                    distance=distance_dict,
                    observations=filtered_observe_dict,
                )
                if next_vp is None:
                    nav_logger.warning(
                        "No executable waypoint candidate remains after filtering; stopping episode to avoid invalid Habitat action"
                    )
                    stop_flag = True
                    dones[0] = True
                    next_vp = _resolve_valid_viewpoint(
                        requested_vp,
                        filtered_observe_dict,
                        nav_logger,
                        context="3-step viewpoint selection",
                    )
                elif remapped:
                    nav_logger.warning(
                        f"Predicted viewpoint {requested_vp} is not executable; fallback to viewpoint {next_vp}"
                    )

                # Track the chosen viewpoint
                last_chosen_vp = next_vp

                # Update step data with chosen viewpoint and GPT interaction
                step_data["chosen_viewpoint"] = next_vp
                step_data["gpt_interaction"] = gpt_interaction
                if next_vp in step_data["viewpoints"]:
                    step_data["viewpoints"][next_vp]["is_chosen"] = True

                step_data["ssa_subtask_active"] = bool(ssa_subtask_active)
                step_data["ssa_viewpoint"] = next_vp
                step_data["ssa_view_yaw_deg"] = (
                    _ssa_view_yaw_deg(radius_dict[next_vp]) if next_vp in radius_dict else 0.0
                )
                selected_ssa_view = filtered_images_dict.get(next_vp)
                if not ssa_enabled:
                    ssa_proposal = {"available": False, "reason": "disabled"}
                elif selected_ssa_view is None:
                    ssa_proposal = {
                        "available": False,
                        "reason": "missing_selected_view",
                    }
                else:
                    ssa_proposal = ssa_controller.update_proposal(
                        instruction=instruction,
                        previous_output=current_action,
                        previous_plan=" ".join(str(item) for item in current_landmarks if item),
                        rgb=np.asarray(selected_ssa_view["rgb"]),
                        depth=np.asarray(selected_ssa_view["depth"]),
                        view_yaw_deg=step_data["ssa_view_yaw_deg"],
                        delegate_infer_fn=lambda *_: '{"delegate": false, "direction": "unknown", "reason": "unused"}',
                        delegate_image_infer_fn=navigator.llm.gpt_infer_with_images,
                        delegate_image=selected_ssa_view,
                        delegate_current_stage=current_action,
                        delegate_history=history_traj,
                        delegate_observation_hint=filtered_observe_dict.get(next_vp, ""),
                        current_position=envs.call_at(0, "get_agent_info", {}).get("position"),
                        oracle_episode=current_episodes[0],
                    )
                step_data["ssa_available"] = bool(ssa_proposal.get("available", False))
                step_data["ssa_delegated"] = False
                step_data["ssa_reason"] = str(ssa_proposal.get("reason", ""))
                if ssa_enabled:
                    ssa_controller.record_step_proposal(
                        step=current_step,
                        available=step_data["ssa_available"],
                        reason=step_data["ssa_reason"],
                        viewpoint=step_data["ssa_viewpoint"],
                        view_yaw_deg=step_data["ssa_view_yaw_deg"],
                    )
                    ssa_estimate = SSAController._compact_estimate(ssa_proposal.get("estimate"))
                    if ssa_estimate:
                        step_data["ssa_estimate"] = ssa_estimate
                        episode_ssa_trace["proposal_estimates"].append(
                            {
                                "step": current_step,
                                "available": step_data["ssa_available"],
                                "reason": step_data["ssa_reason"],
                                "viewpoint": step_data["ssa_viewpoint"],
                                "view_yaw_deg": step_data["ssa_view_yaw_deg"],
                                "estimate": ssa_estimate,
                            }
                        )
                    nav_logger.info(
                        f"[SSA] step={current_step} episode={episode_id} "
                        f"subtask_active={step_data['ssa_subtask_active']} "
                        f"viewpoint={step_data['ssa_viewpoint']} "
                        f"view_yaw_deg={step_data['ssa_view_yaw_deg']:.1f} "
                        f"available={step_data['ssa_available']} reason={step_data['ssa_reason']}"
                    )

                if ssa_enabled and step_data["ssa_available"]:
                    episode_ssa_trace["proposal_seen"] = True
                    episode_ssa_trace["available_steps"].append(current_step)
                elif ssa_enabled and step_data["ssa_reason"]:
                    episode_ssa_trace["rejection_reasons"].append(
                        {
                            "step": current_step,
                            "reason": step_data["ssa_reason"],
                        }
                    )

                if ssa_enabled and ssa_proposal.get("available", False):
                    current_stage_text = action_list[current_action_idx] if current_action_idx < len(action_list) else ""
                    delegate_info = ssa_proposal.get("delegate", {}) if isinstance(ssa_proposal.get("delegate"), dict) else {}
                    delegate_reason = str(delegate_info.get("decision_reason", "vlm_gate"))
                    ssa_takeover_requested = True
                    ssa_takeover_direction = str(ssa_proposal.get("direction", "unknown"))
                    ssa_pre_align_yaw_rad = radius_dict[next_vp] if next_vp in radius_dict else None
                    step_data["ssa_delegated"] = True
                    step_data["ssa_delegate_decision"] = "delegate"
                    step_data["ssa_plan_reason"] = "closed_loop_ready"
                    episode_ssa_trace["delegated"] = True
                    ssa_controller.record_delegate_decision(
                        step=current_step,
                        delegated=True,
                        current_stage=current_stage_text,
                        history=history_traj,
                        observation_hint=filtered_observe_dict.get(next_vp, ""),
                        prompt_has_rgb=bool(delegate_info.get("prompt_has_rgb", False)),
                        raw_response=str(delegate_info.get("raw_response", "")),
                        reason=delegate_reason,
                        direction=ssa_takeover_direction,
                    )
                    ssa_controller.record_plan_outcome(
                        step=current_step,
                        accepted=True,
                        reason="closed_loop_ready",
                        planned_actions=0,
                    )
                    nav_logger.info(
                        f"[SSA] step={current_step} episode={episode_id} delegated=yes mode=closed_loop direction={ssa_takeover_direction}"
                    )
                elif ssa_enabled:
                    current_stage_text = action_list[current_action_idx] if current_action_idx < len(action_list) else ""
                    delegate_info = ssa_proposal.get("delegate", {}) if isinstance(ssa_proposal.get("delegate"), dict) else {}
                    if delegate_info:
                        ssa_controller.record_delegate_decision(
                            step=current_step,
                            delegated=False,
                            current_stage=current_stage_text,
                            history=history_traj,
                            observation_hint=filtered_observe_dict.get(next_vp, ""),
                            prompt_has_rgb=bool(delegate_info.get("prompt_has_rgb", False)),
                            raw_response=str(delegate_info.get("raw_response", "")),
                            reason=str(delegate_info.get("decision_reason", step_data["ssa_reason"])),
                            direction=str(delegate_info.get("direction", "unknown")),
                        )

                # Add step data to episode info
                episode_info["steps"].append(step_data)

                if ssa_takeover_requested:
                    nav_logger.info("Delaying history/GIF update until SSA final observation is available")
                else:
                    # Save history for a normal VLM-selected waypoint.
                    curr_observe = filtered_observe_dict[next_vp]
                    nav_logger.info("========== save history ==========")
                    nav_history = navigator.save_history(nav_logger, current_step, next_vp, thought, curr_observe, nav_history)

                    # Only add image if not stuck (will be determined after env.step)
                    # For now, we'll add it and potentially remove it later if stuck
                    chosen_images.append(filtered_images_dict[next_vp]['rgb'].copy())

                    # Add description for this image
                    angle_deg = np.rad2deg(radius_dict[next_vp]) if next_vp in radius_dict else 0
                    direction_desc = ""
                    if -15 <= angle_deg <= 15:
                        direction_desc = "forward"
                    elif 15 < angle_deg <= 45:
                        direction_desc = "front-left (30°)"
                    elif 45 < angle_deg <= 75:
                        direction_desc = "left (60°)"
                    elif 75 < angle_deg <= 105:
                        direction_desc = "left (90°)"
                    elif 105 < angle_deg <= 135:
                        direction_desc = "back-left (120°)"
                    elif 135 < angle_deg <= 165:
                        direction_desc = "back-left (150°)"
                    elif angle_deg > 165 or angle_deg < -165:
                        direction_desc = "backward (180°)"
                    elif -165 <= angle_deg < -135:
                        direction_desc = "back-right (150°)"
                    elif -135 <= angle_deg < -105:
                        direction_desc = "back-right (120°)"
                    elif -105 <= angle_deg < -75:
                        direction_desc = "right (90°)"
                    elif -75 <= angle_deg < -45:
                        direction_desc = "right (60°)"
                    elif -45 <= angle_deg < -15:
                        direction_desc = "front-right (30°)"

                    image_desc = f"Step {current_step}: Agent at previous position looking {direction_desc} towards chosen next viewpoint"
                    chosen_images_descriptions.append(image_desc)
                    nav_logger.info(f"Added image with description: {image_desc}")

                nav_logger.info("========== Review History after navigation ==========")
                history_traj = navigator.review_history(nav_logger, nav_history) if len(nav_history) > 0 else "Step 0 start position. "

                nav_logger.info("========== Estimate Completion Progress ==========")
                nav_logger.info(f"Completion estimation result: '{completion_estimation}'")
                step_data["estimation_result"] = completion_estimation

                if ssa_enabled and completion_estimation == "Yes":
                    nav_logger.info("========== Navigation Decision Agent ==========")
                    
                    # Import the decision agent
                    from vlnce_baselines.common.navigator.decision_agent import (
                        NavigationDecisionAgent, 
                        DecisionContext,
                        NavigationDecision
                    )
                    
                    # Initialize decision agent with enabled abilities from config
                    enabled_abilities = getattr(config, 'NAVIGATION_AGENT', {}).get('ENABLED_META_ABILITIES',
                                               ["continue", "stay", "backtrack", "look_around"])
                    decision_agent = NavigationDecisionAgent(navigator.llm, enabled_abilities, nav_logger)
                    
                    # Prepare context for decision
                    actions_so_far = " ".join(action_list[:current_action_idx+1])
                    current_action = action_list[current_action_idx] if current_action_idx < len(action_list) else ""
                    current_landmarks = landmark_list[current_action_idx] if current_action_idx < len(landmark_list) else []
                    
                    decision_context = DecisionContext(
                        chosen_images=chosen_images,
                        current_action=current_action,
                        current_landmarks=current_landmarks,
                        actions_completed=actions_so_far,
                        history_trajectory=history_traj,
                        current_step=current_step,
                        total_actions=len(action_list),
                        current_action_idx=current_action_idx,
                        image_descriptions=chosen_images_descriptions  # Pass the descriptions
                    )
                    
                    # Get decision from agent (use informed decision for better understanding)
                    # You can switch between make_decision() and make_informed_decision()
                    use_code_analysis = getattr(config, 'USE_CODE_ANALYSIS', True)
                    if use_code_analysis:
                        decision, confidence, reasoning, decision_interaction = decision_agent.make_informed_decision_with_capture(decision_context)
                    else:
                        decision, confidence, reasoning, decision_interaction = decision_agent.make_decision_with_capture(decision_context)
                    nav_logger.info(f"Agent decision: {decision.value} (confidence: {confidence}/10)")
                    nav_logger.info(f"Agent reasoning: {reasoning}")

                    # Store decision agent interaction in step data
                    if decision_interaction:
                        step_data["decision_agent_interaction"] = decision_interaction
                        nav_logger.info("Decision agent interaction data stored in step_data")
                    else:
                        nav_logger.warning("No decision interaction data received")
                    
                    # Execute decision
                    if decision == NavigationDecision.CONTINUE:
                        # Move to next sub-instruction
                        if current_action_idx < len(action_list) - 1:
                            nav_logger.info("Agent decided to continue to next sub-instruction")
                            current_action_idx += 1
                            nav_history = []
                            history_traj = "Step 0 start position. "
                            # Clear stuck directions when moving to new instruction
                            stuck_directions.clear()
                            nav_logger.info("Cleared stuck directions for new sub-instruction")
                        else:
                            nav_logger.info("Completed all sub-instructions")
                            stop_flag = True
                    
                    elif decision == NavigationDecision.LOOK_AROUND:
                        # Explore more viewpoints for information
                        nav_logger.info("Agent decided to look around for more information")
                        current_action = action_list[current_action_idx] if current_action_idx < len(action_list) else ""
                        current_landmarks = landmark_list[current_action_idx] if current_action_idx < len(landmark_list) else ""
                        
                        # Gather enhanced observations
                        enhanced_observation, enhanced_observe_dict = navigator.look_around(
                            nav_logger, current_step, images_dict, current_action, current_landmarks, history_traj
                        )
                        
                        observation = enhanced_observation
                        observe_dict = enhanced_observe_dict
                        nav_logger.info("Enhanced observations gathered from all viewpoints")
                    
                    elif decision == NavigationDecision.BACKTRACK:
                        # Go back to previous position
                        nav_logger.info("Agent decided to backtrack")
                        if len(env_actions_history) > 0:
                            # Execute reverse action
                            last_action = env_actions_history.pop()
                            reverse_action = self._create_reverse_action(last_action)
                            nav_logger.info(f"Executing reverse action: {reverse_action}")
                            
                            outputs = envs.step([reverse_action])
                            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                            step_low_rgb = _extract_low_level_rgb(observations[-1])
                            if step_low_rgb is not None:
                                low_level_rgb_frames.append(step_low_rgb)

                            for j, ob in enumerate(observations):
                                new_positions = ob.pop('positions')
                                new_collisions = ob.pop('collisions')
                                if len(new_positions) > 0:
                                    previous_position = new_positions[-1]
                                envs.call_at(
                                    j,
                                    'change_current_path',
                                    {
                                        'new_path': new_positions,
                                        'collisions': new_collisions,
                                    },
                                )

                            instruction, images_list = self.generate_input(observations[-1])

                            step_latency = time.time() - step_start_time
                            step_tokens = navigator.llm.get_step_tokens()
                            episode_step_latencies.append(step_latency)
                            episode_step_input_tokens.append(step_tokens['input_tokens'])
                            episode_step_output_tokens.append(step_tokens['output_tokens'])
                            nav_logger.info(
                                f"Step {current_step} backtrack stats: latency={step_latency:.2f}s, "
                                f"input_tokens={step_tokens['input_tokens']}, "
                                f"output_tokens={step_tokens['output_tokens']}"
                            )
                            
                            # Update history
                            if len(nav_history) > 0:
                                nav_history.pop()
                            if len(chosen_images) > 1:  # Keep the initial image
                                chosen_images.pop()
                                chosen_images_descriptions.pop()
                            
                            nav_logger.info("Successfully backtracked")
                            backtrack_flag = True
                            if current_step >= step_length:
                                dones[0] = True
                        else:
                            nav_logger.info("Cannot backtrack - no previous actions")
                    
                    elif decision == NavigationDecision.STAY:
                        # Continue with current sub-instruction
                        nav_logger.info("Agent decided to stay with current sub-instruction")
                    
                    # Check if navigation should stop
                    if decision_agent.should_stop_navigation(decision, decision_context):
                        nav_logger.info("Agent determined navigation should stop")
                        stop_flag = True

                elif ssa_enabled and completion_estimation == "No":
                    nav_logger.info("Instruction not yet completed - continuing with current instruction")
                    nav_logger.info("Skipping decision agent visualization - continuing with standard navigation")

                elif ssa_enabled:
                    nav_logger.error(f"Unexpected estimation result: {completion_estimation}")

            try:
                if not stop_flag:
                    ssa_takeover_finished_episode = False
                    if ssa_takeover_requested and not backtrack_flag:
                        def _ssa_restore_instruction(observation_item):
                            if isinstance(observation_item, (list, tuple)) and observation_item:
                                observation_item = observation_item[-1]
                            if isinstance(observation_item, dict):
                                restored = dict(observation_item)
                                inst = restored.get("instruction")
                                if isinstance(inst, dict):
                                    if "text" not in inst:
                                        restored["instruction"] = {**inst, "text": instruction}
                                else:
                                    restored["instruction"] = {"text": instruction}
                                    if inst is not None:
                                        restored["instruction"]["tokens"] = inst
                                return restored
                            return observation_item

                        def _ssa_get_forward_view(observation_item):
                            observation_item = _ssa_restore_instruction(observation_item)
                            _, ssa_images = self.generate_input(observation_item)
                            ssa_front = ssa_images.get("0") if isinstance(ssa_images, dict) else None
                            if ssa_front is None:
                                raise RuntimeError("SSA takeover requires a forward RGB-D view")
                            return np.asarray(ssa_front["rgb"]), np.asarray(ssa_front["depth"])

                        ssa_segment = proposal_oracle_segment(
                            ssa_proposal,
                            required=(
                                bool(getattr(config, "SSA_EXPERT_ENTRY_POSE", False))
                                or bool(getattr(config, "SSA_ORACLE_EXPERT_REPLAY", False))
                                or bool(getattr(config, "SSA_ORACLE_EXIT_ENABLE", False))
                            ),
                            context="3-step-Nav",
                        )
                        takeover_kwargs = dict(
                            envs=envs,
                            env_index=0,
                            controller=ssa_controller,
                            initial_observation=_ssa_restore_instruction(observations[-1]),
                            get_forward_view=_ssa_get_forward_view,
                            direction=ssa_takeover_direction,
                            step=current_step,
                        )
                        if getattr(config, "SSA_ORACLE_EXPERT_REPLAY", False):
                            episode_record = expert_record_for_episode(
                                self.gt_data, current_episodes[0].episode_id
                            )
                            takeover = execute_oracle_expert_replay(
                                **takeover_kwargs,
                                oracle_segment=ssa_segment,
                                expert_actions=expert_actions_for_segment(episode_record, ssa_segment),
                            )
                        else:
                            takeover = execute_ssa_takeover(
                                **takeover_kwargs,
                                pre_align_yaw_rad=ssa_pre_align_yaw_rad,
                                oracle_exit=ssa_segment if getattr(config, "SSA_ORACLE_EXIT_ENABLE", False) else None,
                                expert_entry_pose=ssa_segment if getattr(config, "SSA_EXPERT_ENTRY_POSE", False) else None,
                                env_turn_degrees=float(config.TASK_CONFIG.SIMULATOR.TURN_ANGLE),
                            )
                        nav_logger.info(f"[SSA] takeover finished | success={takeover.success} reason={takeover.reason} actions={takeover.actions_executed}")
                        episode_ssa_trace["takeover_success"] = bool(takeover.success)
                        episode_ssa_trace["takeover_reason"] = str(takeover.reason)
                        step_data["ssa_takeover_success"] = bool(takeover.success)
                        step_data["ssa_takeover_reason"] = str(takeover.reason)
                        step_latency = time.time() - step_start_time
                        step_tokens = navigator.llm.get_step_tokens()
                        episode_step_latencies.append(step_latency)
                        episode_step_input_tokens.append(step_tokens['input_tokens'])
                        episode_step_output_tokens.append(step_tokens['output_tokens'])

                        observations = takeover.observations
                        dones = takeover.dones
                        infos = takeover.infos
                        env_actions_history.clear()
                        stuck_directions.clear()
                        last_chosen_vp = None
                        agent_info = envs.call_at(0, "get_agent_info", {})
                        previous_position = list(agent_info["position"])
                        observations = [_ssa_restore_instruction(ob) for ob in observations]
                        low_level_rgb_frames.extend(
                            [np.asarray(frame).astype(np.uint8).copy() for frame in takeover.rgb_frames]
                        )
                        instruction, images_list = self.generate_input(observations[-1])
                        final_ssa_view = images_list.get("0") if isinstance(images_list, dict) else None
                        if final_ssa_view is not None:
                            _, final_observe_dict = navigator.observe_environment(
                                nav_logger,
                                current_step,
                                {"0": final_ssa_view},
                            )
                            ssa_thought = ssa_controller.latest_handoff_text()
                            nav_logger.info("========== save SSA history ==========")
                            nav_history = navigator.save_history(
                                nav_logger,
                                current_step,
                                "0",
                                ssa_thought,
                                final_observe_dict["0"],
                                nav_history,
                            )
                            chosen_images.append(final_ssa_view["rgb"].copy())
                            chosen_images_descriptions.append(
                                f"Step {current_step}: SSA takeover final forward view after local stair alignment"
                            )
                            step_data["ssa_history_recorded"] = True
                        else:
                            nav_logger.warning("SSA final forward view missing; history/GIF high frame not updated")
                        observations = extract_instruction_tokens(
                            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
                        )
                        batch = batch_obs(observations, self.device)
                        batch = apply_obs_transforms_batch(batch, obs_transforms)

                        if current_step >= step_length:
                            dones[0] = True
                        if not dones[0]:
                            continue
                        ssa_takeover_finished_episode = True

                    if not backtrack_flag and not ssa_takeover_finished_episode:
                        env_actions = []
                        env_actions.append({'action':
                            {'action': 4,
                            'action_args':{
                                'angle': radius_dict[next_vp],
                                'distance': distance_dict[next_vp],
                            }}})
                        nav_logger.info(f"The final env action: {env_actions}")
                        # Record the action for potential backtracking
                        env_actions_history.append(env_actions[0])
                        outputs = envs.step(env_actions)

                        # Step completed - record step statistics
                        step_latency = time.time() - step_start_time
                        step_tokens = navigator.llm.get_step_tokens()
                        episode_step_latencies.append(step_latency)
                        episode_step_input_tokens.append(step_tokens['input_tokens'])
                        episode_step_output_tokens.append(step_tokens['output_tokens'])
                        nav_logger.info(f"Step {current_step} stats: latency={step_latency:.2f}s, input_tokens={step_tokens['input_tokens']}, output_tokens={step_tokens['output_tokens']}")

                        observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                        step_low_rgb = _extract_low_level_rgb(observations[-1])
                        if step_low_rgb is not None:
                            low_level_rgb_frames.append(step_low_rgb)
                        instruction, images_list = self.generate_input(observations[-1])

                        # Check if agent is stuck (position hasn't changed)
                        is_stuck = False
                        if current_step == step_length:
                            dones[0] = True
                        else:
                            for j, ob in enumerate(observations):
                                new_positions = ob.pop('positions')
                                new_collisions = ob.pop('collisions')

                                # The extended stuck-recovery feedback changes the next LLM
                                # history. Keep it out of the no-SSA path.
                                if ssa_enabled and previous_position is not None and len(new_positions) > 0:
                                    current_position = new_positions[-1]
                                    position_diff = np.linalg.norm(np.array(current_position) - np.array(previous_position))

                                    if position_diff < 0.1:  # Threshold: if moved less than 0.1 meters, consider stuck
                                        is_stuck = True
                                        nav_logger.warning(f"Agent is STUCK! Position changed by only {position_diff:.4f}m")
                                        nav_logger.warning(f"Previous position: {previous_position}")
                                        nav_logger.warning(f"Current position: {current_position}")

                                        # Add the stuck direction to the blacklist
                                        if last_chosen_vp is not None:
                                            stuck_directions.add(last_chosen_vp)
                                            nav_logger.warning(f"Added direction '{last_chosen_vp}' to stuck directions blacklist")
                                            nav_logger.warning(f"Total stuck directions: {stuck_directions}")

                                        # Remove the last added image and description since we're stuck
                                        if len(chosen_images) > 0:
                                            removed_img = chosen_images.pop()
                                            nav_logger.warning(f"Removed stuck observation image (total images now: {len(chosen_images)})")
                                        if len(chosen_images_descriptions) > 0:
                                            removed_desc = chosen_images_descriptions.pop()
                                            nav_logger.warning(f"Removed stuck image description: {removed_desc}")

                                        # Also remove from nav_history since this movement failed
                                        if len(nav_history) > 0:
                                            nav_history.pop()
                                            nav_logger.warning("Removed last navigation history entry due to stuck")
                                    else:
                                        nav_logger.info(f"Agent moved {position_diff:.4f}m successfully")

                                    # Update previous position
                                    previous_position = current_position
                                elif len(new_positions) > 0:
                                    previous_position = new_positions[-1]

                                envs.call_at(j,
                                    'change_current_path',
                                    {'new_path': new_positions,
                                    'collisions': new_collisions}
                                )
                else:
                    dones[0] = True
                
                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8, device=self.device)
                
                for i in range(envs.num_envs):
                    
                    if not dones[i]:
                        continue
                    
                    ep_id = str(envs.current_episodes()[i].episode_id)
                    high_gif_frames = [np.asarray(img).astype(np.uint8) for img in chosen_images]
                    if save_episode_gif:
                        _save_episode_rgb_gif(high_rgb_gif_dir, ep_id, high_gif_frames, nav_logger, gif_max_width, gif_duration)
                        _save_episode_rgb_gif(low_rgb_gif_dir, ep_id, list(low_level_rgb_frames), nav_logger, gif_max_width, gif_duration)

                    current_step = 0
                    current_action_idx = 0
                    nav_history = []
                    chosen_images = []
                    low_level_rgb_frames = []
                    chosen_images_descriptions = []
                    env_actions_history = []
                    backtrack_flag = False
                    previous_position = None  # Reset position tracking for new episode
                    stuck_directions = set()  # Reset stuck directions for new episode
                    last_chosen_vp = None  # Reset last chosen viewpoint for new episode
                    ssa_trace_path = ssa_controller.save_episode_trace(config.RESULTS_DIR, ep_id)
                    ssa_summary = ssa_controller.episode_summary()
                    ssa_trace = ssa_controller.episode_trace()
                    episode_info["ssa_summary"] = ssa_summary
                    episode_info["ssa_trace_path"] = ssa_trace_path
                    nav_logger.info(
                        f"[SSA] episode summary | episode={ep_id} "
                        f"proposal_seen={episode_ssa_trace['proposal_seen']} "
                        f"delegated={episode_ssa_trace['delegated']} "
                        f"takeover_success={episode_ssa_trace['takeover_success']} "
                        f"takeover_reason={episode_ssa_trace['takeover_reason'] or 'none'} "
                        f"available_steps={episode_ssa_trace['available_steps']} "
                        f"delegate_declined_steps={episode_ssa_trace['delegate_declined_steps']} "
                        f"rejection_reasons={episode_ssa_trace['rejection_reasons']}"
                    )
                    ssa_controller.reset()
                    episode_ssa_trace = _new_ssa_episode_trace()

                    # Add initial image for new episode
                    if '0' in images_list:
                        chosen_images.append(images_list['0']['rgb'].copy())
                        chosen_images_descriptions.append("Initial position: Agent standing at start point looking forward")
                        nav_logger.info("Added initial forward-looking image for new episode")
                    next_init_low_rgb = _extract_low_level_rgb(observations[i])
                    if next_init_low_rgb is not None:
                        low_level_rgb_frames.append(next_init_low_rgb)
                    info = infos[i]
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    gt_path = np.array(self.gt_data[ep_id]['locations']).astype(float)
                    if 'current_path' in envs.current_episodes()[i].info.keys():
                        positions_ = np.array(envs.current_episodes()[i].info['current_path']).astype(float)
                        collisions_ = np.array(envs.current_episodes()[i].info['collisions'])
                        assert collisions_.shape[0] == positions_.shape[0] - 1
                    else:
                        positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(float)
                        collisions_ = np.zeros(max(len(positions_) - 1, 0), dtype=float)
                    positions_ = metric_positions(positions_)
                    distance = np.array(info['position']['distance']).astype(float)
                    metric['distance_to_goal'] = distance[-1]
                    metric['success'] = 1. if distance[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distance <= 3.).any() else 0.
                    metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1],axis=1).sum()
                    metric['collisions'] = float(collisions_.mean()) if collisions_.size else 0.0
                    gt_length = distance[0]
                    metric['spl'] = metric['success']*gt_length/max(gt_length,metric['path_length'])

                    act_con_path = positions_
                    gt_con_path = np.array(gt_path).astype(float)
                    dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                    nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))

                    metric['ndtw'] = nDTW
                    metric["ssa_summary"] = ssa_summary
                    metric["ssa_trace_path"] = ssa_trace_path

                    # Calculate episode-level step statistics (average per step for this episode)
                    if len(episode_step_latencies) > 0:
                        metric['avg_latency_per_step'] = sum(episode_step_latencies) / len(episode_step_latencies)
                        metric['avg_input_tokens_per_step'] = sum(episode_step_input_tokens) / len(episode_step_input_tokens)
                        metric['avg_output_tokens_per_step'] = sum(episode_step_output_tokens) / len(episode_step_output_tokens)
                        metric['total_latency'] = sum(episode_step_latencies)
                        metric['total_input_tokens'] = sum(episode_step_input_tokens)
                        metric['total_output_tokens'] = sum(episode_step_output_tokens)

                        # Update global statistics for overall average calculation
                        global_total_latency += sum(episode_step_latencies)
                        global_total_input_tokens += sum(episode_step_input_tokens)
                        global_total_output_tokens += sum(episode_step_output_tokens)
                        global_total_steps += len(episode_step_latencies)

                        nav_logger.info(f"Episode stats: avg_latency={metric['avg_latency_per_step']:.2f}s/step, "
                                       f"avg_input_tokens={metric['avg_input_tokens_per_step']:.0f}/step, "
                                       f"avg_output_tokens={metric['avg_output_tokens_per_step']:.0f}/step")
                    else:
                        metric['avg_latency_per_step'] = 0.0
                        metric['avg_input_tokens_per_step'] = 0
                        metric['avg_output_tokens_per_step'] = 0
                        metric['total_latency'] = 0.0
                        metric['total_input_tokens'] = 0
                        metric['total_output_tokens'] = 0

                    # Reset episode-level statistics for next episode
                    episode_step_latencies = []
                    episode_step_input_tokens = []
                    episode_step_output_tokens = []

                    nav_diag = navigator.diagnostics() if hasattr(navigator, "diagnostics") else {}
                    metric["vlm_timeouts"] = int(nav_diag.get("vlm_timeouts", 0) or 0)
                    metric["vlm_parse_errors"] = int(nav_diag.get("vlm_parse_errors", 0) or 0)

                    stats_episodes[current_episodes[i].episode_id] = metric
                    nav_logger.info(
                        format_episode_metric(
                            current_episodes[i].episode_id,
                            metric,
                            stats=stats_episodes,
                            total=episodes_to_eval,
                        )
                    )

                    # Find current episode debug info to save with results
                    current_episode_debug = None
                    episode_id_str = str(current_episodes[i].episode_id)
                    for ep_info in debug_info["episodes"]:
                        if str(ep_info.get("episode_id")) == episode_id_str:
                            current_episode_debug = ep_info
                            break

                    # Save individual episode result immediately (includes global averages)
                    self._save_episode_result(current_episodes[i].episode_id, metric, config, current_episode_debug,
                                             global_total_latency, global_total_input_tokens,
                                             global_total_output_tokens, global_total_steps)
                    ssa_trajectory = []
                    for result in ssa_trace.get("takeover_results", []) or []:
                        ssa_trajectory.extend(result.get("ssa_trajectory", []) or [])
                    save_trajectory_debug(
                        output_dir=config.EVAL_CKPT_PATH_DIR,
                        episode_id=str(current_episodes[i].episode_id),
                        payload={
                            "episode_id": str(current_episodes[i].episode_id),
                            "scene_id": current_episodes[i].scene_id,
                            "metric": metric,
                            "start_position": positions_[0].tolist() if len(positions_) else [],
                            "goal_position": gt_path[-1].tolist() if len(gt_path) else [],
                            "agent_trajectory": [
                                {"step": int(j), "source": "agent", "position": pos.tolist()}
                                for j, pos in enumerate(positions_)
                            ],
                            "ssa_trajectory": ssa_trajectory,
                            "expert_trajectory": gt_path.tolist(),
                            "ssa_trace": ssa_trace,
                        },
                    )
                    if hasattr(navigator, "reset_diagnostics"):
                        navigator.reset_diagnostics()

                    observations[i] = envs.reset_at(i)[0]
                    instruction, images_list = self.generate_input(observations[i])
                    
                    if config.use_pbar:
                        pbar.update()
                    else:
                        logger.info(
                            log_str.format(
                                evaluated=len(stats_episodes),
                                total=episodes_to_eval,
                                time=round(time.time() - start_time),
                            )
                        )
                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)   
                
                envs_to_pause = []
                next_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if next_episodes[i].episode_id in stats_episodes:
                        envs_to_pause.append(i)

                headings = torch.tensor(headings)
                (
                    envs,
                    not_done_masks,
                    headings,  
                    batch,
                    _,
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    not_done_masks,
                    headings,
                    batch,
                )
                headings = headings.tolist()
            except Exception:
                nav_logger.exception("Fatal error in 3-step navigation loop")
                envs.close()
                raise
        envs.close()
        if config.use_pbar:
            pbar.close()
        if self.world_size > 1:
            distr.barrier()
        split = config.TASK_CONFIG.DATASET.SPLIT
        episode_results_filename = (
            f"episode_results_{split}_r{self.local_rank}_w{self.world_size}.json"
        )
        merged_stats = load_episode_metrics(config.RESULTS_DIR, episode_results_filename)
        merged_stats.update({str(key): value for key, value in stats_episodes.items() if isinstance(value, dict)})
        if merged_stats:
            stats_episodes = merged_stats

        valid_stats = [value for value in stats_episodes.values() if isinstance(value, dict)]
        num_episodes = len(valid_stats)
        if num_episodes == 0:
            logger.info("No newly evaluated episodes with metrics were produced in this run.")
            return
        aggregated_stats = aggregate_numeric_metrics(stats_episodes)
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_stats}")
            for k,v in aggregated_stats.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_stats[k] = v

        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_episodes, f, indent=4)
        
        # Save debug.json with episode information
        debug_fname = os.path.join(
            config.RESULTS_DIR,
            "debug.json"
        )
        with open(debug_fname, "w") as f:
            json.dump(debug_info, f, indent=4)
        nav_logger.info(f"Saved debug info to {debug_fname}")

        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_stats, f, indent=4)

            logger.info(f"Episodes evaluated: {total}")
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.6f}")
        
    def collect_val_traj(self):
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        with gzip.open(
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                split=split)
        ) as f:
            gt_data = json.load(f)
        self.gt_data = gt_data
        trajectories = gt_data
        self.trajectories = gt_data
        trajectories = list(trajectories.keys())[self.config.local_rank::self.config.GPU_NUMBERS]
        # Apply cross-floor filter if EPISODES_ALLOWED is explicitly set
        allowed = self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED
        if allowed is not None:
            trajectories = filter_ids_by_cross_floor(trajectories, allowed)
        return trajectories

    def _save_episode_result(self, episode_id, metric, config, debug_episode_info=None,
                             global_total_latency=0.0, global_total_input_tokens=0,
                             global_total_output_tokens=0, global_total_steps=0):
        """Save individual episode result to JSON file immediately after evaluation."""
        if not config.EVAL.SAVE_RESULTS:
            return

        # Create results directory if it doesn't exist
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        # Path for individual episode results
        episode_results_file = os.path.join(
            config.RESULTS_DIR,
            f"episode_results_{config.TASK_CONFIG.DATASET.SPLIT}_r{self.local_rank}_w{self.world_size}.json"
        )

        # Load existing results or initialize empty dict
        if os.path.exists(episode_results_file):
            try:
                with open(episode_results_file, "r") as f:
                    episode_results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                episode_results = {}
        else:
            episode_results = {}

        # Add current episode result
        episode_results[episode_id] = metric

        # Save updated results
        with open(episode_results_file, "w") as f:
            json.dump(episode_results, f, indent=4)

        # Also update aggregated stats file incrementally (with global step statistics)
        self._update_aggregated_stats(episode_results, config,
                                      global_total_latency, global_total_input_tokens,
                                      global_total_output_tokens, global_total_steps)

        # Update debug.json with episode information if provided
        if debug_episode_info is not None:
            self._update_debug_json(debug_episode_info, config)

    def _update_aggregated_stats(self, episode_results, config,
                                 global_total_latency=0.0, global_total_input_tokens=0,
                                 global_total_output_tokens=0, global_total_steps=0):
        """Update aggregated statistics file with current episode results."""
        if not config.EVAL.SAVE_RESULTS:
            return

        split = config.TASK_CONFIG.DATASET.SPLIT
        aggregated_file = os.path.join(
            config.RESULTS_DIR,
            f"stats_ckpt_{split}_running.json"
        )

        # Calculate aggregated stats from all episodes so far
        num_episodes = len(episode_results)
        if num_episodes == 0:
            return

        # Keys to exclude from episode-level averaging (step-level stats handled separately)
        step_level_keys = {'avg_latency_per_step', 'avg_input_tokens_per_step', 'avg_output_tokens_per_step',
                          'total_latency', 'total_input_tokens', 'total_output_tokens'}

        episode_level_metrics = {
            episode_id: {
                key: value
                for key, value in metric.items()
                if key not in step_level_keys
            }
            for episode_id, metric in episode_results.items()
            if isinstance(metric, dict)
        }
        aggregated_stats = aggregate_numeric_metrics(episode_level_metrics)

        # Add episode count
        aggregated_stats['episodes_evaluated'] = num_episodes

        # Add global step-level statistics (average across ALL steps from ALL episodes)
        if global_total_steps > 0:
            aggregated_stats['global_avg_latency_per_step'] = global_total_latency / global_total_steps
            aggregated_stats['global_avg_input_tokens_per_step'] = global_total_input_tokens / global_total_steps
            aggregated_stats['global_avg_output_tokens_per_step'] = global_total_output_tokens / global_total_steps
            aggregated_stats['global_total_steps'] = global_total_steps

        # Save aggregated stats
        with open(aggregated_file, "w") as f:
            json.dump(aggregated_stats, f, indent=4)

    def _update_debug_json(self, debug_episode_info, config):
        """Update debug.json file with episode information incrementally."""
        if not config.EVAL.SAVE_RESULTS:
            return

        debug_file = os.path.join(
            config.RESULTS_DIR,
            "debug.json"
        )

        # Load existing debug data or initialize
        if os.path.exists(debug_file):
            try:
                with open(debug_file, "r") as f:
                    debug_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                debug_data = {
                    "experiment_name": config.EVAL.SPLIT,
                    "episodes": []
                }
        else:
            debug_data = {
                "experiment_name": config.EVAL.SPLIT,
                "episodes": []
            }

        # Check if this episode already exists (update) or is new (append)
        episode_id = debug_episode_info.get("episode_id")
        existing_episode_idx = None

        for idx, episode in enumerate(debug_data["episodes"]):
            if episode.get("episode_id") == episode_id:
                existing_episode_idx = idx
                break

        if existing_episode_idx is not None:
            # Update existing episode
            debug_data["episodes"][existing_episode_idx] = debug_episode_info
        else:
            # Add new episode
            debug_data["episodes"].append(debug_episode_info)

        # Save updated debug data
        with open(debug_file, "w") as f:
            json.dump(debug_data, f, indent=4)

    def eval(self) -> None:
        r"""Main method of trainer evaluation. 

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION',
                                                     'STEPS_TAKEN',
                                                     ]
        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWEVAL'
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()
        
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations(12)

        # sensor_uuids = []
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                # sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        
        self.config.freeze()
        torch.cuda.set_device(self.device)
        if world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            
        self.traj = self.collect_val_traj()
        self._eval_llm()
