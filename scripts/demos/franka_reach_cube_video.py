# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Franka arm RL training for grasping and lifting a cube.

This script trains a PPO agent to grasp and lift a cube using
the Franka Panda robot arm.

Usage:
    ./isaaclab.sh -p scripts/demos/franka_reach_cube_video.py --headless --num_envs 64 --max_iterations 500

"""

from __future__ import annotations

import argparse
import os
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Franka arm RL training for grasping and lifting cube.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations.")
parser.add_argument("--video", action="store_true", default=False, help="Record video during training.")
parser.add_argument("--video_length", type=int, default=300, help="Number of steps to record per video.")
parser.add_argument("--video_interval", type=int, default=100, help="Training iterations between videos.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
from datetime import datetime

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


# ==============================================================================
# Environment Configuration
# ==============================================================================

@configclass
class FrankaGraspCubeEnvCfg(DirectRLEnvCfg):
    """Configuration for the Franka grasp cube RL environment."""

    # env
    decimation = 2
    episode_length_s = 10.0  # Shorter episodes for faster learning
    action_space = 8  # 7 arm joints + 1 gripper
    observation_space = 26  # Comprehensive state info
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        device="cuda:0",  # Use GPU for faster training
    )
    debug_vis = False

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=True)

    # robot
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # cube to grasp - NOT kinematic so it can be grasped
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
    )

    # target height for lifting
    target_lift_height = 0.25  # 25cm above table
    initial_cube_height = 0.05

    # Reward scales
    dist_reward_scale = 2.0
    grasp_reward_scale = 5.0
    lift_reward_scale = 10.0
    height_bonus_scale = 20.0
    action_penalty_scale = 0.01
    drop_penalty_scale = 5.0

    # Action scaling
    action_scale = 0.1  # Scale for joint position deltas


# ==============================================================================
# PPO Neural Network
# ==============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # Actor (policy) head
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Actions bounded to [-1, 1]
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        return features

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.forward(obs)
        action_mean = self.actor_mean(features)

        if deterministic:
            return action_mean, None, None

        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, dist.entropy().sum(dim=-1)

    def get_value(self, obs: torch.Tensor):
        features = self.forward(obs)
        return self.critic(features).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self.forward(obs)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)

        return value, log_prob, entropy


# ==============================================================================
# PPO Algorithm
# ==============================================================================

class PPO:
    """Proximal Policy Optimization algorithm."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 10,
        mini_batch_size: int = 64,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size

        self.actor_critic = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            return self.actor_critic.get_action(obs, deterministic)

    def get_value(self, obs: torch.Tensor):
        with torch.no_grad():
            return self.actor_critic.get_value(obs)

    def update(self, rollout_buffer):
        """Update policy using collected rollouts."""
        obs = rollout_buffer["obs"]
        actions = rollout_buffer["actions"]
        old_log_probs = rollout_buffer["log_probs"]
        returns = rollout_buffer["returns"]
        advantages = rollout_buffer["advantages"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        # Flatten batch
        batch_size = obs.shape[0] * obs.shape[1]
        obs_flat = obs.reshape(batch_size, -1)
        actions_flat = actions.reshape(batch_size, -1)
        old_log_probs_flat = old_log_probs.reshape(batch_size)
        returns_flat = returns.reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)

        for _ in range(self.num_epochs):
            # Generate random mini-batch indices
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > batch_size:
                    continue

                mb_indices = indices[start:end]

                mb_obs = obs_flat[mb_indices]
                mb_actions = actions_flat[mb_indices]
                mb_old_log_probs = old_log_probs_flat[mb_indices]
                mb_returns = returns_flat[mb_indices]
                mb_advantages = advantages_flat[mb_indices]

                # Evaluate actions
                values, log_probs, entropy = self.actor_critic.evaluate_actions(mb_obs, mb_actions)

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "loss": total_loss / max(num_updates, 1),
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }


# ==============================================================================
# RL Environment
# ==============================================================================

class FrankaGraspCubeEnv(DirectRLEnv):
    """Franka arm RL environment for grasping and lifting a cube."""

    cfg: FrankaGraspCubeEnvCfg

    def __init__(self, cfg: FrankaGraspCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Setup robot entity configuration
        self.robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        self.robot_entity_cfg.resolve(self.scene)

        # Get finger link indices for grasp detection
        self.finger_entity_cfg = SceneEntityCfg(
            "robot", body_names=["panda_leftfinger", "panda_rightfinger"]
        )
        self.finger_entity_cfg.resolve(self.scene)

        # Get jacobian index for end-effector
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # Arm and gripper joint indices
        self.arm_joint_ids, _ = self.robot.find_joints("panda_joint.*")
        self.finger_joint_ids, _ = self.robot.find_joints("panda_finger_joint.*")

        # Episode tracking
        self.cube_was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_cube_height = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Success tracking
        self.success_count = 0
        self.episode_count = 0

    def _setup_scene(self):
        """Setup the scene with robot, cube, and lights."""
        # Add robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Add cube
        self.cube = RigidObject(self.cfg.cube_cfg)
        self.scene.rigid_objects["cube"] = self.cube

        # Add ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # Add light
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions before physics step."""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        # Split actions: arm joints (7) + gripper (1)
        arm_actions = self.actions[:, :7]
        gripper_actions = self.actions[:, 7:8]

        # Get current joint positions
        current_arm_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        current_finger_pos = self.robot.data.joint_pos[:, self.finger_joint_ids]

        # Compute target positions (current + scaled delta)
        target_arm_pos = current_arm_pos + arm_actions * self.cfg.action_scale

        # Clamp arm positions to joint limits
        arm_lower = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, 0]
        arm_upper = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, 1]
        target_arm_pos = torch.clamp(target_arm_pos, arm_lower, arm_upper)

        # Gripper: action in [-1, 1] maps to finger position [0, 0.04]
        # -1 = closed (0), 1 = open (0.04)
        target_finger_pos = (gripper_actions + 1) * 0.02  # [0, 0.04]
        target_finger_pos = target_finger_pos.expand(-1, 2)  # Both fingers

        # Set joint targets
        self.robot.set_joint_position_target(target_arm_pos, joint_ids=self.arm_joint_ids)
        self.robot.set_joint_position_target(target_finger_pos, joint_ids=self.finger_joint_ids)

    def _get_observations(self) -> dict:
        """Get observations for the policy."""
        # Joint positions (7)
        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]

        # Joint velocities (7)
        joint_vel = self.robot.data.joint_vel[:, self.arm_joint_ids]

        # Gripper state (2)
        gripper_pos = self.robot.data.joint_pos[:, self.finger_joint_ids]

        # End-effector position in world frame (3)
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        ee_pos_w = ee_pose_w[:, 0:3]

        # Cube position relative to EE (3)
        cube_pos_w = self.cube.data.root_pos_w - self.scene.env_origins
        relative_cube_pos = cube_pos_w - ee_pos_w

        # Gripper to cube distance (1)
        gripper_cube_dist = torch.norm(relative_cube_pos, dim=-1, keepdim=True)

        # Cube height (1)
        cube_height = cube_pos_w[:, 2:3]

        # Grasp detection
        is_grasping = self._check_grasp()

        # Lifted flag (1)
        is_lifted = (cube_height[:, 0] > self.cfg.target_lift_height).float().unsqueeze(-1)

        # Concatenate observations
        obs = torch.cat([
            joint_pos,              # 7
            joint_vel * 0.1,        # 7 (scaled)
            gripper_pos,            # 2
            ee_pos_w,               # 3
            relative_cube_pos,      # 3
            gripper_cube_dist,      # 1
            cube_height,            # 1
            is_grasping.unsqueeze(-1).float(),  # 1
            is_lifted,              # 1
        ], dim=-1)

        return {"policy": obs}

    def _check_grasp(self) -> torch.Tensor:
        """Check if the cube is grasped between the fingers."""
        # Get finger positions
        left_finger_pos = self.robot.data.body_pos_w[:, self.finger_entity_cfg.body_ids[0]]
        right_finger_pos = self.robot.data.body_pos_w[:, self.finger_entity_cfg.body_ids[1]]
        finger_center = (left_finger_pos + right_finger_pos) / 2

        # Get cube position
        cube_pos_w = self.cube.data.root_pos_w

        # Check distance from finger center to cube
        dist_to_cube = torch.norm(finger_center - cube_pos_w, dim=-1)

        # Check finger separation (gripper closed enough)
        finger_separation = torch.norm(left_finger_pos - right_finger_pos, dim=-1)

        # Grasp conditions: cube close to fingers AND fingers closed
        is_grasping = (dist_to_cube < 0.08) & (finger_separation < 0.06)

        return is_grasping

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for grasping and lifting."""
        # Get positions
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        ee_pos_w = ee_pose_w[:, 0:3]
        cube_pos_w = self.cube.data.root_pos_w - self.scene.env_origins

        # Distance from gripper to cube
        gripper_cube_dist = torch.norm(ee_pos_w - cube_pos_w, dim=-1)

        # Check grasp state
        is_grasping = self._check_grasp()
        self.cube_was_grasped = self.cube_was_grasped | is_grasping

        # Cube height
        cube_height = cube_pos_w[:, 2]
        self.max_cube_height = torch.maximum(self.max_cube_height, cube_height)

        # === Reward Components ===

        # 1. Distance reward (encourage approaching cube)
        dist_reward = 1.0 / (1.0 + gripper_cube_dist ** 2)
        dist_reward = dist_reward * self.cfg.dist_reward_scale

        # 2. Grasp reward (bonus for grasping)
        grasp_reward = is_grasping.float() * self.cfg.grasp_reward_scale

        # 3. Lift reward (height of cube when grasped)
        lift_reward = torch.where(
            is_grasping,
            cube_height * self.cfg.lift_reward_scale,
            torch.zeros_like(cube_height),
        )

        # 4. Height bonus (large bonus when lifted above target)
        height_bonus = torch.where(
            cube_height > self.cfg.target_lift_height,
            torch.ones_like(cube_height) * self.cfg.height_bonus_scale,
            torch.zeros_like(cube_height),
        )

        # 5. Action penalty (smooth actions)
        action_penalty = torch.sum(self.actions ** 2, dim=-1) * self.cfg.action_penalty_scale

        # 6. Drop penalty (if cube was grasped but now dropped)
        dropped = self.cube_was_grasped & (~is_grasping) & (cube_height < 0.1)
        drop_penalty = dropped.float() * self.cfg.drop_penalty_scale

        # Total reward
        reward = dist_reward + grasp_reward + lift_reward + height_bonus - action_penalty - drop_penalty

        # Log stats
        self.extras["log"] = {
            "dist_reward": dist_reward.mean(),
            "grasp_reward": grasp_reward.mean(),
            "lift_reward": lift_reward.mean(),
            "height_bonus": height_bonus.mean(),
            "action_penalty": action_penalty.mean(),
            "drop_penalty": drop_penalty.mean(),
            "is_grasping": is_grasping.float().mean(),
            "cube_height": cube_height.mean(),
            "max_cube_height": self.max_cube_height.mean(),
        }

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if episode is done."""
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Success: cube lifted above target height
        cube_pos_w = self.cube.data.root_pos_w - self.scene.env_origins
        cube_height = cube_pos_w[:, 2]
        success = cube_height > self.cfg.target_lift_height

        # Track success rate
        self.success_count += success.sum().item()
        self.episode_count += time_out.sum().item() + success.sum().item()

        return success, time_out

    def _reset_idx(self, env_ids):
        """Reset environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot to default state
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Randomize cube position within reach
        num_resets = len(env_ids)
        cube_pos = self.cube.data.default_root_state[env_ids, :7].clone()

        # Random offset within robot workspace
        x_offset = torch.rand(num_resets, device=self.device) * 0.2 - 0.1  # [-0.1, 0.1]
        y_offset = torch.rand(num_resets, device=self.device) * 0.2 - 0.1
        cube_pos[:, 0] += x_offset
        cube_pos[:, 1] += y_offset
        cube_pos[:, :3] += self.scene.env_origins[env_ids]

        self.cube.write_root_pose_to_sim(cube_pos, env_ids)

        # Reset cube velocity
        cube_vel = torch.zeros(num_resets, 6, device=self.device)
        self.cube.write_root_velocity_to_sim(cube_vel, env_ids)

        # Reset tracking
        self.cube_was_grasped[env_ids] = False
        self.max_cube_height[env_ids] = self.cfg.initial_cube_height


# ==============================================================================
# Rollout Buffer
# ==============================================================================

class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, num_envs: int, num_steps: int, obs_dim: int, action_dim: int, device: torch.device):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device

        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)

        self.step = 0

    def add(self, obs, actions, log_probs, rewards, dones, values):
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.step += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        returns = torch.zeros_like(self.rewards)
        advantages = torch.zeros_like(self.rewards)

        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_value
            else:
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values

        return {
            "obs": self.obs,
            "actions": self.actions,
            "log_probs": self.log_probs,
            "returns": returns,
            "advantages": advantages,
        }

    def reset(self):
        self.step = 0


# ==============================================================================
# Main Training Loop
# ==============================================================================

def main():
    """Main training function."""
    # Set random seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # Create environment configuration
    print("[INFO]: Creating environment configuration...")
    env_cfg = FrankaGraspCubeEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Create environment
    print("[INFO]: Creating environment...")
    render_mode = "rgb_array" if args_cli.video else None
    env = FrankaGraspCubeEnv(env_cfg, render_mode=render_mode)

    # Set camera view
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.3, 0.0, 0.3])

    # Setup output directory
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "output", f"franka_grasp_rl_{timestamp}"
        )
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO]: Output directory: {output_dir}")

    # Wrap with RecordVideo if needed
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            os.path.join(output_dir, "videos"),
            episode_trigger=lambda x: x % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    # Create PPO agent
    print("[INFO]: Creating PPO agent...")
    device = env.unwrapped.device
    ppo = PPO(
        obs_dim=env_cfg.observation_space,
        action_dim=env_cfg.action_space,
        device=device,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        num_epochs=10,
        mini_batch_size=256,
    )

    # Create rollout buffer
    num_steps_per_update = 64
    buffer = RolloutBuffer(
        num_envs=args_cli.num_envs,
        num_steps=num_steps_per_update,
        obs_dim=env_cfg.observation_space,
        action_dim=env_cfg.action_space,
        device=device,
    )

    # Training loop
    print(f"[INFO]: Starting training for {args_cli.max_iterations} iterations...")
    print(f"[INFO]: {args_cli.num_envs} parallel environments")

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    total_steps = 0
    reward_history = deque(maxlen=100)
    best_reward = float("-inf")

    for iteration in range(args_cli.max_iterations):
        # Collect rollouts
        buffer.reset()

        for step in range(num_steps_per_update):
            # Get action from policy
            actions, log_probs, _ = ppo.get_action(obs)
            values = ppo.get_value(obs)

            # Step environment
            next_obs_dict, rewards, terminated, truncated, info = env.step(actions)
            next_obs = next_obs_dict["policy"]
            dones = terminated | truncated

            # Store in buffer
            buffer.add(obs, actions, log_probs, rewards, dones.float(), values)

            obs = next_obs
            total_steps += args_cli.num_envs

            # Track rewards
            reward_history.append(rewards.mean().item())

        # Compute returns and advantages
        last_value = ppo.get_value(obs)
        rollout_data = buffer.compute_returns_and_advantages(last_value, ppo.gamma, ppo.gae_lambda)

        # Update policy
        update_stats = ppo.update(rollout_data)

        # Logging
        mean_reward = np.mean(list(reward_history)) if reward_history else 0
        if mean_reward > best_reward:
            best_reward = mean_reward
            # Save best model
            torch.save(ppo.actor_critic.state_dict(), os.path.join(output_dir, "best_model.pt"))

        if iteration % 10 == 0:
            unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
            success_rate = (unwrapped_env.success_count / max(unwrapped_env.episode_count, 1)) * 100

            print(
                f"[Iter {iteration:4d}] "
                f"Steps: {total_steps:8d} | "
                f"Reward: {mean_reward:7.3f} | "
                f"Best: {best_reward:7.3f} | "
                f"Success: {success_rate:5.1f}% | "
                f"Loss: {update_stats['loss']:.4f}"
            )

        # Save checkpoint periodically
        if (iteration + 1) % 100 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{iteration + 1}.pt")
            torch.save({
                "iteration": iteration,
                "model_state_dict": ppo.actor_critic.state_dict(),
                "optimizer_state_dict": ppo.optimizer.state_dict(),
                "best_reward": best_reward,
            }, checkpoint_path)
            print(f"[INFO]: Saved checkpoint to {checkpoint_path}")

    # Save final model
    print("[INFO]: Training complete!")
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save(ppo.actor_critic.state_dict(), final_path)
    print(f"[INFO]: Saved final model to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
