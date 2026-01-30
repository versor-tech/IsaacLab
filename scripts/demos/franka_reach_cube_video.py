# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Franka arm reaching to a cube with video recording.

This script spawns a Franka Panda robot arm and a cube in front of it,
then uses differential IK to move the end-effector to the cube position
while recording a video.

Usage:
    ./isaaclab.sh -p scripts/demos/franka_reach_cube_video.py --headless --enable_cameras --num_envs 1

"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Franka arm reaching to cube with video recording.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video_length", type=int, default=300, help="Number of steps to record.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the video.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class FrankaReachCubeEnvCfg(DirectRLEnvCfg):
    """Configuration for the Franka reach cube environment."""

    # env
    decimation = 2
    episode_length_s = 30.0
    action_space = 7  # joint positions
    observation_space = 7
    state_space = 0

    # simulation - use CPU device to avoid GPU PhysX pipeline issues
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        device="cpu",
    )
    debug_vis = True

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    # robot
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # cube to reach
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
    )

    # target position for the cube (in front of the robot)
    cube_position = (0.5, 0.0, 0.05)


class FrankaReachCubeEnv(DirectRLEnv):
    """Franka arm reaching to cube environment with video recording."""

    cfg: FrankaReachCubeEnvCfg

    def __init__(self, cfg: FrankaReachCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Setup robot entity configuration for IK
        self.robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        self.robot_entity_cfg.resolve(self.scene)

        # Get jacobian index for end-effector
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # Create differential IK controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self.diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.num_envs, device=self.device
        )

        # IK command buffer (position + quaternion)
        self.ik_commands = torch.zeros(
            self.num_envs, self.diff_ik_controller.action_dim, device=self.device
        )

        # Target joint positions
        self.joint_pos_des = self.robot.data.default_joint_pos[:, self.robot_entity_cfg.joint_ids].clone()

        # Define sequence of goals: approach cube from above, then reach to cube
        self.ee_goals = torch.tensor(
            [
                # Start position (above cube)
                [0.5, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0],
                # Move closer to cube
                [0.5, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0],
                # Reach to cube (just above it)
                [0.5, 0.0, 0.12, 0.0, 1.0, 0.0, 0.0],
            ],
            device=self.device,
        )
        self.current_goal_idx = 0
        self.goal_switch_counter = 0
        self.steps_per_goal = 100

        # Set initial IK command
        self.ik_commands[:] = self.ee_goals[self.current_goal_idx]
        self.diff_ik_controller.set_command(self.ik_commands)

        # Visual markers
        self._setup_markers()

    def _setup_scene(self):
        """Setup the scene with robot, cube, table, and lights."""
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

    def _setup_markers(self):
        """Setup visualization markers."""
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_current")
        )
        self.goal_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_goal")
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Compute and apply IK to reach the cube."""
        # Update goal if needed
        self.goal_switch_counter += 1
        if self.goal_switch_counter >= self.steps_per_goal:
            self.goal_switch_counter = 0
            self.current_goal_idx = (self.current_goal_idx + 1) % len(self.ee_goals)
            self.ik_commands[:] = self.ee_goals[self.current_goal_idx]
            self.diff_ik_controller.set_command(self.ik_commands)
            print(f"[INFO]: Switching to goal {self.current_goal_idx}: {self.ee_goals[self.current_goal_idx][:3].tolist()}")

        # Get robot state
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
        ]
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

        # Compute end-effector pose in robot base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

        # Compute joint positions via IK
        self.joint_pos_des = self.diff_ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )

        # Apply joint position targets
        self.robot.set_joint_position_target(
            self.joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
        )

        # Update visualization markers
        ee_pose_w_full = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        self.ee_marker.visualize(ee_pose_w_full[:, 0:3], ee_pose_w_full[:, 3:7])
        self.goal_marker.visualize(
            self.ik_commands[:, 0:3] + self.scene.env_origins, self.ik_commands[:, 3:7]
        )

    def _get_observations(self) -> dict:
        """Get observations."""
        obs = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards (not used for this demo)."""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if episode is done."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids):
        """Reset environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot to default state
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset cube position
        cube_pos = self.cube.data.default_root_state[env_ids, :7].clone()
        cube_pos[:, :3] += self.scene.env_origins[env_ids]
        self.cube.write_root_pose_to_sim(cube_pos, env_ids)

        # Reset IK controller and goal
        self.diff_ik_controller.reset()
        self.current_goal_idx = 0
        self.goal_switch_counter = 0
        self.ik_commands[:] = self.ee_goals[self.current_goal_idx]
        self.diff_ik_controller.set_command(self.ik_commands)
        self.joint_pos_des = joint_pos[:, self.robot_entity_cfg.joint_ids].clone()


def main():
    """Main function."""
    # Create environment configuration
    print("Creating environment configuration...")
    env_cfg = FrankaReachCubeEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Create environment with rgb_array render mode for video recording
    print("Creating environment...")
    env = FrankaReachCubeEnv(env_cfg, render_mode="rgb_array")

    # Set camera view for a good angle of the robot and cube
    print("Setting camera view...")
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.3, 0.0, 0.3])

    # Setup output directory
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "output", "franka_reach_cube_video"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Wrap with RecordVideo
    print("Wrapping environment with RecordVideo...")
    env = gym.wrappers.RecordVideo(
        env,
        output_dir,
        episode_trigger=lambda x: True,  # Record all episodes
        video_length=args_cli.video_length,
        disable_logger=True,
    )

    print(f"[INFO]: Recording video to {output_dir}")
    print(f"[INFO]: Running for {args_cli.video_length} steps")
    print("[INFO]: Franka arm will move through waypoints to reach the red cube")

    # Reset and run
    print("Resetting environment...")
    env.reset()

    with torch.inference_mode():
        for step in range(args_cli.video_length):
            # Actions are not used (IK computes internally)
            actions = torch.zeros(
                (args_cli.num_envs, env_cfg.action_space), device=env.unwrapped.device, dtype=torch.float32
            )
            env.step(actions)

            if step % 50 == 0:
                print(f"[INFO]: Step {step}/{args_cli.video_length}")

    print("[INFO]: Recording complete")
    print(f"[INFO]: Video saved to {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
