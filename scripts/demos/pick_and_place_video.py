# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Pick and Place demo with video recording.

This script runs an automated pick and place sequence and records a video
using the gymnasium RecordVideo wrapper.

Usage:
    ./isaaclab.sh -p scripts/demos/pick_and_place_video.py --headless --enable_cameras --num_envs 1

"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and Place demo with video recording.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record.")
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
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
    SurfaceGripper,
    SurfaceGripperCfg,
)
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from isaaclab_assets.robots.pick_and_place import PICK_AND_PLACE_CFG


@configclass
class PickAndPlaceEnvCfg(DirectRLEnvCfg):
    """Configuration for the PickAndPlace robot with video recording."""

    # env
    decimation = 4
    episode_length_s = 240.0
    action_space = 4
    observation_space = 6
    state_space = 0

    # Simulation cfg. Surface grippers are currently only supported on CPU.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        device="cpu",
        render_interval=decimation,
        use_fabric=True,
        enable_scene_query_support=True,
    )
    debug_vis = True

    # robot
    robot_cfg: ArticulationCfg = PICK_AND_PLACE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    x_dof_name = "x_axis"
    y_dof_name = "y_axis"
    z_dof_name = "z_axis"

    # Cube to pick-up
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    # Surface Gripper
    gripper = SurfaceGripperCfg(
        prim_path="/World/envs/env_.*/Robot/picker_head/SurfaceGripper",
        max_grip_distance=0.1,
        shear_force_limit=500.0,
        coaxial_force_limit=500.0,
        retry_interval=0.2,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=12.0, replicate_physics=True)

    # Fixed positions for demo (not random)
    initial_x_pos = 0.0
    initial_y_pos = 0.0
    initial_z_pos = 0.5

    initial_object_x_pos = 1.0
    initial_object_y_pos = -1.0
    initial_object_z_pos = 0.2

    target_x_pos = -1.0
    target_y_pos = 1.0
    target_z_pos = 0.2


class PickAndPlaceEnv(DirectRLEnv):
    """Pick and Place environment with automated sequence for video recording."""

    cfg: PickAndPlaceEnvCfg

    def __init__(self, cfg: PickAndPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Indices used to control the different axes of the gantry
        self._x_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.x_dof_name)
        self._y_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.y_dof_name)
        self._z_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.z_dof_name)

        # joints info
        self.joint_pos = self.pick_and_place.data.joint_pos
        self.joint_vel = self.pick_and_place.data.joint_vel

        # Buffers for control
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_pos[:, 0] = self.cfg.target_x_pos
        self.target_pos[:, 1] = self.cfg.target_y_pos
        self.target_pos[:, 2] = self.cfg.target_z_pos

        self.instant_controls = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.permanent_controls = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

        # State machine for automated demo
        self.demo_state = 0
        self.state_counter = 0

        # Visual marker for the target
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.pick_and_place = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.cube_cfg)
        self.gripper = SurfaceGripper(self.cfg.gripper)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["pick_and_place"] = self.pick_and_place
        self.scene.rigid_objects["cube"] = self.cube
        self.scene.surface_grippers["gripper"] = self.gripper
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply automated pick and place sequence."""
        self.state_counter += 1

        head_pos_x = self.pick_and_place.data.joint_pos[:, self._x_dof_idx[0]]
        head_pos_y = self.pick_and_place.data.joint_pos[:, self._y_dof_idx[0]]

        cube_pos = self.cube.data.root_pos_w - self.scene.env_origins
        cube_pos_x = cube_pos[:, 0]
        cube_pos_y = cube_pos[:, 1]

        # Proportional gain for XY movement
        kp = 50.0

        if self.demo_state == 0:
            # Move to cube XY position
            d_cube_robot_x = cube_pos_x - head_pos_x
            d_cube_robot_y = cube_pos_y - head_pos_y
            self.instant_controls[:, 0] = d_cube_robot_x * kp
            self.instant_controls[:, 1] = d_cube_robot_y * kp
            self.instant_controls[:, 2] = 0
            self.permanent_controls[:, 0] = 0

            # Check if aligned (with tolerance)
            if torch.abs(d_cube_robot_x).max() < 0.15 and torch.abs(d_cube_robot_y).max() < 0.15:
                self.demo_state = 1
                self.state_counter = 0
                print(f"[DEBUG] State 0->1: Aligned with cube")

        elif self.demo_state == 1:
            # Lower to cube
            self.instant_controls[:, 0] = 0
            self.instant_controls[:, 1] = 0
            self.instant_controls[:, 2] = 0
            self.permanent_controls[:, 0] = 150.0  # Move down

            if self.state_counter > 45:
                self.demo_state = 2
                self.state_counter = 0
                print(f"[DEBUG] State 1->2: Lowered, now gripping")

        elif self.demo_state == 2:
            # Grip
            self.instant_controls[:, 0] = 0
            self.instant_controls[:, 1] = 0
            self.instant_controls[:, 2] = 1  # Close gripper
            self.permanent_controls[:, 0] = 0

            if self.state_counter > 20:
                self.demo_state = 3
                self.state_counter = 0
                print(f"[DEBUG] State 2->3: Gripped, now raising")

        elif self.demo_state == 3:
            # Raise with cube
            self.instant_controls[:, 0] = 0
            self.instant_controls[:, 1] = 0
            self.instant_controls[:, 2] = 1  # Keep gripper closed
            self.permanent_controls[:, 0] = -250.0  # Move up

            if self.state_counter > 45:
                self.demo_state = 4
                self.state_counter = 0
                print(f"[DEBUG] State 3->4: Raised, moving to target")

        elif self.demo_state == 4:
            # Move to target XY position
            target_pos_x = self.target_pos[:, 0]
            target_pos_y = self.target_pos[:, 1]
            d_target_robot_x = target_pos_x - head_pos_x
            d_target_robot_y = target_pos_y - head_pos_y
            self.instant_controls[:, 0] = d_target_robot_x * kp
            self.instant_controls[:, 1] = d_target_robot_y * kp
            self.instant_controls[:, 2] = 1  # Keep gripper closed
            self.permanent_controls[:, 0] = 0

            if torch.abs(d_target_robot_x).max() < 0.15 and torch.abs(d_target_robot_y).max() < 0.15:
                self.demo_state = 5
                self.state_counter = 0
                print(f"[DEBUG] State 4->5: At target, lowering")

        elif self.demo_state == 5:
            # Lower to target
            self.instant_controls[:, 0] = 0
            self.instant_controls[:, 1] = 0
            self.instant_controls[:, 2] = 1  # Keep gripper closed
            self.permanent_controls[:, 0] = 150.0  # Move down

            if self.state_counter > 45:
                self.demo_state = 6
                self.state_counter = 0
                print(f"[DEBUG] State 5->6: Lowered, releasing")

        elif self.demo_state == 6:
            # Release
            self.instant_controls[:, 0] = 0
            self.instant_controls[:, 1] = 0
            self.instant_controls[:, 2] = -1  # Open gripper
            self.permanent_controls[:, 0] = 0

            if self.state_counter > 20:
                self.demo_state = 7
                self.state_counter = 0
                print(f"[DEBUG] State 6->7: Released, rising")

        elif self.demo_state == 7:
            # Rise and done
            self.instant_controls[:, 0] = 0
            self.instant_controls[:, 1] = 0
            self.instant_controls[:, 2] = 0
            self.permanent_controls[:, 0] = -250.0  # Move up

        # Apply joint effort targets
        self.pick_and_place.set_joint_effort_target(
            self.instant_controls[:, 0].unsqueeze(dim=1), joint_ids=self._x_dof_idx
        )
        self.pick_and_place.set_joint_effort_target(
            self.instant_controls[:, 1].unsqueeze(dim=1), joint_ids=self._y_dof_idx
        )
        self.pick_and_place.set_joint_effort_target(
            self.permanent_controls[:, 0].unsqueeze(dim=1), joint_ids=self._z_dof_idx
        )
        # Set the gripper command
        self.gripper.set_grippers_command(self.instant_controls[:, 2])

    def _get_observations(self) -> dict:
        gripper_state = self.gripper.state.clone()
        obs = torch.cat(
            (
                self.joint_pos[:, self._x_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._x_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._y_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._y_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._z_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._z_dof_idx[0]].unsqueeze(dim=1),
                self.target_pos[:, 0].unsqueeze(dim=1),
                self.target_pos[:, 1].unsqueeze(dim=1),
                gripper_state.unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros_like(self.reset_terminated, dtype=torch.float32)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.pick_and_place.data.joint_pos
        self.joint_vel = self.pick_and_place.data.joint_vel
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.pick_and_place._ALL_INDICES
        super()._reset_idx(env_ids)

        self.target_pos[env_ids, 0] = self.cfg.target_x_pos
        self.target_pos[env_ids, 1] = self.cfg.target_y_pos
        self.target_pos[env_ids, 2] = self.cfg.target_z_pos

        cube_pos = self.cube.data.default_root_state[env_ids, :7]
        cube_pos[:, 0] = self.cfg.initial_object_x_pos
        cube_pos[:, 1] = self.cfg.initial_object_y_pos
        cube_pos[:, 2] = self.cfg.initial_object_z_pos
        cube_pos[:, :3] += self.scene.env_origins[env_ids]
        self.cube.write_root_pose_to_sim(cube_pos, env_ids)

        joint_pos = self.pick_and_place.data.default_joint_pos[env_ids]
        joint_pos[:, self._x_dof_idx] = self.cfg.initial_x_pos
        joint_pos[:, self._y_dof_idx] = self.cfg.initial_y_pos
        joint_pos[:, self._z_dof_idx] = self.cfg.initial_z_pos
        joint_vel = self.pick_and_place.data.default_joint_vel[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.pick_and_place.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.demo_state = 0
        self.state_counter = 0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = 0.25
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "scene") and hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(self.target_pos + self.scene.env_origins)


def main():
    """Main function."""
    # Create environment configuration
    env_cfg = PickAndPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Create environment with rgb_array render mode for video recording
    env = PickAndPlaceEnv(env_cfg, render_mode="rgb_array")

    # Set camera view
    env.sim.set_camera_view(eye=[8.0, 8.0, 6.0], target=[0.0, 0.0, 1.0])

    # Setup output directory
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "pick_and_place_video")
    os.makedirs(output_dir, exist_ok=True)

    # Wrap with RecordVideo
    env = gym.wrappers.RecordVideo(
        env,
        output_dir,
        episode_trigger=lambda x: True,  # Record all episodes
        video_length=args_cli.video_length,
        disable_logger=True,
    )

    print(f"[INFO]: Recording video to {output_dir}")
    print(f"[INFO]: Running for {args_cli.video_length} steps")

    # Reset and run
    env.reset()

    with torch.inference_mode():
        for step in range(args_cli.video_length):
            actions = torch.zeros((args_cli.num_envs, 4), device=env.unwrapped.device, dtype=torch.float32)
            env.step(actions)

            if step % 100 == 0:
                print(f"[INFO]: Step {step}/{args_cli.video_length}, Demo state: {env.unwrapped.demo_state}")

    print("[INFO]: Recording complete")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
