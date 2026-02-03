# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Franka Pick and Place Cube Demo with Video Recording.

This script demonstrates a Franka Panda robot performing a pick-and-place task,
picking up a cube from a table and placing it in a target location (bowl).
The motion is driven by keyframe interpolation similar to the original Genesis/Viser implementation.

Adapted from: boggart-sim/examples/pick_cube_viser_franka.py

Usage:
    ./isaaclab.sh -p scripts/demos/franka_pick_cube_video.py --headless --enable_cameras --num_envs 1

"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Franka pick and place cube with video recording.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video_length", type=int, default=800, help="Number of steps to record.")
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
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

import math

# Paths to mesh files from boggart-sim
BOGGART_SIM_DATA_DIR = "/home/debian/boggart-sim/data"
COFFEE_TABLE_OBJ_PATH = os.path.join(BOGGART_SIM_DATA_DIR, "coffee_table", "coffee_table.obj")
BOWL_OBJ_PATH = os.path.join(BOGGART_SIM_DATA_DIR, "bowl", "bowl.obj")

# Output directory for converted USD files
USD_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "converted_meshes")


def convert_mesh_to_usd(obj_path: str, usd_name: str, scale: tuple = (1.0, 1.0, 1.0),
                         rotation: tuple = (1.0, 0.0, 0.0, 0.0)) -> str:
    """Convert an OBJ mesh file to USD format.

    Args:
        obj_path: Path to the OBJ file
        usd_name: Name for the output USD file (without extension)
        scale: Scale factor (x, y, z)
        rotation: Rotation quaternion (w, x, y, z)

    Returns:
        Path to the converted USD file
    """
    os.makedirs(USD_OUTPUT_DIR, exist_ok=True)

    mesh_converter_cfg = MeshConverterCfg(
        asset_path=obj_path,
        usd_dir=USD_OUTPUT_DIR,
        usd_file_name=f"{usd_name}.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        scale=scale,
        rotation=rotation,
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=schemas_cfg.MassPropertiesCfg(mass=100.0),
        mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
    )

    mesh_converter = MeshConverter(mesh_converter_cfg)
    print(f"[INFO]: Converted {obj_path} to {mesh_converter.usd_path}")
    return mesh_converter.usd_path


def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles (in radians) to quaternion [w, x, y, z].

    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)

    Returns:
        Tuple of (w, x, y, z) quaternion components
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)


# Convert mesh files to USD (done after simulation app is launched)
print("[INFO]: Converting mesh files to USD...")
COFFEE_TABLE_USD_PATH = convert_mesh_to_usd(
    COFFEE_TABLE_OBJ_PATH,
    "coffee_table",
    scale=(0.012, 0.012, 0.012),  # Scale from boggart-sim
    rotation=euler_to_quat(math.pi / 2, 0, 0),  # 90 degrees around X-axis
)
BOWL_USD_PATH = convert_mesh_to_usd(
    BOWL_OBJ_PATH,
    "bowl",
    scale=(0.3, 0.3, 0.3),  # Scale from boggart-sim
    rotation=euler_to_quat(math.pi / 2, 0, 0),  # 90 degrees around X-axis
)


@configclass
class FrankaPickCubeEnvCfg(DirectRLEnvCfg):
    """Configuration for the Franka pick cube environment."""

    # env
    decimation = 2
    episode_length_s = 60.0
    action_space = 9  # 7 arm joints + 2 gripper fingers
    observation_space = 9
    state_space = 0

    # simulation - use CPU device to avoid GPU PhysX pipeline issues
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        device="cpu",
    )
    debug_vis = False

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    # robot - using high PD gains for stable position control
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.pos = (0.0, 0.35, 0.0)  # Position robot behind the table
    # Rotate robot 90 degrees clockwise about z-axis to face the table
    # 90 degrees clockwise = -90 degrees = -pi/2 radians
    robot_cfg.init_state.rot = euler_to_quat(0.0, 0.0, -math.pi / 2)

    # table configuration - using coffee table mesh from boggart-sim
    # The mesh is pre-scaled and rotated during USD conversion
    table_surface_z = 0.5  # Approximate table surface height
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=COFFEE_TABLE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.1, 0.05)),  # table position
    )

    # cube to pick up (red cube)
    cube_size = 0.05
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(cube_size, cube_size, cube_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),  # red
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.15, table_surface_z + cube_size / 2)),  # on table
    )

    # target bowl - using bowl mesh from boggart-sim
    # The mesh is pre-scaled during USD conversion
    bowl_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOWL_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.1, table_surface_z + 0.02)),  # on table, to the side
    )


class FrankaPickCubeEnv(DirectRLEnv):
    """Franka pick and place environment with keyframe-based motion."""

    cfg: FrankaPickCubeEnvCfg

    def __init__(self, cfg: FrankaPickCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices for the robot
        # Franka Panda: panda_joint1-7 (arm) + panda_finger_joint1-2 (gripper) = 9 DOFs
        self.arm_joint_ids, _ = self.robot.find_joints("panda_joint.*")
        self.finger_joint_ids, _ = self.robot.find_joints("panda_finger_joint.*")
        self.num_robot_joints = len(self.arm_joint_ids) + len(self.finger_joint_ids)

        # Define motion keyframes (7 arm joints + 2 finger joints)
        # Finger values: 0.04 = open, 0.0 = closed
        keyframes_list = [
            # Home pose
            [0, -0.785, 0, -2.356, 0, 2.5, 0.785, 0.04, 0.04],
            # Pre-grasp pose (above cube)
            [0.0, -0.25, 0.0, -2.0, 0.0, 1.9, 0.785, 0.04, 0.04],
            # Approach pose (closer to cube)
            [0.0, -0.15, 0.0, -2.0, 0.0, 1.9, 0.785, 0.04, 0.04],
            # Grasp pose (close gripper)
            [0.0, -0.15, 0.0, -2.0, 0.0, 1.9, 0.785, 0.0, 0.0],
            # Lift pose
            [0, -0.3, 0, -1.57, 0, 1.571, 0.785, 0.0, 0.0],
            # Move to bowl position
            [0.7, -0.2, 0, -1.6, 0, 1.8, 0.785, 0.0, 0.0],
            # Pre-place pose
            [0.7, 0.0, 0, -1.8, 0, 2.0, 0.785, 0.0, 0.0],
            # Place pose (open gripper)
            [0.7, 0.2, 0, -1.8, 0, 2.0, 0.785, 0.04, 0.04],
            # Lift away
            [0.7, -0.3, 0, -1.6, 0, 1.8, 0.785, 0.04, 0.04],
        ]
        self.keyframes = torch.tensor(keyframes_list, device=self.device, dtype=torch.float32)
        self.num_keyframes = self.keyframes.shape[0]

        # Expand keyframes for all environments: [num_envs, num_keyframes, num_joints]
        self.keyframes = self.keyframes.unsqueeze(0).expand(self.num_envs, -1, -1).clone()

        # Motion state tracking
        self.current_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.steps_per_phase = 100

        # Current and target joint positions
        self.current_joint_pos = self.keyframes[:, 0].clone()
        self.target_joint_pos = self.keyframes[:, 0].clone()

    def _setup_scene(self):
        """Setup the scene with robot, table, cube, bowl, and lights."""
        # Add robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Add table
        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table

        # Add cube
        self.cube = RigidObject(self.cfg.cube_cfg)
        self.scene.rigid_objects["cube"] = self.cube

        # Add bowl (target)
        self.bowl = RigidObject(self.cfg.bowl_cfg)
        self.scene.rigid_objects["bowl"] = self.bowl

        # Add ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # Add dome light for better illumination
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply keyframe-interpolated joint positions."""
        for env_idx in range(self.num_envs):
            phase = self.current_phase[env_idx].item()
            step = self.phase_step[env_idx].item()

            if phase < self.num_keyframes:
                # Interpolate between current position and target keyframe
                alpha = (step + 1) / self.steps_per_phase
                target = self.keyframes[env_idx, phase]
                self.target_joint_pos[env_idx] = self.current_joint_pos[env_idx] * (1 - alpha) + target * alpha

                # Advance step
                self.phase_step[env_idx] += 1
                if self.phase_step[env_idx] >= self.steps_per_phase:
                    # Move to next phase
                    self.current_joint_pos[env_idx] = target.clone()
                    self.current_phase[env_idx] += 1
                    self.phase_step[env_idx] = 0

                    if self.current_phase[env_idx] < self.num_keyframes:
                        print(f"[INFO]: Starting phase {self.current_phase[env_idx].item() + 1}/{self.num_keyframes}")
                    else:
                        print("[INFO]: Motion sequence complete. Holding final pose.")
            else:
                # Hold final pose
                self.target_joint_pos[env_idx] = self.keyframes[env_idx, -1]

        # Set joint position targets for all joints (first 9 joints for Franka)
        self.robot.set_joint_position_target(self.target_joint_pos, joint_ids=list(range(9)))

    def _get_observations(self) -> dict:
        """Get observations (joint positions)."""
        obs = self.robot.data.joint_pos[:, :9]
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

        # Reset robot to home pose using default joint positions
        # Start with the default state and update with our home pose for the first 9 joints
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # Set the first 9 joints (arm + fingers) to our home keyframe
        home_pose = self.keyframes[env_ids, 0]
        joint_pos[:, :9] = home_pose

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset cube position
        cube_pos = self.cube.data.default_root_state[env_ids, :7].clone()
        cube_pos[:, :3] += self.scene.env_origins[env_ids]
        self.cube.write_root_pose_to_sim(cube_pos, env_ids)

        # Reset motion state
        self.current_phase[env_ids] = 0
        self.phase_step[env_ids] = 0
        self.current_joint_pos[env_ids] = home_pose.clone()
        self.target_joint_pos[env_ids] = home_pose.clone()


def main():
    """Main function."""
    # Create environment configuration
    print("Creating environment configuration...")
    env_cfg = FrankaPickCubeEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Create environment with rgb_array render mode for video recording
    print("Creating environment...")
    env = FrankaPickCubeEnv(env_cfg, render_mode="rgb_array")

    # Set camera view for a good angle of the robot and scene
    print("Setting camera view...")
    env.sim.set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.4])

    # Setup output directory
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "output", "franka_pick_cube_video"
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
    print("[INFO]: Franka arm will pick up the red cube and place it in the bowl")

    # Reset and run
    print("Resetting environment...")
    env.reset()

    with torch.inference_mode():
        for step in range(args_cli.video_length):
            # Actions are not directly used (motion is keyframe-based)
            actions = torch.zeros(
                (args_cli.num_envs, env_cfg.action_space), device=env.unwrapped.device, dtype=torch.float32
            )
            env.step(actions)

            if step % 100 == 0:
                print(f"[INFO]: Step {step}/{args_cli.video_length}")

    print("[INFO]: Recording complete")
    print(f"[INFO]: Video saved to {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
