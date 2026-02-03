# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Franka Box Sorting Demo with Video Recording.

This script demonstrates a Franka Panda robot environment with cardboard containers
and cereal boxes. Each cereal box is moved one at a time from the source container
to the destination container. The state trajectory of all containers is recorded.

Adapted from: boggart-sim/examples/box_sorting_franka.py

Usage:
    ./isaaclab.sh -p scripts/demos/franka_box_sorting_video.py --headless --enable_cameras --num_envs 1

"""

from __future__ import annotations

import argparse
import os
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Franka box sorting with video recording.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video_length", type=int, default=3000, help="Number of steps to record.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the video.")
parser.add_argument("--trajectory_file", type=str, default=None, help="Path to save trajectory data (JSON).")
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
import json

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg, UrdfConverter, UrdfConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils import configclass

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# Paths to asset files from boggart-sim
BOGGART_SIM_DATA_DIR = "/home/debian/boggart-sim/data"
BOGGART_SIM_EXAMPLES_DIR = "/home/debian/boggart-sim/examples"
COFFEE_TABLE_OBJ_PATH = os.path.join(BOGGART_SIM_DATA_DIR, "coffee_table", "coffee_table.obj")
CARDBOARD_BOX_URDF_PATH = os.path.join(BOGGART_SIM_EXAMPLES_DIR, "cardboard_box.urdf")
CHEERIOS_BOX_URDF_PATH = os.path.join(BOGGART_SIM_EXAMPLES_DIR, "cereal_box_simple.urdf")
CEREAL_BOX_OBJ_PATH = os.path.join(BOGGART_SIM_EXAMPLES_DIR, "cereal_box", "cereal_box.obj")

# Output directory for converted USD files
USD_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "converted_meshes")

# Packing configuration
PACKING_CONFIG = {
    "rows_x": 6,  # Number of cubes along x-axis (reduced for demo)
    "rows_y": 2,  # Number of cubes along y-axis
    "layers_z": 1,  # Number of vertical layers
    "gap": 0.001,  # Gap between cubes for physics stability
}

# Container configuration
CONTAINER_CONFIG = {
    "source_inner_size": (0.3937, 0.381, 0.311),  # (width, depth, height)
    "dest_inner_size": (0.4, 0.39, 0.1),  # Slightly larger to nest on top
    "wall_thickness": 0.01,
    "source_pos": (-0.3, 0.35),  # (x, y) center position
    "dest_pos": (0.3, 0.35),  # (x, y) center position
}


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


def convert_mesh_to_usd(obj_path: str, usd_name: str, scale: tuple = (1.0, 1.0, 1.0),
                         rotation: tuple = (1.0, 0.0, 0.0, 0.0), kinematic: bool = True) -> str:
    """Convert an OBJ mesh file to USD format.

    Args:
        obj_path: Path to the OBJ file
        usd_name: Name for the output USD file (without extension)
        scale: Scale factor (x, y, z)
        rotation: Rotation quaternion (w, x, y, z)
        kinematic: Whether the object is kinematic

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
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
        mass_props=schemas_cfg.MassPropertiesCfg(mass=100.0),
        mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
    )

    mesh_converter = MeshConverter(mesh_converter_cfg)
    print(f"[INFO]: Converted {obj_path} to {mesh_converter.usd_path}")
    return mesh_converter.usd_path


def convert_urdf_to_usd(urdf_path: str, usd_name: str, fix_base: bool = False, has_joints: bool = True) -> str:
    """Convert a URDF file to USD format.

    Args:
        urdf_path: Path to the URDF file
        usd_name: Name for the output USD file (without extension)
        fix_base: Whether to fix the base link
        has_joints: Whether the URDF has movable joints (if False, joint_drive is set to None)

    Returns:
        Path to the converted USD file
    """
    os.makedirs(USD_OUTPUT_DIR, exist_ok=True)

    # Configure joint drive settings
    if has_joints:
        joint_drive_cfg = UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=10.0,
            ),
        )
    else:
        joint_drive_cfg = None

    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=USD_OUTPUT_DIR,
        usd_file_name=f"{usd_name}.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=fix_base,
        joint_drive=joint_drive_cfg,
    )

    urdf_converter = UrdfConverter(urdf_converter_cfg)
    print(f"[INFO]: Converted {urdf_path} to {urdf_converter.usd_path}")
    return urdf_converter.usd_path


def compute_cuboid_size(
    container_inner: tuple, rows_x: int, rows_y: int, layers_z: int, gap: float
) -> tuple:
    """Compute cuboid dimensions for tightest fit in container."""
    w, d, h = container_inner
    size_x = (w - gap * (rows_x + 1)) / rows_x
    size_y = (d - gap * (rows_y + 1)) / rows_y
    size_z = (h - gap * (layers_z + 1)) / layers_z
    return (size_x, size_y, size_z)


def generate_cube_positions(
    container_center: tuple,
    container_inner: tuple,
    cuboid_size: tuple,
    rows_x: int,
    rows_y: int,
    layers_z: int,
    gap: float,
    table_z: float,
    wall_thickness: float,
) -> list:
    """Generate cuboid positions in a dense grid pattern."""
    cx, cy = container_center
    sx, sy, sz = cuboid_size
    positions = []

    total_x = rows_x * sx + (rows_x - 1) * gap
    total_y = rows_y * sy + (rows_y - 1) * gap

    start_x = cx - total_x / 2 + sx / 2
    start_y = cy - total_y / 2 + sy / 2
    start_z = table_z + wall_thickness + gap + sz/2

    for layer in range(layers_z):
        for row_y in range(rows_y):
            for row_x in range(rows_x):
                x = start_x + row_x * (sx + gap)
                y = start_y + row_y * (sy + gap)
                z = start_z + layer * (sz + gap)
                positions.append((x, y, z))

    return positions


# Convert mesh files to USD (done after simulation app is launched)
print("[INFO]: Converting mesh files to USD...")
COFFEE_TABLE_USD_PATH = convert_mesh_to_usd(
    COFFEE_TABLE_OBJ_PATH,
    "coffee_table_sorting",
    scale=(0.02, 0.02, 0.02),
    rotation=euler_to_quat(math.pi / 2, 0, 0),
)

# Convert URDFs to USD
print("[INFO]: Converting URDF files to USD...")
CARDBOARD_BOX_USD_PATH = convert_urdf_to_usd(
    CARDBOARD_BOX_URDF_PATH,
    "cardboard_box",
    fix_base=False,
    has_joints=True,  # Has flap joints
)

# Convert cereal box OBJ mesh to USD directly (URDF converter hangs on GLB)
# OBJ is already in meters (0.20 x 0.30 x 0.07m), scale factors from URDF fine-tune size
print("[INFO]: Converting cereal box mesh to USD...")
CEREAL_BOX_USD_PATH = convert_mesh_to_usd(
    CEREAL_BOX_OBJ_PATH,
    "cereal_box",
    scale=(0.93003, 1.02866, 0.92213),  # Scale factors from URDF
    rotation=euler_to_quat(math.pi / 2, 0, math.pi / 2),  # rpy="1.5708 0 1.5708" from URDF
    kinematic=False,
)


@configclass
class FrankaBoxSortingEnvCfg(DirectRLEnvCfg):
    """Configuration for the Franka box sorting environment."""

    # env
    decimation = 2
    episode_length_s = 120.0
    action_space = 18  # 2 robots x (7 arm joints + 2 gripper fingers)
    observation_space = 18  # 2 robots x 9 joint positions
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        device="cuda",
    )
    debug_vis = False

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # table configuration
    table_surface_z = 0.675

    # robot 1 - right side of source box, facing left toward the box
    robot_1_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_1_cfg.init_state.pos = (0.3, 0.7, 0.0)  # Right of source box at (-0.3, 0.35)
    robot_1_cfg.init_state.rot = euler_to_quat(0.0, 0.0, - math.pi / 2)  # Face left toward box

    # robot 2 - left side of source box, facing right toward the box
    robot_2_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot_2")
    robot_2_cfg.init_state.pos = (-0.3, 0.7, 0.0)  # Left of source box at (-0.3, 0.35)
    robot_2_cfg.init_state.rot = euler_to_quat(0.0, 0.0, - math.pi / 2)  # Face right toward box

    # table
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=COFFEE_TABLE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.2, 0.05)),
    )

    # Source container (cardboard box with flaps from URDF)
    source_pos = CONTAINER_CONFIG["source_pos"]
    source_inner_size = CONTAINER_CONFIG["source_inner_size"]
    wall_thickness = CONTAINER_CONFIG["wall_thickness"]
    source_outer_size = (
        source_inner_size[0] + 2 * wall_thickness,
        source_inner_size[1] + 2 * wall_thickness,
        source_inner_size[2] + wall_thickness,
    )

    # Source container articulation (cardboard box with hinged flaps)
    source_container_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/SourceContainer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CARDBOARD_BOX_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(source_pos[0], source_pos[1], table_surface_z+source_inner_size[2]),
            joint_pos={"flap_front_joint": 0.0, "flap_back_joint": 0.0, "flap_right_joint": 0.0, "flap_left_joint": 0.0},
        ),
        actuators={
            "flaps": ImplicitActuatorCfg(
                joint_names_expr=["flap_.*_joint"],
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )

    # Destination container (simple hollow box)
    dest_pos = CONTAINER_CONFIG["dest_pos"]
    dest_inner_size = CONTAINER_CONFIG["dest_inner_size"]


class FrankaBoxSortingEnv(DirectRLEnv):
    """Franka box sorting environment with container animation."""

    cfg: FrankaBoxSortingEnvCfg

    def __init__(self, cfg: FrankaBoxSortingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices for robot 1
        self.arm_joint_ids_1, _ = self.robot_1.find_joints("panda_joint.*")
        self.finger_joint_ids_1, _ = self.robot_1.find_joints("panda_finger_joint.*")

        # Get joint indices for robot 2
        self.arm_joint_ids_2, _ = self.robot_2.find_joints("panda_joint.*")
        self.finger_joint_ids_2, _ = self.robot_2.find_joints("panda_finger_joint.*")

        # Get joint indices for source container flaps
        self.flap_joint_ids, _ = self.source_container.find_joints("flap_.*_joint")

        # Home pose for robot (parked position)
        self.home_pose = torch.tensor(
            [0, -0.785, 0, -2.356, 0, 2.5, 0.785, 0.04, 0.04],
            device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        # Animation state
        self.animation_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_box_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Container animation parameters
        self.source_pivot = torch.tensor(
            [self.cfg.source_pos[0], self.cfg.source_pos[1], self.cfg.table_surface_z + 0.16],
            device=self.device, dtype=torch.float32
        )
        self.pivot_offset_z = 0.16
        self.lift_height = 0.4  # Lift boxes higher for transfer

        # Per-box movement phase durations (in steps)
        # Each box goes through: settle -> lift -> move -> lower -> release -> settle
        self.box_phase_durations = [
            30,    # Phase 0: Initial settle
            60,    # Phase 1: Lift box from source
            80,    # Phase 2: Move horizontally to destination
            60,    # Phase 3: Lower box into destination
            30,    # Phase 4: Release and settle
        ]
        self.steps_per_box = sum(self.box_phase_durations)

        # Global phases
        self.global_settle_steps = 100  # Initial settle before moving boxes
        self.final_settle_steps = 200   # Final settle after all boxes moved

        # Flap opening angle (radians, ~90 degrees)
        self.flap_open_angle = 1.57

        # Debug tracking
        self.debug_step_counter = 0
        self.last_debug_phase = -1
        self.last_box_idx = -1

        # Trajectory recording
        self.trajectory_data = {
            "timesteps": [],
            "source_container": {"positions": [], "orientations": []},
            "dest_container": {"positions": [], "orientations": []},
            "cereal_boxes": {}  # Will be populated with box indices
        }
        self.trajectory_recording = True

    def _setup_scene(self):
        """Setup the scene with robots, table, containers, and boxes."""
        # Add robot 1
        self.robot_1 = Articulation(self.cfg.robot_1_cfg)
        self.scene.articulations["robot_1"] = self.robot_1

        # Add robot 2
        self.robot_2 = Articulation(self.cfg.robot_2_cfg)
        self.scene.articulations["robot_2"] = self.robot_2

        # Add table
        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table

        # Add ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # Add dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Create source container (cardboard box URDF articulation)
        self._create_source_container_urdf()

        # Create destination container walls
        self._create_dest_container()

        # Create cereal boxes (from URDF)
        self._create_cereal_boxes_urdf()

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

    def _create_source_container_urdf(self):
        """Create the source container from cardboard box URDF (articulation with flaps)."""
        # Add source container articulation
        self.source_container = Articulation(self.cfg.source_container_cfg)
        self.scene.articulations["source_container"] = self.source_container

        # Store configuration for animation
        pos = self.cfg.source_pos
        inner = self.cfg.source_inner_size
        t = self.cfg.wall_thickness

        # Pivot offset: distance from URDF body origin to geometric center of container volume
        self.source_pivot_offset_z = (t + inner[2]) / 2

        # Note: flap_joint_ids will be set in __init__ after scene is built

    def _create_dest_container(self):
        """Create the destination container as a simple hollow box."""
        pos = self.cfg.dest_pos
        inner = self.cfg.dest_inner_size
        t = self.cfg.wall_thickness
        z_base = self.cfg.table_surface_z + inner[2]

        color = (0.2, 0.7, 0.4)

        # Bottom
        RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/DestBottom",
            spawn=sim_utils.CuboidCfg(
                size=(inner[0] + 2 * t, inner[1] + 2 * t, t),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(pos[0], pos[1], z_base + t / 2)),
        ))

        # Walls
        for name, wall_pos, size in [
            ("DestFront", (pos[0], pos[1] - inner[1] / 2 - t / 2, z_base + t + inner[2] / 2), (inner[0] + 2 * t, t, inner[2])),
            ("DestBack", (pos[0], pos[1] + inner[1] / 2 + t / 2, z_base + t + inner[2] / 2), (inner[0] + 2 * t, t, inner[2])),
            ("DestLeft", (pos[0] - inner[0] / 2 - t / 2, pos[1], z_base + t + inner[2] / 2), (t, inner[1], inner[2])),
            ("DestRight", (pos[0] + inner[0] / 2 + t / 2, pos[1], z_base + t + inner[2] / 2), (t, inner[1], inner[2])),
        ]:
            RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{name}",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=wall_pos),
            ))

    def _create_cereal_boxes_urdf(self):
        """Create cereal boxes from URDF inside the source container."""
        # The URDF cereal box has fixed dimensions from cheerios_box.urdf:
        # Collision box size: 0.06422 x 0.18870 x 0.30860
        # We use this size for positioning
        cuboid_size = (0.06422, 0.18870, 0.30860)

        positions = generate_cube_positions(
            container_center=self.cfg.source_pos,
            container_inner=self.cfg.source_inner_size,
            cuboid_size=cuboid_size,
            rows_x=PACKING_CONFIG["rows_x"],
            rows_y=PACKING_CONFIG["rows_y"],
            layers_z=PACKING_CONFIG["layers_z"],
            gap=PACKING_CONFIG["gap"],
            table_z=self.cfg.table_surface_z + self.cfg.source_inner_size[2],
            wall_thickness=self.cfg.wall_thickness,
        )

        self.num_boxes = len(positions)
        self.boxes = []
        self.box_initial_positions = []

        for i, pos in enumerate(positions):
            box = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/CerealBox_{i}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=CEREAL_BOX_USD_PATH,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            ))
            self.boxes.append(box)
            self.box_initial_positions.append(torch.tensor(pos, device=self.device))
            self.scene.rigid_objects[f"cereal_box_{i}"] = box

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply robot positions and animate individual box movement."""
        # Keep both robots at home pose
        self.robot_1.set_joint_position_target(self.home_pose, joint_ids=list(range(9)))
        self.robot_2.set_joint_position_target(self.home_pose, joint_ids=list(range(9)))

        # Animate individual box movement
        for env_idx in range(self.num_envs):
            self._animate_box_transfer(env_idx)

        # Record trajectory
        if self.trajectory_recording:
            self._record_trajectory()

        # Advance animation step
        self.animation_step += 1

    def _animate_box_transfer(self, env_idx: int):
        """Animate moving cereal boxes one at a time from source to destination."""
        step = self.animation_step[env_idx].item()

        # Configuration
        source_pos = self.cfg.source_pos
        dest_pos = self.cfg.dest_pos
        table_z = self.cfg.table_surface_z
        source_inner = self.cfg.source_inner_size
        dest_inner = self.cfg.dest_inner_size

        # Determine which box we're moving and what phase
        if step < self.global_settle_steps:
            # Initial settle phase - keep flaps closed
            self._set_flap_angle(env_idx, 0.0)
            return

        # Open flaps during box movement
        self._set_flap_angle(env_idx, self.flap_open_angle)

        # Calculate which box and what phase within that box's movement
        box_step = step - self.global_settle_steps
        box_idx = min(box_step // self.steps_per_box, self.num_boxes - 1)
        step_within_box = box_step % self.steps_per_box

        # Check if all boxes have been moved
        if box_idx >= self.num_boxes:
            return

        self.current_box_idx[env_idx] = box_idx

        # Debug print on box change
        if box_idx != self.last_box_idx:
            self.last_box_idx = box_idx
            print(f"[Step {step}] Moving box {box_idx + 1}/{self.num_boxes}")

        # Determine phase within box movement
        cumulative = 0
        phase = 0
        phase_step = 0
        for p, duration in enumerate(self.box_phase_durations):
            if step_within_box < cumulative + duration:
                phase = p
                phase_step = step_within_box - cumulative
                break
            cumulative += duration

        self.current_phase[env_idx] = phase

        # Get this box's initial position and compute destination
        box = self.boxes[box_idx]
        initial_pos = self.box_initial_positions[box_idx]

        # Destination position: stack boxes in destination container
        # Arrange in a grid similar to source
        rows_x = PACKING_CONFIG["rows_x"]
        rows_y = PACKING_CONFIG["rows_y"]
        dest_row_x = box_idx % rows_x
        dest_row_y = (box_idx // rows_x) % rows_y
        dest_layer = box_idx // (rows_x * rows_y)

        cuboid_size = (0.06422, 0.18870, 0.30860)
        gap = PACKING_CONFIG["gap"]

        total_x = rows_x * cuboid_size[0] + (rows_x - 1) * gap
        total_y = rows_y * cuboid_size[1] + (rows_y - 1) * gap

        dest_x = dest_pos[0] - total_x / 2 + cuboid_size[0] / 2 + dest_row_x * (cuboid_size[0] + gap)
        dest_y = dest_pos[1] - total_y / 2 + cuboid_size[1] / 2 + dest_row_y * (cuboid_size[1] + gap)
        dest_z = table_z + dest_inner[2] + 0.01 + cuboid_size[2] / 2 + dest_layer * (cuboid_size[2] + gap)

        dest_target = torch.tensor([dest_x, dest_y, dest_z], device=self.device, dtype=torch.float32)

        # Lifted position (above source)
        lift_z = table_z + source_inner[2] + self.lift_height
        lifted_source = torch.tensor([initial_pos[0], initial_pos[1], lift_z], device=self.device, dtype=torch.float32)

        # Lifted position above destination
        lifted_dest = torch.tensor([dest_x, dest_y, lift_z], device=self.device, dtype=torch.float32)

        # Calculate target position based on phase
        if phase == 0:
            # Initial settle - box at start position
            target_pos = initial_pos
        elif phase == 1:
            # Lift box from source
            alpha = phase_step / self.box_phase_durations[1]
            alpha = self._smooth_step(alpha)
            target_pos = initial_pos * (1 - alpha) + lifted_source * alpha
        elif phase == 2:
            # Move horizontally to destination
            alpha = phase_step / self.box_phase_durations[2]
            alpha = self._smooth_step(alpha)
            target_pos = lifted_source * (1 - alpha) + lifted_dest * alpha
        elif phase == 3:
            # Lower box into destination
            alpha = phase_step / self.box_phase_durations[3]
            alpha = self._smooth_step(alpha)
            target_pos = lifted_dest * (1 - alpha) + dest_target * alpha
        else:
            # Release and settle
            target_pos = dest_target

        # Apply kinematic control to move the box
        self._set_box_position(env_idx, box_idx, target_pos)

    def _smooth_step(self, t: float) -> float:
        """Smoothstep interpolation for smoother motion."""
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)

    def _set_flap_angle(self, env_idx: int, flap_angle: float):
        """Set flap joint positions."""
        if self.flap_joint_ids:
            flap_targets = torch.full(
                (self.num_envs, len(self.flap_joint_ids)),
                flap_angle,
                device=self.device, dtype=torch.float32
            )
            self.source_container.set_joint_position_target(flap_targets, joint_ids=self.flap_joint_ids)

    def _set_box_position(self, env_idx: int, box_idx: int, target_pos: torch.Tensor):
        """Set a box's position using kinematic control."""
        box = self.boxes[box_idx]

        # Get current state
        root_state = box.data.root_state_w[env_idx]
        current_pos = root_state[:3] - self.scene.env_origins[env_idx]

        # Check for invalid state
        if torch.isnan(current_pos).any():
            return

        # Compute velocity for smooth motion
        dt = 1.0 / 120.0
        target_lin_vel = (target_pos - current_pos) / dt
        max_vel = 3.0  # m/s
        target_lin_vel = torch.clamp(target_lin_vel, -max_vel, max_vel)

        # Build new root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        new_root_state = torch.zeros(13, device=self.device, dtype=torch.float32)
        new_root_state[:3] = target_pos + self.scene.env_origins[env_idx]
        new_root_state[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # Identity quaternion
        new_root_state[7:10] = target_lin_vel
        new_root_state[10:13] = torch.zeros(3, device=self.device)  # No angular velocity

        # Write the new root state
        box.write_root_state_to_sim(
            root_state=new_root_state.unsqueeze(0),
            env_ids=torch.tensor([env_idx], device=self.device),
        )

    def _record_trajectory(self):
        """Record the state trajectory of all containers."""
        step = self.animation_step[0].item()
        self.trajectory_data["timesteps"].append(step)

        # Record source container state
        src_state = self.source_container.data.root_state_w[0]
        src_pos = src_state[:3].cpu().numpy().tolist()
        src_quat = src_state[3:7].cpu().numpy().tolist()
        self.trajectory_data["source_container"]["positions"].append(src_pos)
        self.trajectory_data["source_container"]["orientations"].append(src_quat)

        # Record destination container state (from rigid objects in scene)
        # The dest container is created as separate walls, so we record the bottom piece position
        dest_state = self.scene.rigid_objects.get("dest_container")
        if dest_state is None:
            # Fallback: record the configured position
            dest_pos = [self.cfg.dest_pos[0], self.cfg.dest_pos[1], self.cfg.table_surface_z]
            dest_quat = [1.0, 0.0, 0.0, 0.0]
        else:
            d_state = dest_state.data.root_state_w[0]
            dest_pos = d_state[:3].cpu().numpy().tolist()
            dest_quat = d_state[3:7].cpu().numpy().tolist()
        self.trajectory_data["dest_container"]["positions"].append(dest_pos)
        self.trajectory_data["dest_container"]["orientations"].append(dest_quat)

        # Record all cereal box states
        for i, box in enumerate(self.boxes):
            box_key = f"box_{i}"
            if box_key not in self.trajectory_data["cereal_boxes"]:
                self.trajectory_data["cereal_boxes"][box_key] = {"positions": [], "orientations": []}

            box_state = box.data.root_state_w[0]
            box_pos = box_state[:3].cpu().numpy().tolist()
            box_quat = box_state[3:7].cpu().numpy().tolist()
            self.trajectory_data["cereal_boxes"][box_key]["positions"].append(box_pos)
            self.trajectory_data["cereal_boxes"][box_key]["orientations"].append(box_quat)

    def save_trajectory(self, filepath: str):
        """Save trajectory data to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.trajectory_data, f, indent=2)
        print(f"[INFO]: Trajectory saved to {filepath}")


    def _get_observations(self) -> dict:
        """Get observations."""
        # Concatenate observations from both robots
        obs_1 = self.robot_1.data.joint_pos[:, :9]
        obs_2 = self.robot_2.data.joint_pos[:, :9]
        obs = torch.cat([obs_1, obs_2], dim=-1)
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
            env_ids = self.robot_1._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot 1 to home pose
        joint_pos_1 = self.robot_1.data.default_joint_pos[env_ids].clone()
        joint_vel_1 = self.robot_1.data.default_joint_vel[env_ids].clone()
        joint_pos_1[:, :9] = self.home_pose[env_ids]
        self.robot_1.write_joint_state_to_sim(joint_pos_1, joint_vel_1, None, env_ids)

        # Reset robot 2 to home pose
        joint_pos_2 = self.robot_2.data.default_joint_pos[env_ids].clone()
        joint_vel_2 = self.robot_2.data.default_joint_vel[env_ids].clone()
        joint_pos_2[:, :9] = self.home_pose[env_ids]
        self.robot_2.write_joint_state_to_sim(joint_pos_2, joint_vel_2, None, env_ids)

        # Reset animation state
        self.animation_step[env_ids] = 0
        self.current_phase[env_ids] = 0
        self.current_box_idx[env_ids] = 0


def main():
    """Main function."""
    print("Creating environment configuration...")
    env_cfg = FrankaBoxSortingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    print("Creating environment...")
    env = FrankaBoxSortingEnv(env_cfg, render_mode="rgb_array")

    print("Setting camera view...")
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.2], target=[0.0, 0.35, 0.5])

    # Setup output directory
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "output", "franka_box_sorting_video"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Setup trajectory file path
    if args_cli.trajectory_file:
        trajectory_file = args_cli.trajectory_file
    else:
        trajectory_file = os.path.join(output_dir, "trajectory.json")

    # Wrap with RecordVideo
    print("Wrapping environment with RecordVideo...")
    env = gym.wrappers.RecordVideo(
        env,
        output_dir,
        episode_trigger=lambda x: True,
        video_length=args_cli.video_length,
        disable_logger=True,
    )

    print(f"[INFO]: Recording video to {output_dir}")
    print(f"[INFO]: Running for {args_cli.video_length} steps")
    print(f"[INFO]: {env.unwrapped.num_boxes} cereal boxes in source container")
    print(f"[INFO]: Animation: Each box moves one at a time to destination")

    # Calculate total animation duration
    unwrapped_env = env.unwrapped
    total_animation_steps = (
        unwrapped_env.global_settle_steps +
        unwrapped_env.num_boxes * unwrapped_env.steps_per_box +
        unwrapped_env.final_settle_steps
    )
    print(f"[INFO]: Total animation duration: {total_animation_steps} steps")
    print(f"[INFO]: Steps per box: {unwrapped_env.steps_per_box}")

    if args_cli.video_length < total_animation_steps:
        print(f"[WARNING]: video_length ({args_cli.video_length}) < total animation ({total_animation_steps})")
        print(f"[WARNING]: Consider using --video_length {total_animation_steps + 100} to capture full animation")

    print("Resetting environment...")
    env.reset()

    with torch.inference_mode():
        for step in range(args_cli.video_length):
            actions = torch.zeros(
                (args_cli.num_envs, env_cfg.action_space), device=unwrapped_env.device, dtype=torch.float32
            )
            env.step(actions)

            if step % 100 == 0:
                box_idx = unwrapped_env.current_box_idx[0].item()
                phase = unwrapped_env.current_phase[0].item()
                print(f"[INFO]: Step {step}/{args_cli.video_length}, Box {box_idx + 1}/{unwrapped_env.num_boxes}, Phase {phase}")

    # Save trajectory data
    unwrapped_env.save_trajectory(trajectory_file)

    print("[INFO]: Recording complete")
    print(f"[INFO]: Video saved to {output_dir}")
    print(f"[INFO]: Trajectory saved to {trajectory_file}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
