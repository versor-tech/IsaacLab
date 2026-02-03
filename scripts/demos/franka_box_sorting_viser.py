# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Franka Box Sorting Demo with Interactive Viser Visualization.

This script demonstrates a Franka Panda robot environment with cardboard containers
and cereal boxes being dumped from one container to another. The source container
is lifted, flipped, and shaken to dump the boxes out.

Uses Viser for interactive 3D visualization with mouse/keyboard controls:
- Left-click + drag: Orbit camera
- Right-click + drag: Pan camera
- Scroll: Zoom in/out

Adapted from: boggart-sim/examples/box_sorting_franka.py

Usage:
    ./isaaclab.sh -p scripts/demos/franka_box_sorting_viser.py --headless --enable_cameras --num_envs 1

Then open http://localhost:8080 in your browser.
"""

from __future__ import annotations

import argparse
import os
import math
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Franka box sorting with viser visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import viser

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg, UrdfConverter, UrdfConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_mul

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# Paths to asset files from boggart-sim
BOGGART_SIM_DATA_DIR = "/home/debian/boggart-sim/data"
BOGGART_SIM_EXAMPLES_DIR = "/home/debian/boggart-sim/examples"
COFFEE_TABLE_OBJ_PATH = os.path.join(BOGGART_SIM_DATA_DIR, "coffee_table", "coffee_table.obj")
CARDBOARD_BOX_URDF_PATH = os.path.join(BOGGART_SIM_EXAMPLES_DIR, "cardboard_box.urdf")
DEST_CONTAINER_URDF_PATH = os.path.join(BOGGART_SIM_EXAMPLES_DIR, "destination_container.urdf")
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


def wxyz_to_rotation_matrix(wxyz):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix using scipy."""
    w, x, y, z = wxyz
    quat_xyzw = np.array([x, y, z, w])
    return Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)


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

DEST_CONTAINER_USD_PATH = convert_urdf_to_usd(
    DEST_CONTAINER_URDF_PATH,
    "destination_container",
    fix_base=False,
    has_joints=False,  # No joints, single rigid body
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
class FrankaBoxSortingViserEnvCfg(DirectRLEnvCfg):
    """Configuration for the Franka box sorting environment with Viser."""

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
        device="cpu",
    )
    debug_vis = False

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # table configuration
    table_surface_z = 0.675

    # Default Franka joint positions (from FRANKA_PANDA_CFG)
    _franka_default_joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -1.810,
        "panda_joint5": 0.0,
        "panda_joint6": 0.0,
        "panda_joint7": 0.741,
        "panda_finger_joint.*": 0.04,
    }

    # robot 1 - behind source box (higher y), facing forward toward box
    robot_1_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_1",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(CONTAINER_CONFIG["source_pos"][0], CONTAINER_CONFIG["source_pos"][1] + 0.6, 0.0),
            rot=euler_to_quat(0.0, 0.0, -math.pi / 2),  # Face forward (-y direction)
            joint_pos=_franka_default_joint_pos,
        ),
    )

    # robot 2 - in front of source box (lower y), facing backward toward box
    robot_2_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_2",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(CONTAINER_CONFIG["source_pos"][0], CONTAINER_CONFIG["source_pos"][1] - 0.6, 0.0),
            rot=euler_to_quat(0.0, 0.0, math.pi / 2),  # Face backward (+y direction)
            joint_pos=_franka_default_joint_pos,
        ),
    )

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

    # Destination container (single rigid body with walls from URDF)
    dest_pos = CONTAINER_CONFIG["dest_pos"]
    dest_inner_size = CONTAINER_CONFIG["dest_inner_size"]

    dest_container_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/DestContainer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEST_CONTAINER_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(dest_pos[0], dest_pos[1], table_surface_z),
        ),
    )

    # Camera for rendering (will be updated dynamically from Viser)
    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/ViserCamera",
        update_period=0.0,  # Update every physics step
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
    )


class FrankaBoxSortingViserEnv(DirectRLEnv):
    """Franka box sorting environment with Viser visualization."""

    cfg: FrankaBoxSortingViserEnvCfg

    def __init__(self, cfg: FrankaBoxSortingViserEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices for robot 1
        self.arm_joint_ids_1, self.arm_joint_names_1 = self.robot_1.find_joints("panda_joint.*")
        self.finger_joint_ids_1, self.finger_joint_names_1 = self.robot_1.find_joints("panda_finger_joint.*")
        self.all_joint_ids_1 = self.arm_joint_ids_1 + self.finger_joint_ids_1

        # Get joint indices for robot 2
        self.arm_joint_ids_2, self.arm_joint_names_2 = self.robot_2.find_joints("panda_joint.*")
        self.finger_joint_ids_2, self.finger_joint_names_2 = self.robot_2.find_joints("panda_finger_joint.*")
        self.all_joint_ids_2 = self.arm_joint_ids_2 + self.finger_joint_ids_2


        # Get joint indices for source container flaps
        self.flap_joint_ids, _ = self.source_container.find_joints("flap_.*_joint")

        # Home pose for robot (parked position with gripper open)
        self.home_pose = torch.tensor(
            [0, -0.785, 0, -1.356, 0, 0, 0.785, 0.04, 0.04],
            device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        # Gripper states
        self.gripper_open = 0.04  # Open gripper width
        self.gripper_closed = 0.0  # Closed gripper width

        # Animation state
        self.animation_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Setup differential IK controllers for both robots
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",  # Damped least squares
            ik_params={"lambda_val": 0.1},
        )
        self.ik_controller_1 = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)
        self.ik_controller_2 = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # Get end-effector body indices (panda_hand)
        self.ee_body_idx_1 = self.robot_1.find_bodies("panda_hand")[0][0]
        self.ee_body_idx_2 = self.robot_2.find_bodies("panda_hand")[0][0]

        # Get Jacobian body indices relative to root
        # For fixed-base robots, Jacobian index is body_id - 1 (root not included in Jacobians)
        # For floating-base robots, Jacobian index is body_id
        if self.robot_1.is_fixed_base:
            self.jacobi_body_idx_1 = self.ee_body_idx_1 - 1
        else:
            self.jacobi_body_idx_1 = self.ee_body_idx_1

        if self.robot_2.is_fixed_base:
            self.jacobi_body_idx_2 = self.ee_body_idx_2 - 1
        else:
            self.jacobi_body_idx_2 = self.ee_body_idx_2

        # Container grasp parameters
        self.box_center = torch.tensor(
            [self.cfg.source_pos[0], self.cfg.source_pos[1], self.cfg.table_surface_z + self.cfg.source_inner_size[2] / 2],
            device=self.device, dtype=torch.float32
        )
        self.box_half_depth = self.cfg.source_inner_size[1] / 2 + 0.02  # Half depth + offset for grasp

        # Define grasp poses (robot 1 grasps back, robot 2 grasps front)
        self.grasp_height = self.cfg.table_surface_z + self.cfg.source_inner_size[2] / 2  # Mid-height of box
        self.pre_grasp_offset = 0.15  # Distance above grasp point for approach
        self.lift_height = 0.3  # How high to lift the box

        # End-effector orientations for grasping (defined in robot base frame)
        # Gripper pointing forward (+x in base frame) with fingers horizontal
        # 90° rotation around y-axis to point gripper forward instead of down
        self.ee_quat_base = torch.tensor([0.707, 0.0, 0.707, 0.0], device=self.device, dtype=torch.float32)
        self.ee_quat_1 = self.ee_quat_base.clone()
        self.ee_quat_2 = self.ee_quat_base.clone()

        # Robot manipulation phase durations (in steps)
        self.phase_durations = [
            60,    # Phase 0: Initial settle / move to home
            120,   # Phase 1: Move to pre-grasp position
            80,    # Phase 2: Move down to grasp position
            30,    # Phase 3: Close grippers
            120,   # Phase 4: Lift the box
            200,   # Phase 5: Hold lifted position
        ]
        self.total_phases = len(self.phase_durations)

        # Flap opening angle (radians, ~90 degrees)
        self.flap_open_angle = 1.57

        # Current target poses for IK (will be updated each phase)
        self.target_ee_pos_1 = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.target_ee_pos_2 = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.target_ee_quat_1 = self.ee_quat_1.unsqueeze(0).expand(self.num_envs, -1).clone()
        self.target_ee_quat_2 = self.ee_quat_2.unsqueeze(0).expand(self.num_envs, -1).clone()

        # Joint position targets (will be computed via IK)
        self.joint_targets_1 = self.home_pose.clone()
        self.joint_targets_2 = self.home_pose.clone()

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

        # Add camera for Viser rendering
        self.viser_camera = Camera(self.cfg.camera_cfg)
        self.scene.sensors["viser_camera"] = self.viser_camera

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

    def _create_dest_container(self):
        """Create the destination container from the destination container USD."""
        self.dest_container = RigidObject(self.cfg.dest_container_cfg)
        self.scene.rigid_objects["dest_container"] = self.dest_container

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
        """Apply robot manipulation to grasp and lift the box."""
        # Compute current phase and phase progress
        step = self.animation_step[0].item()
        cumulative = 0
        phase = 0
        phase_step = 0

        for p, duration in enumerate(self.phase_durations):
            if step < cumulative + duration:
                phase = p
                phase_step = step - cumulative
                break
            cumulative += duration
        else:
            phase = self.total_phases - 1
            phase_step = self.phase_durations[-1] - 1

        self.current_phase[0] = phase
        alpha = phase_step / max(1, self.phase_durations[phase])

        # Get current box position (may have moved if grasped)
        box_state = self.source_container.data.root_state_w[0]
        box_pos = box_state[:3].unsqueeze(0) - self.scene.env_origins  # World frame position (1, 3)

        # Compute grasp positions based on current box position
        grasp_pos_1 = box_pos.clone()
        grasp_pos_1[:, 1] += self.box_half_depth - 0.01  # Behind the box (+y)
        grasp_pos_1[:, 2] = self.grasp_height

        grasp_pos_2 = box_pos.clone()
        grasp_pos_2[:, 1] -= self.box_half_depth + 0.01  # In front of box (-y)
        grasp_pos_2[:, 2] = self.grasp_height

        pre_grasp_pos_1 = grasp_pos_1.clone()
        # pre_grasp_pos_1[:, 2] += self.pre_grasp_offset

        pre_grasp_pos_2 = grasp_pos_2.clone()
        # pre_grasp_pos_2[:, 2] += self.pre_grasp_offset

        # Lifted positions
        lifted_pos_1 = grasp_pos_1.clone()
        lifted_pos_1[:, 2] += self.lift_height

        lifted_pos_2 = grasp_pos_2.clone()
        lifted_pos_2[:, 2] += self.lift_height

        # Gripper targets
        gripper_target = self.gripper_open

        if phase == 0:
            # Phase 0: Move to home position
            self.target_ee_pos_1[0] = pre_grasp_pos_1
            self.target_ee_pos_2[0] = pre_grasp_pos_2
            gripper_target = self.gripper_open

        elif phase == 1:
            # Phase 1: Move to pre-grasp position
            self.target_ee_pos_1[0] = pre_grasp_pos_1
            self.target_ee_pos_2[0] = pre_grasp_pos_2
            gripper_target = self.gripper_open

        elif phase == 2:
            # Phase 2: Move down to grasp position
            self.target_ee_pos_1[0] = pre_grasp_pos_1 * (1 - alpha) + grasp_pos_1 * alpha
            self.target_ee_pos_2[0] = pre_grasp_pos_2 * (1 - alpha) + grasp_pos_2 * alpha
            gripper_target = self.gripper_open

        elif phase == 3:
            # Phase 3: Close grippers
            self.target_ee_pos_1[0] = grasp_pos_1
            self.target_ee_pos_2[0] = grasp_pos_2
            gripper_target = self.gripper_open * (1 - alpha) + self.gripper_closed * alpha

        elif phase == 4:
            # Phase 4: Lift the box
            self.target_ee_pos_1[0] = grasp_pos_1 * (1 - alpha) + lifted_pos_1 * alpha
            self.target_ee_pos_2[0] = grasp_pos_2 * (1 - alpha) + lifted_pos_2 * alpha
            gripper_target = self.gripper_closed

        else:
            # Phase 5+: Hold lifted position
            self.target_ee_pos_1[0] = lifted_pos_1
            self.target_ee_pos_2[0] = lifted_pos_2
            gripper_target = self.gripper_closed

        print("IK targets for robot 1")
        # Compute IK for robot 1
        self._compute_ik_targets(
            robot=self.robot_1,
            ik_controller=self.ik_controller_1,
            target_pos=self.target_ee_pos_1,
            target_quat=self.target_ee_quat_1,
            ee_body_idx=self.ee_body_idx_1,
            jacobi_body_idx=self.jacobi_body_idx_1,
            joint_targets=self.joint_targets_1,
        )

        print("IK targets for robot 2")
        # Compute IK for robot 2
        self._compute_ik_targets(
            robot=self.robot_2,
            ik_controller=self.ik_controller_2,
            target_pos=self.target_ee_pos_2,
            target_quat=self.target_ee_quat_2,
            ee_body_idx=self.ee_body_idx_2,
            jacobi_body_idx=self.jacobi_body_idx_2,
            joint_targets=self.joint_targets_2,
        )

        # Set gripper targets
        self.joint_targets_1[:, 7:9] = gripper_target
        self.joint_targets_2[:, 7:9] = gripper_target

        # Apply joint position targets using resolved joint IDs
        self.robot_1.set_joint_position_target(self.joint_targets_1, joint_ids=self.all_joint_ids_1)
        self.robot_2.set_joint_position_target(self.joint_targets_2, joint_ids=self.all_joint_ids_2)

        # Debug: print applied targets vs actual positions
        print(f"[APPLY] Robot 1 joint targets: {self.joint_targets_1[0, :7].cpu().numpy()}")
        print(f"[APPLY] Robot 1 actual joints: {self.robot_1.data.joint_pos[0, :7].cpu().numpy()}")
        print(f"[APPLY] Robot 2 joint targets: {self.joint_targets_2[0, :7].cpu().numpy()}")
        print(f"[APPLY] Robot 2 actual joints: {self.robot_2.data.joint_pos[0, :7].cpu().numpy()}")
        print("=== END STEP ===\n")

        # Advance animation step
        self.animation_step += 1

    def _compute_ik_targets(
        self,
        robot: Articulation,
        ik_controller: DifferentialIKController,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        ee_body_idx: int,
        jacobi_body_idx: int,
        joint_targets: torch.Tensor,
    ):
        """Compute IK joint targets for a robot.

        Args:
            robot: The robot articulation
            ik_controller: The differential IK controller
            target_pos: Target end-effector position in env frame (num_envs, 3)
            target_quat: Target end-effector quaternion (num_envs, 4)
            ee_body_idx: End-effector body index
            jacobi_body_idx: Jacobian body index relative to root
            joint_targets: Output joint targets tensor to update
        """
        
        # Get current joint positions (arm only, first 7 joints)
        current_joint_pos = robot.data.joint_pos[:, :7]

        # Get current end-effector pose in world frame
        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]
        ee_pos_w = ee_pose_w[:, 0:3]
        ee_quat_w = ee_pose_w[:, 3:7]

        # Transform target position from env frame to world frame
        target_pos_w = target_pos + self.scene.env_origins

        # Get robot root orientation in world frame
        root_quat_w = robot.data.root_state_w[:, 3:7]

        # Target orientation: gripper pointing forward in base frame
        # Transform from base frame to world frame by composing with root orientation
        target_quat_b = self.ee_quat_base.unsqueeze(0).expand(self.num_envs, -1)  # (num_envs, 4)
        target_quat_w = quat_mul(root_quat_w, target_quat_b)

        # Debug prints
        print(f"[DEBUG] target_pos (env frame): {target_pos[0].cpu().numpy()}")
        print(f"[DEBUG] target_pos_w (world frame): {target_pos_w[0].cpu().numpy()}")
        print(f"[DEBUG] ee_pos_w (current EE in world frame): {ee_pos_w[0].cpu().numpy()}")
        print("---")

        # Get Jacobian in world frame
        jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, :7]

        # Set IK controller command (target pose in world frame)
        ik_controller.set_command(torch.cat([target_pos_w, target_quat_w], dim=-1))

        # Compute IK
        joint_pos_des = ik_controller.compute(
            ee_pos_w, ee_quat_w, jacobian, current_joint_pos
        )

        # Debug IK computation
        print(f"[DEBUG] target_quat_w: {target_quat_w[0].cpu().numpy()}")
        print(f"[DEBUG] current_joint_pos: {current_joint_pos[0].cpu().numpy()}")
        print(f"[DEBUG] joint_pos_des (IK output): {joint_pos_des[0].cpu().numpy()}")
        print(f"[DEBUG] joint_pos_delta: {(joint_pos_des - current_joint_pos)[0].cpu().numpy()}")
        print("===")

        # Update arm joint targets
        joint_targets[:, :7] = joint_pos_des


    def set_camera_pose(self, position: np.ndarray, rotation_matrix: np.ndarray):
        """Set the Viser camera pose.

        Args:
            position: Camera position in world coordinates (3,)
            rotation_matrix: Camera rotation matrix (3, 3)
        """
        # Convert rotation matrix to quaternion (w, x, y, z)
        rot = Rotation.from_matrix(rotation_matrix)
        quat_xyzw = rot.as_quat()  # scipy returns (x, y, z, w)
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Update camera position and orientation for all environments
        pos_tensor = torch.tensor(position, device=self.device, dtype=torch.float32).unsqueeze(0)
        quat_tensor = torch.tensor(quat_wxyz, device=self.device, dtype=torch.float32).unsqueeze(0)

        # Expand for all environments
        pos_tensor = pos_tensor.expand(self.num_envs, -1).clone()
        quat_tensor = quat_tensor.expand(self.num_envs, -1).clone()

        # Add environment origins to position
        pos_tensor = pos_tensor + self.scene.env_origins

        # Set camera pose
        self.viser_camera.set_world_poses(pos_tensor, quat_tensor)

    def get_camera_image(self) -> np.ndarray:
        """Get the current camera RGB image.

        Returns:
            RGB image as numpy array (H, W, 3) with values 0-255
        """
        # Update camera data
        self.viser_camera.update(self.cfg.sim.dt)

        # Get RGB image from camera
        rgb_data = self.viser_camera.data.output["rgb"]

        if rgb_data is not None and rgb_data.numel() > 0:
            # Shape: (num_envs, H, W, 4) for RGBA or (num_envs, H, W, 3) for RGB
            frame = rgb_data[0].cpu().numpy()  # Take first environment

            # Handle RGBA vs RGB
            if frame.shape[-1] == 4:
                frame = frame[..., :3]  # Drop alpha channel

            # Ensure uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            return frame
        else:
            # Return blank frame if no data
            return np.zeros((self.cfg.camera_cfg.height, self.cfg.camera_cfg.width, 3), dtype=np.uint8)

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

        # Reset IK controllers
        self.ik_controller_1.reset(env_ids)
        self.ik_controller_2.reset(env_ids)

        # Reset joint targets
        self.joint_targets_1[env_ids] = self.home_pose[env_ids]
        self.joint_targets_2[env_ids] = self.home_pose[env_ids]

        # Reset animation state
        self.animation_step[env_ids] = 0
        self.current_phase[env_ids] = 0


def main():
    """Main function."""
    print("Creating environment configuration...")
    env_cfg = FrankaBoxSortingViserEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    print("Creating environment...")
    env = FrankaBoxSortingViserEnv(env_cfg, render_mode="rgb_array")

    print("Setting initial camera view...")
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.2], target=[0.0, 0.35, 0.5])

    # Create Viser server
    print(f"Starting Viser server on port {args_cli.port}...")
    server = viser.ViserServer(port=args_cli.port)
    print(f"Viser server started at http://localhost:{args_cli.port}")
    print("Controls:")
    print("  - Left-click + drag: Orbit camera")
    print("  - Right-click + drag: Pan camera")
    print("  - Scroll: Zoom in/out")

    # Calculate total animation duration
    total_animation_steps = sum(env.phase_durations)
    print(f"[INFO]: {env.num_boxes} cereal boxes in source container")
    print(f"[INFO]: Total animation duration: {total_animation_steps} steps")
    print("[INFO]: Animation phases:")
    print("  - Phase 0: Initial settle / move to home")
    print("  - Phase 1: Move to pre-grasp position")
    print("  - Phase 2: Move down to grasp position")
    print("  - Phase 3: Close grippers")
    print("  - Phase 4: Lift the box")
    print("  - Phase 5: Hold lifted position")

    # Reset environment
    print("Resetting environment...")
    env.reset()

    # Default camera pose (used when no client is connected)
    default_eye = np.array([1.5, 1.5, 1.2])
    default_target = np.array([0.0, 0.35, 0.5])

    frame_count = 0
    last_time = time.time()

    print("\n[INFO]: Waiting for browser connection...")

    with torch.inference_mode():
        while True:
            # Step simulation with zero actions
            actions = torch.zeros(
                (args_cli.num_envs, env_cfg.action_space), device=env.device, dtype=torch.float32
            )
            env.step(actions)

            # Get camera pose from Viser clients and render
            clients = server.get_clients()
            if clients:
                for client in clients.values():
                    camera = client.camera

                    # Get camera pose from Viser
                    cam_pos = np.array(camera.position)
                    cam_wxyz = camera.wxyz

                    # Convert quaternion to rotation matrix
                    R = wxyz_to_rotation_matrix(cam_wxyz)

                    # Set camera pose in simulation
                    env.set_camera_pose(cam_pos, R)

                    # Get rendered frame
                    frame = env.get_camera_image()

                    # Send frame to Viser client
                    client.set_background_image(frame, format="jpeg", jpeg_quality=80)

            # FPS counter
            frame_count += 1
            if frame_count % 60 == 0:
                now = time.time()
                fps = 60 / (now - last_time)
                last_time = now
                phase = env.current_phase[0].item()
                step = env.animation_step[0].item()
                print(f"[INFO]: FPS: {fps:.1f}, Step {step}/{total_animation_steps}, Phase {phase}")

            # Reset animation if completed
            if env.animation_step[0].item() >= total_animation_steps:
                print("[INFO]: Animation complete, restarting...")
                env.reset()

            # Small sleep to avoid overwhelming
            time.sleep(0.008)  # ~120Hz max


if __name__ == "__main__":
    main()
    simulation_app.close()
