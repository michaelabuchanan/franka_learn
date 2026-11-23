# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils

from . import mdp

##
# Pre-defined configs
##

from .fr3 import FRANKA_FR3_CFG


##
# Scene definition
##


@configclass
class FrankaLearnSceneCfg(InteractiveSceneCfg):
    """Configuration for a Franka scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = FRANKA_FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # manipuland -- referenced as SceneEntityCfg("object") by the observation,
    # reward and termination terms below.
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.055], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path="/home/michaela/robotics/isaacsim_assets/Assets/Isaac/6.2/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # end effector frame
    ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fr3_link0",
            debug_vis=False,
            # visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr3_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

    # cameras
    side_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/side_cam",
        update_period=0.01,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.05707, 2.10128, 0.3), rot=(0.06435, -0.70392, -0.70439, 0.06466), convention="opengl"),
    )
    
    front_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_cam",
        update_period=0.01,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(2.41549, 0.00102, 0.55849), rot=(0.43302, 0.4821, 0.55655, 0.51993 ), convention="opengl"),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["fr3_joint.*"], scale=0.5, use_default_offset=True
        )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["fr3_finger.*"],
        open_command_expr={"fr3_finger_.*": 0.04},
        close_command_expr={"fr3_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # Disabled for a vision-based policy: this reads the cube pose straight from the
        # physics state, which removes any need to infer it from the camera images.
        # Re-enable to train a state-based baseline.
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class FrontCamCfg(ObsGroup):
        """Front camera image. Kept in its own group: a group is concatenated internally,
        so 1-D state terms and (H, W, C) images cannot share one."""

        # permute=True -> (num_envs, C, H, W), which is the layout rsl_rl's CNNModel expects.
        image_front = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "permute": True},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SideCamCfg(ObsGroup):
        """Side camera image."""

        image_side = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("side_cam"), "data_type": "rgb", "permute": True},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    image_front: FrontCamCfg = FrontCamCfg()
    image_side: SideCamCfg = SideCamCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-4, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-4, "num_steps": 10000}
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="fr3_hand",  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )
##
# Environment configuration
##


@configclass
class FrankaLearnEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: FrankaLearnSceneCfg = FrankaLearnSceneCfg(num_envs=64, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation