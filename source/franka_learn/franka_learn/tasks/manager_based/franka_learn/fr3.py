import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

FRANKA_FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path= "/home/michaela/robotics/isaacsim_assets/Assets/Isaac/6.2/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaFR3/fr3/fr3.usda",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": -0.569,
            "fr3_joint3": 0.0,
            "fr3_joint4": -2.810,
            "fr3_joint5": 0.0,
            "fr3_joint6": 3.037,
            "fr3_joint7": 0.741,
            "fr3_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            joint_effort_limit=87.0,
            stiffness=80.0,
            damping=4.0,
            armature=1e-3,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            joint_effort_limit=12.0,
            stiffness=80.0,
            damping=4.0,
            armature=1e-3,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            joint_effort_limit=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)