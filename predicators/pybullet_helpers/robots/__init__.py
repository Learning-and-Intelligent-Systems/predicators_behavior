"""Handles the creation of robots."""
from typing import Dict, Type

from predicators.pybullet_helpers.geometry import Pose, Pose3D
from predicators.pybullet_helpers.robots.fetch import FetchPyBulletRobot
from predicators.pybullet_helpers.robots.panda import PandaPyBulletRobot
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.settings import CFG

# Note: these are static base poses which suffice for the current environments.
_ROBOT_TO_BASE_POSE: Dict[str, Pose] = {
    "fetch": Pose(position=(0.75, 0.7441, 0.0)),
    "panda": Pose(position=(0.8, 0.7441, 0.195)),
}

_ROBOT_TO_CLS: Dict[str, Type[SingleArmPyBulletRobot]] = {
    "fetch": FetchPyBulletRobot,
    "panda": PandaPyBulletRobot,
}


def create_single_arm_pybullet_robot(
        robot_name: str,
        physics_client_id: int,
        ee_home_pose: Pose3D = (1.35, 0.6, 0.7),
) -> SingleArmPyBulletRobot:
    """Create a single-arm PyBullet robot."""
    robot_to_ee_orn = CFG.pybullet_robot_ee_orns[CFG.env]
    if robot_name in _ROBOT_TO_CLS and robot_name in robot_to_ee_orn:
        assert robot_name in _ROBOT_TO_BASE_POSE, \
            f"Base pose not specified for robot {robot_name}."
        base_pose = _ROBOT_TO_BASE_POSE[robot_name]
        ee_orientation = robot_to_ee_orn[robot_name]
        cls = _ROBOT_TO_CLS[robot_name]
        return cls(ee_home_pose,
                   ee_orientation,
                   physics_client_id,
                   base_pose=base_pose)
    raise NotImplementedError(f"Unrecognized robot name: {robot_name}.")
