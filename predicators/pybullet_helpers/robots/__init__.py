"""Handles the creation of robots."""
from typing import Dict

from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots.fetch import FetchPyBulletRobot
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot

# Note: these are static base poses which suffice for the current environments.
_ROBOT_TO_BASE_POSE: Dict[str, Pose] = {
    "fetch": Pose(position=(0.75, 0.7441, 0.0)),
}


def create_single_arm_pybullet_robot(
        robot_name: str, ee_home_pose: Pose3D, ee_orientation: Quaternion,
        physics_client_id: int) -> SingleArmPyBulletRobot:
    """Create a single-arm PyBullet robot."""
    if robot_name == "fetch":
        assert robot_name in _ROBOT_TO_BASE_POSE, \
            f"Base pose not specified for robot {robot_name}."
        base_pose = _ROBOT_TO_BASE_POSE[robot_name]
        return FetchPyBulletRobot(ee_home_pose,
                                  ee_orientation,
                                  physics_client_id,
                                  base_pose=base_pose)
    raise NotImplementedError(f"Unrecognized robot name: {robot_name}.")