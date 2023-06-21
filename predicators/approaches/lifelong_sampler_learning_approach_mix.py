import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set
import copy
from collections import defaultdict
import time
import pickle as pkl

import numpy as np
import pybullet as p
import scipy
from scipy.special import logsumexp
import torch
from gym.spaces import Box

from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.explorers import create_explorer
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.settings import CFG
from predicators.structs import Array, Dataset, GroundAtom, GroundAtomsHoldQuery, \
    GroundAtomsHoldResponse, InteractionRequest, InteractionResult, \
    LowLevelTrajectory, NSRT, NSRTSampler, Object, ParameterizedOption, Predicate, \
    Query, State, Task, Type, Variable
from predicators.ml_models import DiffusionRegressor
from predicators import utils
from predicators.behavior_utils.behavior_utils import load_checkpoint_state, get_closest_point_on_aabb
from predicators.envs import get_or_create_env
from predicators.envs.behavior import BehaviorEnv
from multiprocessing import Pool
from predicators.teacher import Teacher, TeacherInteractionMonitor

from predicators.mpi_utils import proc_id, num_procs, mpi_sum, mpi_concatenate, mpi_concatenate_object, broadcast_object, mpi_min, mpi_max

def _featurize_state(state, ground_nsrt_objects):
    return state.vec(ground_nsrt_objects)

def _aux_labels(nsrt_name, x, a):
    env = get_or_create_env(CFG.env)
    # robot_obj = env.igibson_behavior_env.robots[0]
    # robot_pos = robot_obj.get_position()
    # robot_orn = robot_obj.get_orientation()

    # hand_obj = env.igibson_behavior_env.robots[0].parts["right_hand"]
    # hand_pos = hand_obj.get_position()
    # hand_orn = hand_obj.get_orientation()

    if nsrt_name.startswith("NavigateTo"):
        # single object: target
        target_pos = x[:3]
        target_orn = x[3:7]
        target_bbox = x[7:10]

        # Update: now second object: agent
        robot_pos = x[10:13]

        # 2 params: offset x,y
        offset_x = a[0]
        offset_y = a[1]

        end_robot_pos_x = offset_x + target_pos[0]
        end_robot_pos_y = offset_y + target_pos[1]
        end_robot_pos_z = robot_pos[2]
        end_robot_pos = np.r_[end_robot_pos_x, end_robot_pos_y, end_robot_pos_z]
        end_robot_yaw = np.arctan2(offset_y, offset_x) - np.pi

        # aux_labels = np.empty(12)
        aux_labels = np.empty(10)

        # Absolute positions and orientation
        # We need to know about this bc it is the intended action effect
        # aux_labels[0] = end_robot_pos_x
        # aux_labels[1] = end_robot_pos_y
        aux_labels[0] = np.sin(end_robot_yaw)
        aux_labels[1] = np.cos(end_robot_yaw)

        # Distance to the object (no bbox)
        # We need to know about this to satisfy reachability
        aux_labels[2] = np.linalg.norm(end_robot_pos - target_pos)

        # Everything below requires careful geometric thinking, which I haven't done

        # Nearest point on bbox
        # We need to know about this to avoid collision with target
        # I want the end robot pos relative to the object bbox, which would allow us to treat the bbox as axis-aligned
        # My understanding is that the bbox is just sizes, and the object location is the center,
        world_pos_in_target_frame, world_orn_in_target_frame = p.invertTransform(target_pos, target_orn)
        robot_pos_in_target_frame, _ = p.multiplyTransforms(world_pos_in_target_frame, world_orn_in_target_frame, end_robot_pos, np.zeros_like(target_orn))
        robot_pos_in_target_frame = np.array(robot_pos_in_target_frame)
        nearest_point_on_bbox = np.array(get_closest_point_on_aabb(robot_pos_in_target_frame, -target_bbox / 2, target_bbox / 2))
        aux_labels[3:6] = nearest_point_on_bbox

        # Distance to the nearest bbox point
        aux_labels[6] = np.linalg.norm(nearest_point_on_bbox - robot_pos_in_target_frame)

        # Relative reoriented coordinates
        aux_labels[7:10] = robot_pos_in_target_frame

    elif nsrt_name.startswith("Grasp"):
        # two objects: target, surface
        target_pos = x[:3]
        target_orn = x[3:7]
        target_bbox = x[7:10]

        surf_pos = x[10:13]
        surf_orn = x[13:17]
        surf_bbox = x[17:20]

        # 3 params: offset x,y,z
        offset_x = a[0]
        offset_y = a[1]
        offset_z = a[2]

        end_hand_pos_x = offset_x + target_pos[0]     
        end_hand_pos_y = offset_y + target_pos[1]     
        end_hand_pos_z = offset_z + target_pos[2]
        end_hand_pos = np.r_[end_hand_pos_x, end_hand_pos_y, end_hand_pos_z]

        hand_to_obj_unit_vector = a / np.linalg.norm(a)
        unit_z_vector = np.array([0.0, 0.0, -1.0])
        c_var = np.dot(unit_z_vector, hand_to_obj_unit_vector)
        if c_var not in [-1.0, 1.0]:
            v_var = np.cross(unit_z_vector, hand_to_obj_unit_vector)
            s_var = np.linalg.norm(v_var)
            v_x = np.array([
                [0, -v_var[2], v_var[1]],
                [v_var[2], 0, -v_var[0]],
                [-v_var[1], v_var[0], 0],
            ])
            R = (np.eye(3) + v_x + np.linalg.matrix_power(v_x, 2) * ((1 - c_var) /
                                                                     (s_var**2)))
            r = scipy.spatial.transform.Rotation.from_matrix(R)
            euler_angles = r.as_euler("xyz")
        else:
            if c_var == 1.0:
                euler_angles = np.zeros(3, dtype=float)
            else:
                euler_angles = np.array([0.0, np.pi, 0.0])
        end_hand_orn = p.getQuaternionFromEuler(euler_angles)
        
        # aux_labels = np.empty(37)
        aux_labels = np.empty(34)
        # Absolute positions and orientation
        # aux_labels[0] = end_hand_pos_x
        # aux_labels[1] = end_hand_pos_y
        # aux_labels[2] = end_hand_pos_z
        aux_labels[:3] = np.sin(euler_angles)
        aux_labels[3:6] = np.cos(euler_angles)

        # Distance to objects (no bbox)
        aux_labels[6] = np.linalg.norm(end_hand_pos - target_pos)
        aux_labels[7] = np.linalg.norm(end_hand_pos - surf_pos)

        # Nearest point on bboxes
        world_pos_in_target_frame, world_orn_in_target_frame = p.invertTransform(target_pos, target_orn)
        hand_pos_in_target_frame, hand_orn_in_target_frame = p.multiplyTransforms(world_pos_in_target_frame, world_orn_in_target_frame, end_hand_pos, end_hand_orn)
        hand_pos_in_target_frame = np.array(hand_pos_in_target_frame)
        hand_in_target_euler_angles = p.getEulerFromQuaternion(hand_orn_in_target_frame)
        nearest_point_on_target_bbox = np.array(get_closest_point_on_aabb(hand_pos_in_target_frame, -target_bbox / 2, target_bbox / 2))
        aux_labels[8:11] = nearest_point_on_target_bbox

        world_pos_in_surf_frame, world_orn_in_surf_frame = p.invertTransform(surf_pos, surf_orn)
        hand_pos_in_surf_frame, hand_orn_in_surf_frame = p.multiplyTransforms(world_pos_in_surf_frame, world_orn_in_surf_frame, end_hand_pos, end_hand_orn)
        hand_pos_in_surf_frame = np.array(hand_pos_in_surf_frame)
        hand_in_surf_euler_angles = p.getEulerFromQuaternion(hand_orn_in_surf_frame)
        nearest_point_on_surf_bbox = np.array(get_closest_point_on_aabb(hand_pos_in_surf_frame, -surf_bbox / 2, surf_bbox / 2))
        aux_labels[11:14] = nearest_point_on_surf_bbox

        # Distance to the nearest bbox point
        aux_labels[14] = np.linalg.norm(nearest_point_on_target_bbox - hand_pos_in_target_frame)
        aux_labels[15] = np.linalg.norm(nearest_point_on_surf_bbox - hand_pos_in_surf_frame)

        # Relative reoriented coordinates
        aux_labels[16:19] = hand_pos_in_target_frame
        aux_labels[19:22] = np.sin(hand_in_target_euler_angles)
        aux_labels[22:25] = np.cos(hand_in_target_euler_angles)
        aux_labels[25:28] = hand_pos_in_surf_frame
        aux_labels[28:31] = np.sin(hand_in_surf_euler_angles)
        aux_labels[31:34] = np.cos(hand_in_surf_euler_angles)

    elif nsrt_name.startswith("PlaceOnTop") or nsrt_name.startswith("PlaceInside") or nsrt_name.startswith("PlaceUnder"):
        # two objects: held, surface
        held_pos = x[:3]
        held_orn = x[3:7]
        held_bbox = x[7:10]

        surf_pos = x[10:13]
        surf_orn = x[13:17]
        surf_bbox = x[17:20]

        # Update: third object: agent
        # robot_pos [unused] = x[20:23]
        # robot_orn [unused] = x[23: 27]
        hand_pos = x[27:30]
        hand_orn = x[30: 34]

        # 3 params: offset x,y,z
        offset = copy.copy(a)
        offset[2] += 0.2    # not sure why, but this is done in motion_planner or option_model_fn
        end_hand_pos = offset + surf_pos
        end_hand_orn = p.getQuaternionFromEuler((0, np.pi * 7 / 6, 0))  # Obtained from motion_planner_fns (end_conf)

        og_world_pos_in_hand_frame, og_world_orn_in_hand_frame = p.invertTransform(hand_pos, hand_orn)
        held_pos_in_hand_frame, held_orn_in_hand_frame = p.multiplyTransforms(og_world_pos_in_hand_frame, og_world_orn_in_hand_frame, held_pos, held_orn)
        end_held_pos, end_held_orn = p.multiplyTransforms(end_hand_pos, end_hand_orn, held_pos_in_hand_frame, held_orn_in_hand_frame)
        end_held_pos = np.array(end_held_pos)
        # assert np.allclose(end_held_orn, held_orn), f"The orientation isn't supposed to change. Got {end_hand_orn} vs {held_orn}"

        # aux_labels = np.empty(38)
        aux_labels = np.empty(32)
        # Absolute positions (no orientation because that's fixed)
        # aux_labels[0:3] = end_hand_pos
        # aux_labels[3:6] = end_held_pos

        # Distance to the object (no bbox)
        aux_labels[0] = np.linalg.norm(end_hand_pos - surf_pos)
        aux_labels[1] = np.linalg.norm(end_held_pos - surf_pos)

        # Nearest point on bbox
        world_pos_in_surf_frame, world_orn_in_surf_frame = p.invertTransform(surf_pos, surf_orn)
        
        hand_pos_in_surf_frame, hand_orn_in_surf_frame = p.multiplyTransforms(world_pos_in_surf_frame, world_orn_in_surf_frame, end_hand_pos, end_hand_orn)
        hand_pos_in_surf_frame = np.array(hand_pos_in_surf_frame)
        hand_in_surf_euler_angles = p.getEulerFromQuaternion(hand_orn_in_surf_frame)
        hand_nearest_point_on_bbox = np.array(get_closest_point_on_aabb(hand_pos_in_surf_frame, -surf_bbox / 2, surf_bbox / 2))
        aux_labels[2:5] = hand_nearest_point_on_bbox

        held_pos_in_surf_frame, held_orn_in_surf_frame = p.multiplyTransforms(world_pos_in_surf_frame, world_orn_in_surf_frame, end_held_pos, end_held_orn)
        held_pos_in_surf_frame = np.array(held_pos_in_surf_frame)
        held_in_surf_euler_angles = p.getEulerFromQuaternion(held_orn_in_surf_frame)
        held_nearest_point_on_bbox = np.array(get_closest_point_on_aabb(held_pos_in_surf_frame, -surf_bbox / 2, surf_bbox / 2))
        aux_labels[5:8] = held_nearest_point_on_bbox

        world_pos_in_held_frame, world_orn_in_held_frame = p.invertTransform(end_held_pos, end_held_orn)
        surf_pos_in_held_frame, surf_orn_in_held_frame = p.multiplyTransforms(world_pos_in_held_frame, world_orn_in_held_frame, surf_pos, surf_orn)
        surf_pos_in_held_frame = np.array(surf_pos_in_held_frame)
        surf_nearest_point_on_bbox = np.array(get_closest_point_on_aabb(surf_pos_in_held_frame, -held_bbox / 2, held_bbox / 2))
        aux_labels[8:11] = surf_nearest_point_on_bbox

        # Distance to the nearest bbox point
        aux_labels[11] = np.linalg.norm(hand_nearest_point_on_bbox - hand_pos_in_surf_frame)
        aux_labels[12] = np.linalg.norm(held_nearest_point_on_bbox - held_pos_in_surf_frame)
        aux_labels[13] = np.linalg.norm(surf_nearest_point_on_bbox - surf_pos_in_held_frame)

        # Relative reoriented coordinates
        aux_labels[14:17] = hand_pos_in_surf_frame
        aux_labels[17:20] = np.sin(hand_in_surf_euler_angles)
        aux_labels[20:23] = np.cos(hand_in_surf_euler_angles)
        aux_labels[23:26] = held_pos_in_surf_frame
        aux_labels[26:29] = np.sin(held_in_surf_euler_angles)
        aux_labels[29:32] = np.cos(held_in_surf_euler_angles)

    elif nsrt_name.startswith("PlaceNextToOnTop"):
        # three objects: held, target, surface
        held_pos = x[:3]
        held_orn = x[3:7]
        held_bbox = x[7:10]

        target_pos = x[10:13]
        target_orn = x[13:17]
        target_bbox = x[17:20]

        surf_pos = x[20:23]
        surf_orn = x[23:27]
        surf_bbox = x[27:30]

        # Update: fourth object: agent
        # robot_pos [unused] = x[30:33]
        # robot_orn [unused] = x[33: 37]
        hand_pos = x[37:40]
        hand_orn = x[40: 44]

        # 3 params: offset x,y,z
        offset = copy.copy(a)
        offset[2] += 0.2    # not sure why, but this is done in motion_planner or option_model_fn
        end_hand_pos = offset + surf_pos
        end_hand_orn = p.getQuaternionFromEuler((0, np.pi * 7 / 6, 0))  # Obtained from motion_planner_fns (end_conf)

        og_world_pos_in_hand_frame, og_world_orn_in_hand_frame = p.invertTransform(hand_pos, hand_orn)
        held_pos_in_hand_frame, held_orn_in_hand_frame = p.multiplyTransforms(og_world_pos_in_hand_frame, og_world_orn_in_hand_frame, held_pos, held_orn)
        end_held_pos, end_held_orn = p.multiplyTransforms(end_hand_pos, end_hand_orn, held_pos_in_hand_frame, held_orn_in_hand_frame)
        end_held_pos = np.array(end_held_pos)
        # assert np.allclose(end_held_orn, held_orn), "The orientation isn't supposed to change"

        # aux_labels = np.empty(70)
        aux_labels = np.empty(64)
        # Absolute positions (no orientation because that's fixed)
        # aux_labels[0:3] = end_hand_pos
        # aux_labels[3:6] = end_held_pos


        # Distance to objects (no bbox)
        # Surf
        aux_labels[0] = np.linalg.norm(end_hand_pos - surf_pos)
        aux_labels[1] = np.linalg.norm(end_held_pos - surf_pos)
        # Target
        aux_labels[2] = np.linalg.norm(end_hand_pos - target_pos)
        aux_labels[3] = np.linalg.norm(end_held_pos - target_pos)

        # Nearest point on bbox
        # Surf
        world_pos_in_surf_frame, world_orn_in_surf_frame = p.invertTransform(surf_pos, surf_orn)
        
        hand_pos_in_surf_frame, hand_orn_in_surf_frame = p.multiplyTransforms(world_pos_in_surf_frame, world_orn_in_surf_frame, end_hand_pos, end_hand_orn)
        hand_pos_in_surf_frame = np.array(hand_pos_in_surf_frame)
        hand_in_surf_euler_angles = p.getEulerFromQuaternion(hand_orn_in_surf_frame)
        hand_nearest_point_on_surf_bbox = np.array(get_closest_point_on_aabb(hand_pos_in_surf_frame, -surf_bbox / 2, surf_bbox / 2))
        aux_labels[4:7] = hand_nearest_point_on_surf_bbox

        held_pos_in_surf_frame, held_orn_in_surf_frame = p.multiplyTransforms(world_pos_in_surf_frame, world_orn_in_surf_frame, end_held_pos, end_held_orn)
        held_pos_in_surf_frame = np.array(held_pos_in_surf_frame)
        held_in_surf_euler_angles = p.getEulerFromQuaternion(held_orn_in_surf_frame)
        held_nearest_point_on_surf_bbox = np.array(get_closest_point_on_aabb(held_pos_in_surf_frame, -surf_bbox / 2, surf_bbox / 2))
        aux_labels[7:10] = held_nearest_point_on_surf_bbox

        world_pos_in_held_frame, world_orn_in_held_frame = p.invertTransform(end_held_pos, end_held_orn)
        surf_pos_in_held_frame, surf_orn_in_held_frame = p.multiplyTransforms(world_pos_in_held_frame, world_orn_in_held_frame, surf_pos, surf_orn)
        surf_pos_in_held_frame = np.array(surf_pos_in_held_frame)
        surf_nearest_point_on_held_bbox = np.array(get_closest_point_on_aabb(surf_pos_in_held_frame, -held_bbox / 2, held_bbox / 2))
        aux_labels[10:13] = surf_nearest_point_on_held_bbox

        # Target
        world_pos_in_target_frame, world_orn_in_target_frame = p.invertTransform(target_pos, target_orn)
        
        hand_pos_in_target_frame, hand_orn_in_target_frame = p.multiplyTransforms(world_pos_in_target_frame, world_orn_in_target_frame, end_hand_pos, end_hand_orn)
        hand_pos_in_target_frame = np.array(hand_pos_in_target_frame)
        hand_in_target_euler_angles = p.getEulerFromQuaternion(hand_orn_in_target_frame)
        hand_nearest_point_on_target_bbox = np.array(get_closest_point_on_aabb(hand_pos_in_target_frame, -target_bbox / 2, target_bbox / 2))
        aux_labels[13:16] = hand_nearest_point_on_target_bbox

        held_pos_in_target_frame, held_orn_in_target_frame = p.multiplyTransforms(world_pos_in_target_frame, world_orn_in_target_frame, end_held_pos, end_held_orn)
        held_pos_in_target_frame = np.array(held_pos_in_target_frame)
        held_in_target_euler_angles = p.getEulerFromQuaternion(held_orn_in_target_frame)
        held_nearest_point_on_target_bbox = np.array(get_closest_point_on_aabb(held_pos_in_target_frame, -target_bbox / 2, target_bbox / 2))
        aux_labels[16:19] = held_nearest_point_on_target_bbox

        world_pos_in_held_frame, world_orn_in_held_frame = p.invertTransform(end_held_pos, end_held_orn)
        target_pos_in_held_frame, target_orn_in_held_frame = p.multiplyTransforms(world_pos_in_held_frame, world_orn_in_held_frame, target_pos, target_orn)
        target_pos_in_held_frame = np.array(target_pos_in_held_frame)
        target_nearest_point_on_held_bbox = np.array(get_closest_point_on_aabb(target_pos_in_held_frame, -held_bbox / 2, held_bbox / 2))
        aux_labels[19:22] = target_nearest_point_on_held_bbox

        # Distance to the nearest bbox point
        # Surf
        aux_labels[22] = np.linalg.norm(hand_nearest_point_on_surf_bbox - hand_pos_in_surf_frame)
        aux_labels[23] = np.linalg.norm(held_nearest_point_on_surf_bbox - held_pos_in_surf_frame)
        aux_labels[24] = np.linalg.norm(surf_nearest_point_on_held_bbox - surf_pos_in_held_frame)
        #Target
        aux_labels[25] = np.linalg.norm(hand_nearest_point_on_target_bbox - hand_pos_in_target_frame)
        aux_labels[26] = np.linalg.norm(held_nearest_point_on_target_bbox - held_pos_in_target_frame)
        aux_labels[27] = np.linalg.norm(target_nearest_point_on_held_bbox - target_pos_in_held_frame)

        # Relative reoriented coordinates
        # Surf
        aux_labels[28:31] = hand_pos_in_surf_frame
        aux_labels[31:34] = np.sin(hand_in_surf_euler_angles)
        aux_labels[34:37] = np.cos(hand_in_surf_euler_angles)
        aux_labels[37:40] = held_pos_in_surf_frame
        aux_labels[40:43] = np.sin(held_in_surf_euler_angles)
        aux_labels[43:46] = np.cos(held_in_surf_euler_angles)
        # Target
        aux_labels[46:49] = hand_pos_in_target_frame
        aux_labels[49:52] = np.sin(hand_in_target_euler_angles)
        aux_labels[52:55] = np.cos(hand_in_target_euler_angles)
        aux_labels[55:58] = held_pos_in_target_frame
        aux_labels[58:61] = np.sin(held_in_target_euler_angles)
        aux_labels[61:64] = np.cos(held_in_target_euler_angles)
    else:
        aux_labels = np.ones(1)
    return aux_labels


class LifelongSamplerLearningApproachMix(BilevelPlanningApproach):
    """A bilevel planning approach that uses hand-specified Operators
    but learns the samplers from interaction."""
    
    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks, task_planning_heuristic,
                         max_skeletons_optimized)
        self._online_learning_cycle = 0
        self._next_train_task = 0
        assert CFG.timeout == float('inf'), "We don't want to let methods time out in these experiments"
        self._save_dict = {}
        self._ebms = {}
        self._replay = {}
        self._option_needs_generic_sampler = {}
        self._generic_option_samplers = {}
        self._explorer_calls = 0

        if CFG.load_lifelong_checkpoint:
            logging.info("\nLoading lifelong checkpoint...")
            chkpt_config_str = f"behavior__lifelong_sampler_learning_mix__{CFG.seed}__checkpoint.pt"
            self._save_dict = torch.load(f"{CFG.results_dir}/{chkpt_config_str}")
            for sampler_name in self._save_dict:
                ebm = self._create_network()
                if "replay" in self._save_dict[sampler_name]:
                    # It's a specialized sampler
                    self._ebms[sampler_name] = ebm
                    self._online_learning_cycle = self._save_dict[sampler_name]["online_learning_cycle"] + 1
                    self._explorer_calls = self._save_dict[sampler_name]["explorer_calls"]
                    if proc_id() == 0:
                        self._replay[sampler_name] = self._save_dict[sampler_name]["replay"]
                    else:
                        self._save_dict[sampler_name]["replay"] = None
                        self._replay[sampler_name] = ([], [], [])
                else:
                    # It's a generic sampler
                    self._generic_option_samplers[sampler_name] = ebm
                    self._option_needs_generic_sampler[sampler_name] = True

                ebm._input_scale = self._save_dict[sampler_name]["input_scale"]
                ebm._input_shift = self._save_dict[sampler_name]["input_shift"]
                ebm._output_scale = self._save_dict[sampler_name]["output_scale"]
                ebm._output_shift = self._save_dict[sampler_name]["output_shift"]
                ebm._output_aux_scale = self._save_dict[sampler_name]["output_aux_scale"]
                ebm._output_aux_shift = self._save_dict[sampler_name]["output_aux_shift"]
                ebm.is_trained = self._save_dict[sampler_name]["is_trained"]
                ebm._x_cond_dim = self._save_dict[sampler_name]["x_cond_dim"]
                ebm._t_dim = self._save_dict[sampler_name]["t_dim"]
                ebm._y_dim = self._save_dict[sampler_name]["y_dim"]
                ebm._x_dim = self._save_dict[sampler_name]["x_dim"]
                ebm._y_aux_dim = self._save_dict[sampler_name]["y_aux_dim"]
                ebm._initialize_net()
                ebm.to(ebm._device)
                ebm.load_state_dict(self._save_dict[sampler_name]["model_state"])
                if proc_id() == 0:
                    ebm._create_optimizer()
                    ebm._optimizer.load_state_dict(self._save_dict[sampler_name]["optimizer_state"])
                else:
                    del self._save_dict[sampler_name]["optimizer_state"]
                    del self._save_dict[sampler_name]["online_learning_cycle"]
                    del self._save_dict[sampler_name]["explorer_calls"]

            tasks_so_far = (CFG.lifelong_burnin_period or CFG.interactive_num_requests_per_cycle) + (self._online_learning_cycle - 1) * CFG.interactive_num_requests_per_cycle
            tasks_so_far_local = int(np.ceil(tasks_so_far / num_procs()))
            self._next_train_task = tasks_so_far_local

    @classmethod
    def get_name(cls) -> str:
        return "lifelong_sampler_learning_mix"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_predicates(self) -> Set[Predicate]:
        if CFG.env == "behavior":
            env = get_or_create_env("behavior")
            self._initial_predicates, _ = utils.parse_config_excluded_predicates(env)
        return self._initial_predicates

    def _get_current_gt_nsrts(self) -> Set[NSRT]:
        if CFG.env == "behavior":
            env = get_or_create_env("behavior")
            preds = self._get_current_predicates()
            self._initial_options = env.options
            self._gt_nsrts = get_gt_nsrts(env.get_name(), preds,
                                       self._initial_options)
        return self._gt_nsrts

    def _get_current_nsrts(self) -> Set[NSRT]:
        # This call internally checks what are the current Behavior NSRTs,
        # creates any EBM samplers that may be missing, and sets self._nsrts
        self._initialize_ebm_samplers()
        return self._nsrts

    def _create_network(self) -> DiffusionRegressor:
        return DiffusionRegressor(
                    seed=CFG.seed,
                    hid_sizes=CFG.mlp_classifier_hid_sizes,
                    max_train_iters=CFG.sampler_mlp_classifier_max_itr,
                    timesteps=100,
                    learning_rate=CFG.learning_rate
               )

    def _initialize_ebm_samplers(self) -> None:
        # Behavior is written such that there's a different option for every type (e.g., NavigateTo-bucket vs NavigateTo-book),
        # and a separate NSRT for every instance (e.g., NavigateTo-bucket-0 vs NavigateTo-bucket-1) using the type-specific
        # option. To match up the 2D setting, "specialized" samplers are type-specific (so shared across NSRTs, but not across
        # options), and "generic" samplers are shared across options of the same underlying controller 
        sampler_names = set()
        gt_nsrts = self._get_current_gt_nsrts()

        for nsrt in gt_nsrts:
            # Generic option name
            generic_name = '-'.join(nsrt.name.split('-')[:-2])
            if generic_name not in self._option_needs_generic_sampler:
                self._option_needs_generic_sampler[generic_name] = True
                self._generic_option_samplers[generic_name] = self._create_network()


        for nsrt in gt_nsrts:
            # Specialized NSRT name
            sampler_names.add('-'.join(nsrt.name.split('-')[:-1]))

        for name in sampler_names:
            if name not in self._ebms:
                states_replay = []
                actions_replay = []
                aux_labels_replay = []
                self._replay[name] = (states_replay, actions_replay, aux_labels_replay)
                ebm = self._create_network()
                self._ebms[name] = ebm

        new_nsrts = []
        for nsrt in gt_nsrts:
            specialized_name = '-'.join(nsrt.name.split('-')[:-1])
            generic_name = '-'.join(nsrt.name.split('-')[:-2])
            if self._option_needs_generic_sampler[generic_name]:
                generic_ebm = self._generic_option_samplers[generic_name]
            else:
                generic_ebm = None
            assert CFG.ebm_aux_training == "geometry+"
            new_sampler = _LearnedSampler(nsrt.name, self._ebms[specialized_name], generic_ebm, nsrt.parameters, nsrt.option, nsrt.sampler).sampler
            new_nsrts.append(NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                                  nsrt.add_effects, nsrt.delete_effects,
                                  nsrt.ignore_effects, nsrt.option, 
                                  nsrt.option_vars, new_sampler))
        self._nsrts = new_nsrts

        # Ensure all keys are present
        specialized_names = mpi_concatenate_object(set(self._ebms.keys()))
        if proc_id() == 0:
            specialized_names = set.union(*specialized_names)
        specialized_names = broadcast_object(specialized_names)

        generic_names = mpi_concatenate_object(set(self._option_needs_generic_sampler.keys()))
        if proc_id() == 0:
            generic_names = set.union(*generic_names)
        generic_names = broadcast_object(generic_names)
        for name in specialized_names:
            if name not in self._ebms:
                self._ebms[name] = self._create_network()
                self._replay[name] = ([], [], [])
        for name in generic_names:
            if name not in self._option_needs_generic_sampler:
                self._option_needs_generic_sampler[name] = True
                self._generic_option_samplers[name] = self._create_network()


        # Sort dicts by name so order matches across processes
        self._ebms = dict(sorted(self._ebms.items()))
        self._replay = dict(sorted(self._replay.items()))
        self._option_needs_generic_sampler = dict(sorted(self._option_needs_generic_sampler.items()))
        self._generic_option_samplers = dict(sorted(self._generic_option_samplers.items()))

    def load(self, online_learning_cycle: Optional[int]) -> None:
        raise 'NotImplementedError'
        # TODO: I should probably implement checkpointing here

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # TODO: I'm not sure whether it makes more sense to treat the initial
        # data collection like all the others or have it be "demos"
        assert len(dataset.trajectories) == 0

    def get_interaction_requests(self):
        requests = []
        if CFG.lifelong_burnin_period is None or self._online_learning_cycle > 0:
            num_tasks = CFG.interactive_num_requests_per_cycle
        else:
            num_tasks = CFG.lifelong_burnin_period
        num_tasks_local = int(np.ceil(num_tasks / num_procs()))
        num_tasks = num_tasks_local * num_procs()   # in case num_tasks isn't divisible by num_procs
        first_train_task = self._next_train_task
        self._next_train_task += num_tasks_local
        task_list_this_proc = range(first_train_task, self._next_train_task) #range(first_train_task + proc_id(), self._next_train_task, num_procs())
        logging.info(f"{proc_id()}: {task_list_this_proc}")

        metrics: Metrics = defaultdict(float)
        explorer_metrics: Metrics = defaultdict(float)
        # Get the next tasks in the sequence
        total_time = 0
        for train_task_idx in task_list_this_proc:
            # Create the explorer in the loop to make sure the task is properly set.
            # This is probably necessary to set the correct nsrts/preds

            # If we're running on BEHAVIOR, we need to make sure we are
            # currently in the correct environment/task so that we can
            # get the correct predicates and NSRTs to run our planner.
            if CFG.env == "behavior":  # pragma: no cover
                env = get_or_create_env("behavior")
                assert isinstance(env, BehaviorEnv)
                # Note: this part of the code assumes that all tasks that will
                # be explored in a single call are of the same type. Otherwise,
                # We'd need to create a different explorer for each. Since we
                # are not doing that, failing to ensure that all tasks are of 
                # the same type will fail "ungracefully"
                task = self._train_tasks[train_task_idx]
                if not task.init.allclose(
                        env.current_ig_state_to_state(
                            save_state=False,
                            use_test_scene=False)):
                    load_checkpoint_state(task.init, env, reset=True)

            nsrts = self._get_current_nsrts()
            preds = self._get_current_predicates()

            logging.info(f"({proc_id()} Creating explorer for task {train_task_idx}")
            explorer = create_explorer(
                "partial_planning",
                preds,
                self._initial_options,
                self._types,
                self._action_space,
                self._train_tasks,
                nsrts,
                self._option_model)
            explorer._num_calls = self._explorer_calls
            logging.info(f"({proc_id()} Created explorer for task {train_task_idx}")
            explorer.initialize_metrics(explorer_metrics)

            query_policy = self._create_none_query_policy()
            explore_start = time.perf_counter()
            logging.info(f"{(proc_id())} Creating exploration strategy for task {train_task_idx}")
            option_plan, termination_fn, skeleton, traj = explorer.get_exploration_strategy(
                train_task_idx, CFG.timeout)
            logging.info(f"{(proc_id())} Created exploration strategy for task {train_task_idx}")
            explore_time = time.perf_counter() - explore_start
            logging.info(f"{(proc_id())} Creating interaction request for task {train_task_idx}")
            requests.append(InteractionRequest(train_task_idx, option_plan, query_policy, termination_fn, skeleton, traj))
            logging.info(f"{(proc_id())} Created interaction request for task {train_task_idx}")
            total_time += explore_time
            explorer_metrics = explorer.metrics
            self._explorer_calls = explorer._num_calls
        
        logging.info(f"{(proc_id())} Aggregating metrics for task {train_task_idx}")
        num_unsolved = int(mpi_sum(explorer_metrics["num_unsolved"]))
        num_solved = int(mpi_sum(explorer_metrics["num_solved"]))
        num_total = num_unsolved + num_solved
        avg_time = mpi_sum(total_time) / num_tasks
        assert num_total == num_tasks, f"Tasks don't match: {num_total} (={num_solved}+{num_unsolved}), {num_tasks}"
        metrics["num_solved"] = num_solved
        metrics["num_unsolved"] = num_unsolved
        metrics["num_total"] = num_tasks
        metrics["avg_time"] = avg_time
        metrics["min_num_samples"] = mpi_min(explorer_metrics["min_num_samples"])
        metrics["max_num_samples"] = mpi_max(explorer_metrics["max_num_samples"])
        metrics["min_num_skeletons_optimized"] = mpi_min(explorer_metrics["min_num_skeletons_optimized"])
        metrics["max_num_skeletons_optimized"] = mpi_max(explorer_metrics["max_num_skeletons_optimized"])
        metrics["num_solve_failures"] = mpi_sum(num_unsolved)

        for metric_name in [
                "num_samples", "num_skeletons_optimized", "num_nodes_expanded",
                "num_nodes_created", "num_nsrts", "num_preds", "plan_length",
                "num_failures_discovered"
        ]:
            total = mpi_sum(explorer_metrics[f"total_{metric_name}"])
            metrics[f"avg_{metric_name}"] = (
                total / num_solved if num_solved > 0 else float("inf"))
        total = mpi_sum(explorer_metrics["total_num_samples_failed"])
        metrics["avg_num_samples_failed"] = total / num_unsolved if num_unsolved > 0 else float("inf")
        logging.info(f"{(proc_id())} Aggregated metrics for task {train_task_idx}")

        if len(CFG.behavior_task_list) != 1:
            env = get_or_create_env(CFG.env)
            subenv_name = CFG.behavior_task_list[env.task_list_indices[env.task_num]]

            num_unsolved_env = int(mpi_sum(explorer_metrics[f"env_{subenv_name}_num_unsolved"]))
            num_solved_env = int(mpi_sum(explorer_metrics[f"env_{subenv_name}_num_solved"]))
            total_tasks_env = num_unsolved_env + num_solved_env

            metrics[f"{subenv_name}_num_solved"] = num_solved_env
            metrics[f"{subenv_name}_num_total"] = total_tasks_env
            metrics[f"{subenv_name}_min_num_samples"] = mpi_min(explorer_metrics[f"env_{subenv_name}_min_num_samples"])
            metrics[f"{subenv_name}_max_num_samples"] = mpi_max(explorer_metrics[f"env_{subenv_name}_max_num_samples"])
            metrics[f"{subenv_name}_min_num_skeletons_optimized"] = mpi_min(explorer_metrics[f"env_{subenv_name}_min_num_skeletons_optimized"])
            metrics[f"{subenv_name}_max_num_skeletons_optimized"] = mpi_max(explorer_metrics[f"env_{subenv_name}_max_num_skeletons_optimized"])
            metrics[f"{subenv_name}_num_solve_failures"] = mpi_sum(num_unsolved_env)

            for metric_name in [
                    "num_samples", "num_skeletons_optimized", "num_nodes_expanded",
                    "num_nodes_created", "num_nsrts", "num_preds", "plan_length",
                    "num_failures_discovered"
            ]:
                total = mpi_sum(explorer_metrics[f"env_{subenv_name}_total_{metric_name}"])
                metrics[f"{subenv_name}_avg_{metric_name}"] = (
                    total / num_solved_env if num_solved_env > 0 else float("inf"))
            total = mpi_sum(explorer_metrics[f"env_{subenv_name}_total_num_samples_failed"])
            metrics[f"{subenv_name}_avg_num_samples_failed"] = total / num_unsolved_env if num_unsolved_env > 0 else float("inf")


        if proc_id() == 0:
            logging.info(f"Tasks solved: {num_solved} / {num_tasks}")
            outfile = (f"{CFG.results_dir}/{utils.get_config_path_str()}__"
                   f"{self._online_learning_cycle}.pkl")
            
            cfg_copy = copy.copy(CFG)
            cfg_copy.pybullet_robot_ee_orns = None
            cfg_copy.get_arg_specific_settings = None
            outdata = {
                "config": cfg_copy,
                "results": metrics.copy(),
                "git_commit_hash": utils.get_git_commit_hash()
            }
            # Dump the CFG, results, and git commit hash to a pickle file.
            with open(outfile, "wb") as f:
                pkl.dump(outdata, f)

            logging.info(f"Exploration results: {metrics}")
            logging.info(f"Average time per task: {avg_time:.5f} seconds")
            logging.info(f"Wrote out test results to {outfile}")

        return requests

    def _create_none_query_policy(self) -> Callable[[State], Optional[Query]]:
        def _query_policy(s: State) -> Optional[GroundAtomsHoldQuery]:
            return None
        return _query_policy
    
    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        traj_list: List[LowLevelTrajectory] = []
        annotations_list: List[Any] = []
        skeleton_list: List[Any] = []

        for result in results:
            skeleton = result.skeleton
            task = self._train_tasks[result.train_task_idx]
            state_annotations = [True for _ in result.states[1:]]
            ######### State annotations
            # state_annotations = []
            # init_atoms = utils.abstract(result.states[0], self._initial_predicates)
            # atoms_sequence = [init_atoms]
            # for ground_nsrt in skeleton:
            #     atoms_sequence.append(utils.apply_operator(ground_nsrt, atoms_sequence[-1]))
            # necessary_atoms_sequence = utils.compute_necessary_atoms_seq(skeleton, atoms_sequence, task.goal)

            # for state, expected_atoms, ground_nsrt in zip(result.states[1:], necessary_atoms_sequence[1:], result.skeleton):
            #     state_annotations.append(all(a.holds(state) for a in expected_atoms))
            #     if not state_annotations[-1]:
            #         logging.info("Somehow a state annotation is False")
            #         logging.info(f"{state}")
            #         logging.info(f"expected_atoms")
            #         logging.info(f"{[a.holds(state) for a in expected_Atoms]}")
            #         exit()
            ######### End state annotations
            traj = LowLevelTrajectory(result.states, result.options)
            traj_list.append(traj)
            annotations_list.append(state_annotations)
            skeleton_list.append(result.skeleton)
        self._update_samplers(traj_list, annotations_list, skeleton_list)
        self._online_learning_cycle += 1

    def _update_samplers(self, trajectories: List[LowLevelTrajectory], annotations_list: List[Any], skeletons: List[Any]) -> None:
        logging.info("\nUpdating the samplers...")
        if self._online_learning_cycle == 0:
            self._initialize_ebm_samplers()
        logging.info("Featurizing the samples...")

        logging.info(f"Specialized: {list(self._ebms.keys())}")
        logging.info(f"Generic: {list(self._option_needs_generic_sampler.keys())}")

        option_generic_states = {name: [] for name, needs in self._option_needs_generic_sampler.items() if needs}
        option_generic_actions = {name: [] for name, needs in self._option_needs_generic_sampler.items() if needs}
        option_generic_aux_labels = {name: [] for name, needs in self._option_needs_generic_sampler.items() if needs}
        option_generic_replay_states = {name: [] for name, needs in self._option_needs_generic_sampler.items() if needs}
        option_generic_replay_actions = {name: [] for name, needs in self._option_needs_generic_sampler.items() if needs}
        option_generic_replay_aux_labels = {name: [] for name, needs in self._option_needs_generic_sampler.items() if needs}

        for specialized_sampler_name in self._ebms.keys():
            ebm = self._ebms[specialized_sampler_name]
            replay = self._replay[specialized_sampler_name]
            # Build replay buffer for generic option sampler training in the other loop
            # Note: it's important to do this at the top and not the bottom of this loop bc we later add the _new_ data into the replay buffer
            generic_sampler_name = '-'.join(specialized_sampler_name.split('-')[:-1])
            if self._option_needs_generic_sampler[generic_sampler_name]:
                option_generic_replay_states[generic_sampler_name].extend(replay[0])
                option_generic_replay_actions[generic_sampler_name].extend(replay[1])
                option_generic_replay_aux_labels[generic_sampler_name].extend(replay[2])

            states = []
            actions = []
            aux_labels = []
            for traj, annotations, skeleton in zip(trajectories, annotations_list, skeletons):
                for state, option, annotation, ground_nsrt in zip(traj.states[:-1], traj.options, annotations, skeleton):
                    # Get this NSRT's positive (successful) data only
                    if annotation > 0:
                        specialized_name = '-'.join(ground_nsrt.name.split('-')[:-1])
                        generic_name = '-'.join(ground_nsrt.name.split('-')[:-2])
                        if specialized_sampler_name == specialized_name:
                            x = _featurize_state(state, ground_nsrt.objects)
                            a = option.params
                            aux = _aux_labels(specialized_name, x, a)
                            states.append(x)
                            actions.append(a)
                            aux_labels.append(aux)
                            if self._option_needs_generic_sampler[generic_name]:
                                # Get the generic option's new data
                                option_generic_states[generic_name].append(x)
                                option_generic_actions[generic_name].append(a)
                                option_generic_aux_labels[generic_name].append(aux)

            states_arr = np.array(states)
            actions_arr = np.array(actions)
            aux_labels_arr = np.array(aux_labels)
            states_arr = mpi_concatenate(states_arr)
            actions_arr = mpi_concatenate(actions_arr)
            aux_labels_arr = mpi_concatenate(aux_labels_arr)
            if proc_id() == 0:
                logging.info(f"{specialized_sampler_name}: {states_arr.shape[0]} samples, {states_arr.shape[1]} features, {actions_arr.shape[1]} outputs, {aux_labels_arr.shape[1]} auxs")
                if states_arr.shape[0] > 0:
                    start_time = time.perf_counter()
                    if not ebm.is_trained:
                        ebm.fit(states_arr, actions_arr, aux_labels_arr)
                    else:
                        states_replay = np.array(replay[0])
                        actions_replay = np.array(replay[1])
                        aux_labels_replay = np.array(replay[2])

                        if CFG.lifelong_method == "distill" or CFG.lifelong_method == "2-distill":
                            # First copy: train model just on new data
                            ebm_new = copy.deepcopy(ebm)
                            ebm_new.fit(states_arr, actions_arr, aux_labels_arr)

                            # Second copy: previous version to distill into updated model
                            ebm_old = copy.deepcopy(ebm)

                            # Distill new and old models into updated model
                            ebm_new_data = (ebm_new, (states_arr, actions_arr, aux_labels_arr))
                            ebm_old_data = (ebm_old, (states_replay, actions_replay, aux_labels_replay))
                            ebm.distill(ebm_old_data, ebm_new_data)
                        elif CFG.lifelong_method == "retrain":
                            # Instead, try re-training the model as a performance upper bound
                            states_full = np.r_[states_arr, states_replay]
                            actions_full = np.r_[actions_arr, actions_replay]
                            aux_labels_full = np.r_[aux_labels_arr, aux_labels_replay]
                            ebm.fit(states_full, actions_full, aux_labels_full)
                        elif CFG.lifelong_method == "retrain-scratch":
                            ebm._linears = torch.nn.ModuleList()
                            ebm._optimizer = None
                            ebm.is_trained = False
                            states_full = np.r_[states_arr, states_replay]
                            actions_full = np.r_[actions_arr, actions_replay]
                            aux_labels_full = np.r_[aux_labels_arr, aux_labels_replay]
                            ebm.fit(states_full, actions_full, aux_labels_full)
                        elif CFG.lifelong_method == 'finetune':
                            ebm.fit(states_arr, actions_arr, aux_labels_arr)
                        elif CFG.lifelong_method == "retrain-balanced":
                            new_data = (states_arr, actions_arr, aux_labels_arr)
                            old_data = (states_replay, actions_replay, aux_labels_replay)
                            ebm.fit_balanced(old_data, new_data)
                        else:
                            raise NotImplementedError(f"Unknown lifelong method {CFG.lifelong_method}")            
                    end_time = time.perf_counter()
                    logging.info(f"Training time: {(end_time - start_time):.5f} seconds")
                    replay[0].extend(list(states_arr))
                    replay[1].extend(list(actions_arr))
                    replay[2].extend(list(aux_labels_arr))
                    self._save_dict[specialized_sampler_name] = {
                        "optimizer_state": ebm._optimizer.state_dict(),
                        "model_state": ebm.state_dict(),
                        "input_scale": ebm._input_scale,
                        "input_shift": ebm._input_shift,
                        "output_scale": ebm._output_scale,
                        "output_shift": ebm._output_shift,
                        "output_aux_scale": ebm._output_aux_scale,
                        "output_aux_shift": ebm._output_aux_shift,
                        "is_trained": ebm.is_trained,
                        "x_cond_dim": ebm._x_cond_dim,
                        "t_dim": ebm._t_dim,
                        "y_dim": ebm._y_dim,
                        "x_dim": ebm._x_dim,
                        "y_aux_dim": ebm._y_aux_dim,
                        "replay": replay,
                        "online_learning_cycle": self._online_learning_cycle,
                        "explorer_calls": self._explorer_calls,
                    }   
        
        for generic_sampler_name in self._option_needs_generic_sampler:
            if self._option_needs_generic_sampler[generic_sampler_name]:
                ebm = self._generic_option_samplers[generic_sampler_name]
                states_arr = np.array(option_generic_states[generic_sampler_name])
                actions_arr = np.array(option_generic_actions[generic_sampler_name])
                aux_labels_arr = np.array(option_generic_aux_labels[generic_sampler_name])
                states_arr = mpi_concatenate(states_arr)
                actions_arr = mpi_concatenate(actions_arr)
                aux_labels_arr = mpi_concatenate(aux_labels_arr)
                if proc_id() == 0:
                    logging.info(f"{generic_sampler_name}: {states_arr.shape[0]} samples, {states_arr.shape[1]} features, {actions_arr.shape[1]} outputs, {aux_labels_arr.shape[1]} auxs")
                    if states_arr.shape[0] > 0:
                        start_time = time.perf_counter()
                        if not ebm.is_trained:
                            ebm.fit(states_arr, actions_arr, aux_labels_arr)
                        else:
                            states_replay = np.array(option_generic_replay_states[generic_sampler_name])
                            actions_replay = np.array(option_generic_replay_actions[generic_sampler_name])
                            aux_labels_replay = np.array(option_generic_replay_aux_labels[generic_sampler_name])
                            if CFG.lifelong_method == "distill" or CFG.lifelong_method == "2-distill":
                                # First copy: train model just on new data
                                ebm_new = copy.deepcopy(ebm)
                                ebm_new.fit(states_arr, actions_arr, aux_labels_arr)

                                # Second copy: previous version to distill into updated model
                                ebm_old = copy.deepcopy(ebm)

                                # Distill new and old models into updated model
                                ebm_new_data = (ebm_new, (states_arr, actions_arr, aux_labels_arr))
                                ebm_old_data = (ebm_old, (states_replay, actions_replay, aux_labels_replay))
                                ebm.distill(ebm_old_data, ebm_new_data)
                            elif CFG.lifelong_method == "retrain":
                                # Instead, try re-training the model as a performance upper bound
                                states_full = np.r_[states_arr, states_replay]
                                actions_full = np.r_[actions_arr, actions_replay]
                                aux_labels_full = np.r_[aux_labels_arr, aux_labels_replay]
                                ebm.fit(states_full, actions_full, aux_labels_full)
                            elif CFG.lifelong_method == "retrain-scratch":
                                ebm._linears = torch.nn.ModuleList()
                                ebm._optimizer = None
                                ebm.is_trained = False
                                states_full = np.r_[states_arr, states_replay]
                                actions_full = np.r_[actions_arr, actions_replay]
                                aux_labels_full = np.r_[aux_labels_arr, aux_labels_replay]
                                ebm.fit(states_full, actions_full, aux_labels_full)
                            elif CFG.lifelong_method == 'finetune':
                                ebm.fit(states_arr, actions_arr, aux_labels_arr)
                            elif CFG.lifelong_method == "retrain-balanced":
                                new_data = (states_arr, actions_arr, aux_labels_arr)
                                old_data = (states_replay, actions_replay, aux_labels_replay)
                                ebm.fit_balanced(old_data, new_data)
                            else:
                                raise NotImplementedError(f"Unknown lifelong method {CFG.lifelong_method}")            
                        end_time = time.perf_counter()
                        logging.info(f"Training time: {(end_time - start_time):.5f} seconds")
                        self._save_dict[generic_sampler_name] = {
                            "optimizer_state": ebm._optimizer.state_dict(),
                            "model_state": ebm.state_dict(),
                            "input_scale": ebm._input_scale,
                            "input_shift": ebm._input_shift,
                            "output_scale": ebm._output_scale,
                            "output_shift": ebm._output_shift,
                            "output_aux_scale": ebm._output_aux_scale,
                            "output_aux_shift": ebm._output_aux_shift,
                            "is_trained": ebm.is_trained,
                            "x_cond_dim": ebm._x_cond_dim,
                            "t_dim": ebm._t_dim,
                            "y_dim": ebm._y_dim,
                            "x_dim": ebm._x_dim,
                            "y_aux_dim": ebm._y_aux_dim,
                            "online_learning_cycle": self._online_learning_cycle,
                            "explorer_calls": self._explorer_calls,
                        }   


        logging.info(f"Out of training loop ({proc_id()})")
        if proc_id() == 0:
            chkpt_config_str = f"behavior__lifelong_sampler_learning_mix__{CFG.seed}__checkpoint.pt"
            torch.save(self._save_dict, f"{CFG.results_dir}/{chkpt_config_str}")
        self._save_dict = broadcast_object(self._save_dict, root=0)
        logging.info(f"Broadcasted save_dict ({proc_id()})")
        if proc_id() != 0:
            for specialized_sampler_name in self._ebms.keys():
                ebm = self._ebms[specialized_sampler_name]
                if specialized_sampler_name in self._save_dict:
                    logging.info(f"Loading keys into process {proc_id()}: {self._save_dict[specialized_sampler_name]['model_state'].keys()}")
                    del self._save_dict[specialized_sampler_name]["replay"]
                    del self._save_dict[specialized_sampler_name]["optimizer_state"]
                    del self._save_dict[specialized_sampler_name]["online_learning_cycle"]
                    del self._save_dict[specialized_sampler_name]["explorer_calls"]
                    ebm._input_scale = self._save_dict[specialized_sampler_name]["input_scale"]
                    ebm._input_shift = self._save_dict[specialized_sampler_name]["input_shift"]
                    ebm._output_scale = self._save_dict[specialized_sampler_name]["output_scale"]
                    ebm._output_shift = self._save_dict[specialized_sampler_name]["output_shift"]
                    ebm._output_aux_scale = self._save_dict[specialized_sampler_name]["output_aux_scale"]
                    ebm._output_aux_shift = self._save_dict[specialized_sampler_name]["output_aux_shift"]
                    ebm.is_trained = self._save_dict[specialized_sampler_name]["is_trained"]
                    ebm._x_cond_dim = self._save_dict[specialized_sampler_name]["x_cond_dim"]
                    ebm._t_dim = self._save_dict[specialized_sampler_name]["t_dim"]
                    ebm._y_dim = self._save_dict[specialized_sampler_name]["y_dim"]
                    ebm._x_dim = self._save_dict[specialized_sampler_name]["x_dim"]
                    ebm._y_aux_dim = self._save_dict[specialized_sampler_name]["y_aux_dim"]

                    ebm._initialize_net()
                    ebm.to(ebm._device)
                    ebm.load_state_dict(self._save_dict[specialized_sampler_name]["model_state"])

            for generic_sampler_name in self._option_needs_generic_sampler.keys():
                if self._option_needs_generic_sampler[generic_sampler_name]:
                    ebm = self._generic_option_samplers[generic_sampler_name]
                    if generic_sampler_name in self._save_dict:
                        logging.info(f"Loading keys into process {proc_id()}: {self._save_dict[generic_sampler_name]['model_state'].keys()}")
                        del self._save_dict[generic_sampler_name]["optimizer_state"]
                        del self._save_dict[generic_sampler_name]["online_learning_cycle"]
                        del self._save_dict[generic_sampler_name]["explorer_calls"]
                        ebm._input_scale = self._save_dict[generic_sampler_name]["input_scale"]
                        ebm._input_shift = self._save_dict[generic_sampler_name]["input_shift"]
                        ebm._output_scale = self._save_dict[generic_sampler_name]["output_scale"]
                        ebm._output_shift = self._save_dict[generic_sampler_name]["output_shift"]
                        ebm._output_aux_scale = self._save_dict[generic_sampler_name]["output_aux_scale"]
                        ebm._output_aux_shift = self._save_dict[generic_sampler_name]["output_aux_shift"]
                        ebm.is_trained = self._save_dict[generic_sampler_name]["is_trained"]
                        ebm._x_cond_dim = self._save_dict[generic_sampler_name]["x_cond_dim"]
                        ebm._t_dim = self._save_dict[generic_sampler_name]["t_dim"]
                        ebm._y_dim = self._save_dict[generic_sampler_name]["y_dim"]
                        ebm._x_dim = self._save_dict[generic_sampler_name]["x_dim"]
                        ebm._y_aux_dim = self._save_dict[generic_sampler_name]["y_aux_dim"]

                        ebm._initialize_net()
                        ebm.to(ebm._device)
                        ebm.load_state_dict(self._save_dict[generic_sampler_name]["model_state"])


@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler:
    """A convenience class for holding the models underlying a learned
    sampler."""
    _name: str
    _ebm: DiffusionRegressor
    _generic_ebm: DiffusionRegressor
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption
    _original_sampler: NSRTSampler

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object],
                return_failed_samples: Optional[bool] = False) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        assert return_failed_samples == False
        x_lst: List[Any] = []  
        sub = dict(zip(self._variables, objects))
        for var in self._variables:
            x_lst.extend(state[sub[var]])
        x = np.array(x_lst)
        assert (x == state.vec(objects)).all()

        if not self._ebm.is_trained:
            # If I haven't trained the specialized model, uniformly choose between original and generic
            if self._generic_ebm is not None and self._generic_ebm.is_trained:
                # chosen_sampler_idx = rng.choice([1, 2])
                chosen_sampler_idx = 1  # Use default samplers (so, generate demos)
                if chosen_sampler_idx == 2:
                    return np.array(self._generic_ebm.predict_sample(x, rng),
                                     dtype=self._param_option.params_space.dtype), 1
                # return self._original_sampler(state, goal, rng, objects, max_internal_samples=1)
            return self._original_sampler(state, goal, rng, objects)
        
        if self._generic_ebm is None:
            num_samplers = 2
        else:
            num_samplers = 3

        ebm_a = np.array(self._ebm.predict_sample(x, rng),
                         dtype=self._param_option.params_space.dtype)
        aux = _aux_labels(self._name, x, ebm_a)
        ebm_square_err = self._ebm.aux_square_error(x[None], ebm_a[None], aux[None])
        ebm_err = np.sqrt(np.sum(ebm_square_err))
        if num_samplers == 3:
            generic_ebm_a = np.array(self._generic_ebm.predict_sample(x, rng),
                                     dtype=self._param_option.params_space.dtype)
            aux = _aux_labels(self._name, x, generic_ebm_a)
            generic_ebm_square_err = self._generic_ebm.aux_square_error(x[None], generic_ebm_a[None], aux[None])
            generic_ebm_err = np.sqrt(np.sum(generic_ebm_square_err))
            # if np.isinf(ebm_err) and np.isinf(generic_ebm_err):
            #     original_err = 1
            # else:
            #     original_err = 9 / (1 / (ebm_err+1e-6) + 1 / (generic_ebm_err+1e-6))      # This is a value such that the original choice probability is fixed at 0.1
            original_err = np.inf
            if np.isinf(ebm_err) and np.isinf(generic_ebm_err):
                ebm_err = 1
                generic_ebm_err = 1
            choice_probabilities = 1 / np.array([ebm_err + 1e-6, original_err, generic_ebm_err + 1e-6])
        else:
            # if np.isinf(ebm_err):
            #     original_err = 1
            # else:
            #     original_err = 9 * (ebm_err+1e-6)   # This is a value such that the original choice probability is fixed at 0.1
            # choice_probabilities = 1 / np.array([ebm_err + 1e-6, original_err])
            choice_probabilities = np.array([1, 0])
        choice_probabilities /= choice_probabilities.sum()

        # logging.info(f"{ebm_err}, {generic_ebm_err}, {original_err}, {choice_probabilities}")

        chosen_sampler_idx = rng.choice(num_samplers, replace=False, p=choice_probabilities)
        if chosen_sampler_idx == 0:
            if CFG.ebm_aux_training is not None:# and CFG.ebm_aux_training.startswith('geometry'):
                params = ebm_a
            else:
                params = np.array(self._ebm.predict_sample(x, rng),
                                  dtype=self._param_option.params_space.dtype)
        elif chosen_sampler_idx == 1:
            params = self._original_sampler(state, goal, rng, objects, max_internal_samples=1)[0]
        else:
            if CFG.ebm_aux_training is not None:# and CFG.ebm_aux_training.startswith('geometry'):
                params = generic_ebm_a
            else:
                params = np.array(self._generic_ebm.predict_sample(x, rng),
                                  dtype=self._param_option.params_space.dtype)
        return params, 1
