import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
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
from predicators.ml_models import NeuralGaussianRegressor, MLPBinaryClassifier
from predicators import utils
from predicators.behavior_utils.behavior_utils import load_checkpoint_state, get_closest_point_on_aabb
from predicators.envs import get_or_create_env
from predicators.envs.behavior import BehaviorEnv
from multiprocessing import Pool
from predicators.teacher import Teacher, TeacherInteractionMonitor

from predicators.mpi_utils import proc_id, num_procs, mpi_sum, mpi_concatenate, mpi_concatenate_object, broadcast_object, mpi_min, mpi_max
from predicators.nsrt_learning.nsrt_learning_main import get_ground_atoms_dataset, _learn_pnad_options_with_learner
from predicators.nsrt_learning.segmentation import _segment_with_oracle
from predicators.nsrt_learning.strips_learning.oracle_learner import OracleSTRIPSLearner
from predicators.nsrt_learning.option_learning import _OracleOptionLearner


def _featurize_state(state, ground_nsrt_objects):
    return state.vec(ground_nsrt_objects)

class LifelongSamplerLearningApproachGaussian(BilevelPlanningApproach):
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
        self._explorer_calls = 0

        if CFG.load_lifelong_checkpoint:
            logging.info("\nLoading lifelong checkpoint...")
            chkpt_config_str = f"behavior__lifelong_sampler_learning_gaussian__{CFG.seed}__checkpoint.pt"
            self._save_dict = torch.load(f"{CFG.results_dir}/{chkpt_config_str}")
            for sampler_name in self._save_dict:
                ebm = self._create_network()
                regressor, classifier = ebm
                # It's a specialized sampler
                self._ebms[sampler_name] = ebm
                self._online_learning_cycle = self._save_dict[sampler_name]["online_learning_cycle"] + 1
                self._explorer_calls = self._save_dict[sampler_name]["explorer_calls"]
                if proc_id() == 0:
                    self._replay[sampler_name] = self._save_dict[sampler_name]["replay"]
                else:
                    self._save_dict[sampler_name]["replay"] = None
                    self._replay[sampler_name] = ([], [], [])

                regressor._input_scale = self._save_dict[sampler_name]["regressor_input_scale"]
                regressor._input_shift = self._save_dict[sampler_name]["regressor_input_shift"]
                regressor._output_scale = self._save_dict[sampler_name]["regressor_output_scale"]
                regressor._output_shift = self._save_dict[sampler_name]["regressor_output_shift"]
                regressor._x_dim = self._save_dict[sampler_name]["regressor_x_dim"]
                regressor._y_dim = self._save_dict[sampler_name]["regressor_y_dim"]
                if regressor._x_dim != -1:
                    regressor._initialize_net()
                    regressor.load_state_dict(self._save_dict[sampler_name]["regressor_state"])

                classifier._input_scale = self._save_dict[sampler_name]["classifier_input_scale"]
                classifier._input_shift = self._save_dict[sampler_name]["classifier_input_shift"]
                classifier._x_dim = self._save_dict[sampler_name]["classifier_x_dim"]
                if classifier._x_dim != -1:
                    classifier._initialize_net()
                    classifier.load_state_dict(self._save_dict[sampler_name]["classifier_state"])

                if proc_id() == 0:
                    pass
                else:
                    del self._save_dict[sampler_name]["online_learning_cycle"]
                    del self._save_dict[sampler_name]["explorer_calls"]

            tasks_so_far = (CFG.lifelong_burnin_period or CFG.interactive_num_requests_per_cycle) + (self._online_learning_cycle - 1) * CFG.interactive_num_requests_per_cycle
            tasks_so_far_local = int(np.ceil(tasks_so_far / num_procs()))
            self._next_train_task = tasks_so_far_local

    @classmethod
    def get_name(cls) -> str:
        return "lifelong_sampler_learning_gaussian"

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

    def _create_network(self) -> Tuple[NeuralGaussianRegressor, MLPBinaryClassifier]:
        regressor = NeuralGaussianRegressor(
            seed=CFG.seed,
            hid_sizes=CFG.neural_gaus_regressor_hid_sizes,
            max_train_iters=CFG.neural_gaus_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate)
        classifier = MLPBinaryClassifier(
            seed=CFG.seed,
            balance_data=CFG.mlp_classifier_balance_data,
            max_train_iters=CFG.sampler_mlp_classifier_max_itr,
            learning_rate=CFG.learning_rate,
            n_iter_no_change=CFG.mlp_classifier_n_iter_no_change,
            hid_sizes=CFG.mlp_classifier_hid_sizes,
            n_reinitialize_tries=CFG.sampler_mlp_classifier_n_reinitialize_tries,
            weight_init="default")
        return regressor, classifier


    def _initialize_ebm_samplers(self) -> None:
        # Behavior is written such that there's a different option for every type (e.g., NavigateTo-bucket vs NavigateTo-book),
        # and a separate NSRT for every instance (e.g., NavigateTo-bucket-0 vs NavigateTo-bucket-1) using the type-specific
        # option. To match up the 2D setting, "specialized" samplers are type-specific (so shared across NSRTs, but not across
        # options), and "generic" samplers are shared across options of the same underlying controller 
        sampler_names = set()
        gt_nsrts = self._get_current_gt_nsrts()

        for nsrt in gt_nsrts:
            # Specialized NSRT name
            sampler_names.add('-'.join(nsrt.name.split('-')[:-1]))

        for name in sampler_names:
            if name not in self._ebms:
                states_replay = []
                actions_replay = []
                negative_states_replay = []
                negative_actions_replay = []
                self._replay[name] = (states_replay, actions_replay, negative_states_replay, negative_actions_replay)
                ebm = self._create_network()
                self._ebms[name] = ebm

        new_nsrts = []
        for nsrt in gt_nsrts:
            specialized_name = '-'.join(nsrt.name.split('-')[:-1])
            new_sampler = _LearnedSampler(nsrt.name, self._ebms[specialized_name], nsrt.parameters, nsrt.option, nsrt.sampler).sampler
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

        for name in specialized_names:
            if name not in self._ebms:
                self._ebms[name] = self._create_network()
                self._replay[name] = ([], [], [])

        # Sort dicts by name so order matches across processes
        self._ebms = dict(sorted(self._ebms.items()))
        self._replay = dict(sorted(self._replay.items()))

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

            logging.info(f"({proc_id()}) Creating explorer for task {train_task_idx}")
            explorer = create_explorer(
                "partial_planning_failures",
                preds,
                self._initial_options,
                self._types,
                self._action_space,
                self._train_tasks,
                nsrts,
                self._option_model)
            explorer._num_calls = self._explorer_calls
            logging.info(f"({proc_id()}) Created explorer for task {train_task_idx}")
            explorer.initialize_metrics(explorer_metrics)

            query_policy = self._create_none_query_policy()
            explore_start = time.perf_counter()
            logging.info(f"({proc_id()}) Creating exploration strategy for task {train_task_idx}")
            option_plan, termination_fn, skeleton, traj = explorer.get_exploration_strategy(
                train_task_idx, CFG.timeout)
            logging.info(f"({proc_id()}) Created exploration strategy for task {train_task_idx}")
            explore_time = time.perf_counter() - explore_start
            logging.info(f"({proc_id()}) Creating interaction request for task {train_task_idx}")
            # requests.append(InteractionRequest(train_task_idx, option_plan, query_policy, termination_fn, skeleton, traj))
            length_of_longest_refinement = max([len(option_plan[i]) for i in range(len(option_plan))])
            cnt_max_len = 0
            for i in range(len(option_plan)):
                if len(option_plan[i]) < length_of_longest_refinement:
                    start_from = len(option_plan[i]) - 1
                    # logging.info(f"{start_from}")
                    # logging.info(f"\t{[s.name for s in skeleton[i]]}")
                    # logging.info(f"\t{[o.name for o in option_plan[i]]}")
                    option_plan[i] = option_plan[i][start_from:]
                    skeleton[i] = skeleton[i][start_from:]
                    traj[i] = traj[i][start_from:]
                else:
                    cnt_max_len += 1

            logging.info(f"\t{proc_id()} Will use {cnt_max_len} / {len(option_plan)} refinements for positive samples (len={length_of_longest_refinement})")
            requests += [InteractionRequest(train_task_idx, option_plan[i], query_policy, termination_fn, skeleton[i], traj[i]) for i in range(len(option_plan))]
            logging.info(f"({proc_id()}) Created interaction request for task {train_task_idx}")
            total_time += explore_time
            explorer_metrics = explorer.metrics
            self._explorer_calls = explorer._num_calls
        
        logging.info(f"({proc_id()}) Aggregating metrics for task {train_task_idx}")
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
        logging.info(f"({proc_id()}) Aggregated metrics for task {train_task_idx}")

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
        # logging.info("Entering learning from interaction results")
        traj_list: List[LowLevelTrajectory] = []
        annotations_list: List[Any] = []
        skeleton_list: List[Any] = []

        atoms_cache = {}
        for result in results:
            # logging.info("Entering result annotation loop")
            # logging.info(f"\tTrajectory length: {len(result.options)}")
            skeleton = result.skeleton
            task = self._train_tasks[result.train_task_idx]
            state_annotations = []
            if result.states[0].vanilla_str() not in atoms_cache:
                init_atoms = utils.abstract(result.states[0], self._initial_predicates)
                if len(result.states[0].data) > 0:
                    atoms_cache[result.states[0].vanilla_str()] = init_atoms
            else:
                init_atoms = atoms_cache[result.states[0].vanilla_str()]
            # logging.info("\tObtained init atoms")
            atoms_sequence = [init_atoms]
            for ground_nsrt, state in zip(skeleton, result.states[1:]):
                atoms_sequence.append(utils.apply_operator(ground_nsrt, atoms_sequence[-1]))
                if len(state.data) > 0:
                    if state.vanilla_str() not in atoms_cache:
                        atoms_cache[state.vanilla_str()] = atoms_sequence[-1]
                    # else:
                    #     assert atoms_cache[state.vanilla_str()] == atoms_sequence[-1]

            necessary_atoms_sequence = utils.compute_necessary_atoms_seq(skeleton, atoms_sequence, task.goal)
            # logging.info("\tObtained necessary atoms sequence")

            for state, expected_atoms, ground_nsrt in zip(result.states[1:], necessary_atoms_sequence[1:], result.skeleton):
                if len(state.data) == 0:
                    state_annotations.append(False)     # it is a default state and so the transition failed
                else:
                    state_annotations.append(all(a.holds(state) for a in expected_atoms))
            # logging.info("\tObtained state annotations")
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

        for specialized_sampler_name in self._ebms.keys():
            regressor, classifier = self._ebms[specialized_sampler_name]
            replay = self._replay[specialized_sampler_name]

            states = []
            actions = []
            negative_states = []
            negative_actions = []
            for traj, annotations, skeleton in zip(trajectories, annotations_list, skeletons):
                for state, option, annotation, ground_nsrt in zip(traj.states[:-1], traj.options, annotations, skeleton):
                    # Get this NSRT's positive (successful) data only
                    specialized_name = '-'.join(ground_nsrt.name.split('-')[:-1])
                    if specialized_sampler_name == specialized_name:
                        x = _featurize_state(state, ground_nsrt.objects)
                        a = option.params
                        if annotation > 0:
                            states.append(x)
                            actions.append(a)
                        else:
                            negative_states.append(x)
                            negative_actions.append(a)

            states_arr = np.array(states)
            actions_arr = np.array(actions)
            negative_states_arr = np.array(negative_states)
            negative_actions_arr = np.array(negative_actions)

            states_arr = mpi_concatenate(states_arr)
            actions_arr = mpi_concatenate(actions_arr)
            negative_states_arr = mpi_concatenate(negative_states_arr)
            negative_actions_arr = mpi_concatenate(negative_actions_arr)

            if proc_id() == 0:
                logging.info(f"{specialized_sampler_name}: {states_arr.shape[0]} positive samples, {negative_states_arr.shape[0]} negative samples, {states_arr.shape[1]} features, {actions_arr.shape[1]} outputs")
                if states_arr.shape[0] > 0:
                    start_time = time.perf_counter()
                    if regressor._x_dim == -1:
                        unique_actions = np.unique(actions_arr, axis=0)
                        if unique_actions.shape[0] == 1 \
                            or specialized_sampler_name.startswith("Open") \
                            or specialized_sampler_name.startswith("Close") \
                            or specialized_sampler_name.startswith("ToggleOn") \
                            or specialized_sampler_name.startswith("CleanDusty"):
                            logging.info(f"Setting regressor to constant value {unique_actions[0]}")
                            regressor.predict_sample = lambda x, rng: unique_actions[0]
                            classifier.classify = lambda x: True
                        else:
                            regressor.fit(states_arr, actions_arr)
                            if negative_states_arr.shape[0] > 0:
                                X_positive = np.c_[states_arr, actions_arr]
                                X_negative = np.c_[negative_states_arr, negative_actions_arr]
                                # logging.info(f"s: {states_arr.shape}, a: {actions_arr.shape}, sn: {negative_states_arr.shape}, an: {negative_actions_arr.shape}")
                                X_classifier = np.r_[X_positive, X_negative]
                                y_classifier = np.r_[np.ones(X_positive.shape[0]), np.zeros(X_negative.shape[0])]
                                classifier.fit(X_classifier, y_classifier)
                    else:
                        states_replay = np.array(replay[0])
                        actions_replay = np.array(replay[1])
                        negative_states_replay = np.array(replay[2])
                        negative_actions_replay = np.array(replay[3])

                        if CFG.lifelong_method == "distill" or CFG.lifelong_method == "2-distill":
                            # # First copy: train model just on new data
                            # ebm_new = copy.deepcopy(ebm)
                            # ebm_new.fit(states_arr, actions_arr, aux_labels_arr)

                            # # Second copy: previous version to distill into updated model
                            # ebm_old = copy.deepcopy(ebm)

                            # # Distill new and old models into updated model
                            # ebm_new_data = (ebm_new, (states_arr, actions_arr, aux_labels_arr))
                            # ebm_old_data = (ebm_old, (states_replay, actions_replay, aux_labels_replay))
                            # ebm.distill(ebm_old_data, ebm_new_data)
                            raise NotImplementedError
                        elif CFG.lifelong_method == "retrain":
                            # # Instead, try re-training the model as a performance upper bound
                            # states_full = np.r_[states_arr, states_replay]
                            # actions_full = np.r_[actions_arr, actions_replay]
                            # aux_labels_full = np.r_[aux_labels_arr, aux_labels_replay]
                            # ebm.fit(states_full, actions_full, aux_labels_full)
                            raise NotImplementedError
                        elif CFG.lifelong_method == "retrain-scratch":
                            states_all = np.r_[states_arr, states_replay]
                            actions_all = np.r_[actions_arr, actions_replay]
                            unique_actions = np.unique(actions_all, axis=0)
                            if unique_actions.shape[0] == 1 \
                                or specialized_sampler_name.startswith("Open") \
                                or specialized_sampler_name.startswith("Close") \
                                or specialized_sampler_name.startswith("ToggleOn") \
                                or specialized_sampler_name.startswith("CleanDusty"):
                                logging.info(f"Setting regressor to constant value {unique_actions[0]}")
                                regressor.predict_sample = lambda x, rng: unique_actions[0]
                                classifier.classify = lambda x: True
                            else:
                                regressor.fit(states_all, actions_all)

                                if negative_states_arr.shape[0] > 0 or negative_states_replay.shape[0] > 0:
                                    X_positive_new = np.c_[states_arr, actions_arr]
                                    X_negative_new = np.c_[negative_states_arr, negative_actions_arr]
                                    if X_negative_new.shape[0] > 0:
                                        X_classifier_new = np.r_[X_positive_new, X_negative_new]
                                        y_classifier_new = np.r_[np.ones(X_positive_new.shape[0]), np.zeros(X_negative_new.shape[0])]
                                    else:
                                        X_classifier_new = X_positive_new
                                        y_classifier_new = np.ones(X_positive_new.shape[0])
                                    
                                    X_positive_old = np.c_[states_replay, actions_replay]
                                    X_negative_old = np.c_[negative_states_replay, negative_actions_replay]
                                    if X_negative_old.shape[0] > 0:
                                        X_classifier_old = np.r_[X_positive_old, X_negative_old]
                                        y_classifier_old = np.r_[np.ones(X_positive_old.shape[0]), np.zeros(X_negative_old.shape[0])]
                                    else:
                                        X_classifier_old = X_positive_old
                                        y_classifier_old = np.ones(X_positive_old.shape[0])
                                    X_classifier_all = np.r_[X_classifier_new, X_classifier_old]
                                    y_classifier_all = np.r_[y_classifier_new, y_classifier_old]
                                    classifier.fit(X_classifier_all, y_classifier_all)
                        elif CFG.lifelong_method == 'finetune':
                            # ebm.fit(states_arr, actions_arr, aux_labels_arr)
                            raise NotImplementedError
                        elif CFG.lifelong_method == "retrain-balanced":
                            raise NotImplementedError
                        else:
                            raise NotImplementedError(f"Unknown lifelong method {CFG.lifelong_method}")            
                    end_time = time.perf_counter()
                    logging.info(f"Training time: {(end_time - start_time):.5f} seconds")
                    replay[0].extend(list(states_arr))
                    replay[1].extend(list(actions_arr))
                    replay[2].extend(list(negative_states_arr))
                    replay[3].extend(list(negative_actions_arr))
                    self._save_dict[specialized_sampler_name] = {
                        # regressor
                        "regressor_state": regressor.state_dict(),
                        "regressor_input_scale": regressor._input_scale,
                        "regressor_input_shift": regressor._input_shift,
                        "regressor_output_scale": regressor._output_scale,
                        "regressor_output_shift": regressor._output_shift,
                        "regressor_y_dim": regressor._y_dim,
                        "regressor_x_dim": regressor._x_dim,
                        # classifier
                        "classifier_state": classifier.state_dict(),
                        "classifier_input_scale": classifier._input_scale,
                        "classifier_input_shift": classifier._input_shift,
                        "classifier_x_dim": classifier._x_dim,
                        # others
                        "replay": replay,
                        "online_learning_cycle": self._online_learning_cycle,
                        "explorer_calls": self._explorer_calls,
                    }   

        logging.info(f"Out of training loop ({proc_id()})")
        if proc_id() == 0:
            chkpt_config_str = f"behavior__lifelong_sampler_learning_gaussian__{CFG.seed}__checkpoint.pt"
            torch.save(self._save_dict, f"{CFG.results_dir}/{chkpt_config_str}")
        self._save_dict = broadcast_object(self._save_dict, root=0)
        logging.info(f"Broadcasted save_dict ({proc_id()})")
        if proc_id() != 0:
            for specialized_sampler_name in self._ebms.keys():
                ebm = self._ebms[specialized_sampler_name]
                regressor, classifier = ebm
                if specialized_sampler_name in self._save_dict:
                    # logging.info(f"Loading keys into process {proc_id()}")
                    del self._save_dict[specialized_sampler_name]["replay"]
                    del self._save_dict[specialized_sampler_name]["online_learning_cycle"]
                    del self._save_dict[specialized_sampler_name]["explorer_calls"]
                    regressor._input_scale = self._save_dict[specialized_sampler_name]["regressor_input_scale"]
                    regressor._input_shift = self._save_dict[specialized_sampler_name]["regressor_input_shift"]
                    regressor._output_scale = self._save_dict[specialized_sampler_name]["regressor_output_scale"]
                    regressor._output_shift = self._save_dict[specialized_sampler_name]["regressor_output_shift"]
                    regressor._x_dim = self._save_dict[specialized_sampler_name]["regressor_x_dim"]
                    regressor._y_dim = self._save_dict[specialized_sampler_name]["regressor_y_dim"]
                    if regressor._x_dim != -1:
                        regressor._initialize_net()
                        regressor.load_state_dict(self._save_dict[specialized_sampler_name]["regressor_state"])

                    classifier._input_scale = self._save_dict[specialized_sampler_name]["classifier_input_scale"]
                    classifier._input_shift = self._save_dict[specialized_sampler_name]["classifier_input_shift"]
                    classifier._x_dim = self._save_dict[specialized_sampler_name]["classifier_x_dim"]
                    if classifier._x_dim != -1:
                        classifier._initialize_net()
                        classifier.load_state_dict(self._save_dict[specialized_sampler_name]["classifier_state"])

@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler:
    """A convenience class for holding the models underlying a learned
    sampler."""
    _name: str
    _ebm: Tuple[NeuralGaussianRegressor, MLPBinaryClassifier]
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption
    _original_sampler: NSRTSampler

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object],
                return_failed_samples: Optional[bool] = False) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        x_lst: List[Any] = []  
        sub = dict(zip(self._variables, objects))
        for var in self._variables:
            x_lst.extend(state[sub[var]])
        x = np.array(x_lst)
        assert (x == state.vec(objects)).all()

        regressor, classifier = self._ebm
        if regressor._x_dim == -1: # hasn't been fit, revert to og 
            return self._original_sampler(state, goal, rng, objects, return_failed_samples=return_failed_samples)
        if classifier._x_dim == -1:
            params = np.array(regressor.predict_sample(x, rng),
                                  dtype=self._param_option.params_space.dtype)
        else:
            num_rejections = 0
            while num_rejections <= CFG.max_rejection_sampling_tries:
                params = np.array(regressor.predict_sample(x, rng),
                                      dtype=self._param_option.params_space.dtype)
                if self._param_option.params_space.contains(params) and \
                    classifier.classify(np.r_[x, params]):
                    break  
                num_rejections += 1              
        if return_failed_samples:
            return params, 1, []
        return params, 1
