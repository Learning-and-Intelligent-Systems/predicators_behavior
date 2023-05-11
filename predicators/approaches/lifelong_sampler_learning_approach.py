import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set
import copy
from collections import defaultdict
import time
import pickle as pkl

import numpy as np
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
from predicators.behavior_utils.behavior_utils import load_checkpoint_state
from predicators.envs import get_or_create_env
from predicators.envs.behavior import BehaviorEnv
from multiprocessing import Pool
from predicators.teacher import Teacher, TeacherInteractionMonitor

from predicators.mpi_utils import proc_id, num_procs, mpi_sum, mpi_concatenate, broadcast_object, mpi_min, mpi_max

def _featurize_state(state, ground_nsrt_objects):
    return state.vec(ground_nsrt_objects)

class LifelongSamplerLearningApproach(BilevelPlanningApproach):
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
        # self._gt_nsrts = get_gt_nsrts(CFG.env, self._initial_predicates,
        #                            self._initial_options)
        self._online_learning_cycle = 0
        self._next_train_task = 0
        assert CFG.timeout == float('inf'), "We don't want to let methods time out in these experiments"
        self._save_dict = {}
        self._ebms = {}
        self._replay = {}

    @classmethod
    def get_name(cls) -> str:
        return "lifelong_sampler_learning"

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
            if CFG.use_specialized_ebms:
                sampler_names.add('-'.join(nsrt.name.split('-')[:-1]))
            else:
                sampler_names.add('-'.join(nsrt.name.split('-')[:-2]))

        for name in sampler_names:
            if name not in self._ebms:
                states_replay = []
                actions_replay = []
                self._replay[name] = (states_replay, actions_replay)
                ebm = self._create_network()
                self._ebms[name] = ebm

        new_nsrts = []
        for nsrt in gt_nsrts:
            if CFG.use_specialized_ebms:
                name = '-'.join(nsrt.name.split('-')[:-1])
            else:
                name = '-'.join(nsrt.name.split('-')[:-2])
            new_sampler = _LearnedSampler(nsrt.name, self._ebms[name], nsrt.parameters, nsrt.option, nsrt.sampler).sampler
            new_nsrts.append(NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                                  nsrt.add_effects, nsrt.delete_effects,
                                  nsrt.ignore_effects, nsrt.option, 
                                  nsrt.option_vars, new_sampler))
        self._nsrts = new_nsrts

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

            explorer = create_explorer(
                "partial_planning",
                preds,
                self._initial_options,
                self._types,
                self._action_space,
                self._train_tasks,
                nsrts,
                self._option_model)
            explorer.initialize_metrics(explorer_metrics)

            query_policy = self._create_none_query_policy()
            explore_start = time.perf_counter()
            option_plan, termination_fn, skeleton, traj = explorer.get_exploration_strategy(
                train_task_idx, CFG.timeout)
            explore_time = time.perf_counter() - explore_start
            requests.append(InteractionRequest(train_task_idx, option_plan, query_policy, termination_fn, skeleton, traj))
            total_time += explore_time

            explorer_metrics = explorer.metrics
        
        num_unsolved = int(mpi_sum(explorer_metrics["num_unsolved"]))
        num_solved = int(mpi_sum(explorer_metrics["num_solved"]))
        num_total = num_unsolved + num_solved
        avg_time = mpi_sum(total_time) / num_tasks
        assert num_total == num_tasks
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
            state_annotations = []
            init_atoms = utils.abstract(result.states[0], self._initial_predicates)
            atoms_sequence = [init_atoms]
            for ground_nsrt in skeleton:
                atoms_sequence.append(utils.apply_operator(ground_nsrt, atoms_sequence[-1]))
            necessary_atoms_sequence = utils.compute_necessary_atoms_seq(skeleton, atoms_sequence, task.goal)

            for state, expected_atoms, ground_nsrt in zip(result.states[1:], necessary_atoms_sequence[1:], result.skeleton):
                state_annotations.append(all(a.holds(state) for a in expected_atoms))
            logging.info(f"State annotations: {state_annotations}")
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

        # for ebm, nsrt, replay in zip(self._ebms, self._nsrts, self._replay):
        for sampler_name in self._ebms.keys():
            ebm = self._ebms[sampler_name]
            replay = self._replay[sampler_name]
            states = []
            actions = []
            for traj, annotations, skeleton in zip(trajectories, annotations_list, skeletons):
                for state, option, annotation, ground_nsrt in zip(traj.states[:-1], traj.options, annotations, skeleton):
                    # Get this NSRT's positive (successful) data only
                    if annotation > 0:
                        if CFG.use_specialized_ebms:
                            name = '-'.join(ground_nsrt.name.split('-')[:-1])
                        else:
                            name = '-'.join(ground_nsrt.name.split('-')[:-2])
                        if sampler_name == name:
                            x = _featurize_state(state, ground_nsrt.objects)
                            a = option.params
                            states.append(x)
                            actions.append(a)

            states_arr = np.array(states)
            actions_arr = np.array(actions)
            states_arr = mpi_concatenate(states_arr)
            actions_arr = mpi_concatenate(actions_arr)
            if proc_id() == 0:
                logging.info(f"{sampler_name}: {states_arr.shape[0]} samples, {states_arr.shape[1]} features, {actions_arr.shape[1]} outputs")
                if states_arr.shape[0] > 0:
                    start_time = time.perf_counter()
                    if not ebm.is_trained:
                        ebm.fit(states_arr, actions_arr)
                    else:
                        states_replay = np.array(replay[0])
                        actions_replay = np.array(replay[1])

                        if CFG.lifelong_method == "distill" or CFG.lifelong_method == "2-distill":
                            # First copy: train model just on new data
                            ebm_new = copy.deepcopy(ebm)
                            ebm_new.fit(states_arr, actions_arr)

                            # Second copy: previous version to distill into updated model
                            ebm_old = copy.deepcopy(ebm)

                            # Distill new and old models into updated model
                            ebm_new_data = (ebm_new, (states_arr, actions_arr, None))
                            ebm_old_data = (ebm_old, (states_replay, actions_replay, None))
                            ebm.distill(ebm_old_data, ebm_new_data)
                        elif CFG.lifelong_method == "retrain":
                            # Instead, try re-training the model as a performance upper bound
                            states_full = np.r_[states_arr, states_replay]
                            actions_full = np.r_[actions_arr, actions_replay]
                            ebm.fit(states_full, actions_full)
                        elif CFG.lifelong_method == "retrain-scratch":
                            ebm._linears = torch.nn.ModuleList()
                            ebm._optimizer = None
                            ebm.is_trained = False
                            states_full = np.r_[states_arr, states_replay]
                            actions_full = np.r_[actions_arr, actions_replay]
                            ebm.fit(states_full, actions_full)
                        elif CFG.lifelong_method == 'finetune':
                            ebm.fit(states_arr, actions_arr)
                        else:
                            raise NotImplementedError(f"Unknown lifelong method {CFG.lifelong_method}")            
                    end_time = time.perf_counter()
                    logging.info(f"Training time: {(end_time - start_time):.5f} seconds")
                    replay[0].extend(list(states_arr))
                    replay[1].extend(list(actions_arr))
                    self._save_dict[sampler_name] = {
                        "optimizer_state": ebm._optimizer.state_dict(),
                        "model_state": ebm.state_dict(),
                        "input_scale": ebm._input_scale,
                        "input_shift": ebm._input_shift,
                        "output_scale": ebm._output_scale,
                        "output_shift": ebm._output_shift,
                        "is_trained": ebm.is_trained,
                        "x_cond_dim": ebm._x_cond_dim,
                        "t_dim": ebm._t_dim,
                        "y_dim": ebm._y_dim,
                        "x_dim": ebm._x_dim,
                        "replay": replay,
                        "online_learning_cycle": self._online_learning_cycle,
                    }   
        if proc_id() == 0:
            torch.save(self._save_dict, f"{CFG.results_dir}/{utils.get_config_path_str()}__checkpoint.pt")
        self._save_dict = broadcast_object(self._save_dict, root=0)
        if proc_id() != 0:
            for sampler_name in self._ebms.keys():
                ebm = self._ebms[sampler_name]
                if sampler_name in self._save_dict:
                    logging.info(f"Loading keys into process {proc_id()}: {self._save_dict[sampler_name]['model_state'].keys()}")
                    del self._save_dict[sampler_name]["replay"]
                    del self._save_dict[sampler_name]["optimizer_state"]
                    del self._save_dict[sampler_name]["online_learning_cycle"]
                    ebm._input_scale = self._save_dict[sampler_name]["input_scale"]
                    ebm._input_shift = self._save_dict[sampler_name]["input_shift"]
                    ebm._output_scale = self._save_dict[sampler_name]["output_scale"]
                    ebm._output_shift = self._save_dict[sampler_name]["output_shift"]
                    ebm.is_trained = self._save_dict[sampler_name]["is_trained"]
                    ebm._x_cond_dim = self._save_dict[sampler_name]["x_cond_dim"]
                    ebm._t_dim = self._save_dict[sampler_name]["t_dim"]
                    ebm._y_dim = self._save_dict[sampler_name]["y_dim"]
                    ebm._x_dim = self._save_dict[sampler_name]["x_dim"]

                    ebm._initialize_net()
                    ebm.to(ebm._device)
                    ebm.load_state_dict(self._save_dict[sampler_name]["model_state"])


@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler:
    """A convenience class for holding the models underlying a learned
    sampler."""
    _name: str
    _ebm: DiffusionRegressor
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption
    _original_sampler: NSRTSampler

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object]) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        if not self._ebm.is_trained:
            return self._original_sampler(state, goal, rng, objects)
        x_lst: List[Any] = []  
        sub = dict(zip(self._variables, objects))
        for var in self._variables:
            x_lst.extend(state[sub[var]])
        x = np.array(x_lst)
        assert (x == state.vec(objects)).all()

        params = np.array(self._ebm.predict_sample(x, rng),
                         dtype=self._param_option.params_space.dtype)
        # Returns parameters and "internal_samples", which in this case is just always 1 (there's no internal rejection sampling, and concretely no calls to the simulator)
        return params, 1
