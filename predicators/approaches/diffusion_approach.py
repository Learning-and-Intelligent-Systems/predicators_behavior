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
    Query, State, Task, Type, Variable, Metrics
from predicators.ml_models import MLPBinaryClassifier, \
    NeuralGaussianRegressor, DiffusionRegressor
from predicators import utils
from predicators.envs import get_or_create_env
from multiprocessing import Pool
from predicators.teacher import Teacher, TeacherInteractionMonitor

def _featurize_state(state, ground_nsrt_objects):
    assert not CFG.use_full_state
    assert not CFG.use_skeleton_state
    return state.vec(ground_nsrt_objects)

def _featurize_state(all_args):
    nsrt_names, nsrt_parameters, horizon, state, ground_nsrt_objects, skeleton_names, skeleton_objects = all_args
    x = state.vec(ground_nsrt_objects)
    if CFG.use_full_state:
        # The full state is represented as the image observation of the env
        env = get_or_create_env(CFG.env)
        img = env.render_state(state, None)[0][::6,::6,:3].reshape(-1)
        # print(img.shape)
        # plt.imshow(img)
        # plt.show()
        # exit()
        # img = env.grid_state(state).reshape(-1)
        x = np.r_[img, x]
    if CFG.use_skeleton_state:
        # The skeleton representation is a series of self._horizon one-hot vectors
        # indicating which action is executed, plus a series of self._horizon * num_actions
        # vectors, where the chosen action per step contains the object features of the
        # operator objects, while the other actions contain all-zeros
        num_nsrts = len(nsrt_names)
        skeleton_rep = np.zeros(0)
        for t in range(horizon):
            one_hot = np.zeros(num_nsrts)
            if t < len(skeleton_names):
                one_hot[nsrt_names.index(skeleton_names[t])] = 1
            nsrt_object_rep = np.zeros(0)
            for nsrt_tmp_name, nsrt_tmp_parameters in zip(nsrt_names, nsrt_parameters):
                if t < len(skeleton_names) and nsrt_tmp_name == skeleton_names[t]:
                    rep = state.vec(skeleton_objects[t])
                    assert state.vec(skeleton_objects[t]).shape[0] == sum(obj.type.dim for obj in nsrt_tmp_parameters), f'{state.vec(skeleton_objects[t]).shape[0]}, {sum(obj.type.dim for obj in nsrt_tmp_parameters)}, {nsrt_tmp_name}, {skeleton_objects[t]}, {nsrt_tmp_parameters}'
                else:
                    rep = np.zeros(sum(obj.type.dim for obj in nsrt_tmp_parameters))
                nsrt_object_rep = np.r_[nsrt_object_rep, rep]
            skeleton_rep = np.r_[skeleton_rep, one_hot, nsrt_object_rep]
        x = np.r_[x, skeleton_rep]
    return x
class DiffusionApproach(BilevelPlanningApproach):
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
        self._nsrts = get_gt_nsrts(CFG.env, self._initial_predicates,
                                   self._initial_options)
        self._online_learning_cycle = 0
        self._next_train_task = 0
        # assert CFG.timeout == float('inf'), "We don't want to let methods time out in these experiments"
        # assert not CFG.bookshelf_add_sampler_idx_to_params, "Code assumes the env does not expect extra dummy param"
        self._save_dict = {}

    @classmethod
    def get_name(cls) -> str:
        return "diffusion"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_nsrts(self) -> Set[NSRT]:
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
        new_nsrts = []
        self._ebms = []
        self._replay = []

        for nsrt in self._nsrts:
            states_replay = []
            actions_replay = []
            self._replay.append((states_replay, actions_replay))
            ebm = self._create_network()

            new_sampler = _LearnedSampler(nsrt.name, ebm, nsrt.parameters, nsrt.option, self._nsrts,
                                          nsrt.sampler).sampler
            self._ebms.append(ebm)

            new_nsrts.append(NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                                  nsrt.add_effects, nsrt.delete_effects,
                                  nsrt.ignore_effects, nsrt.option,
                                  nsrt.option_vars, new_sampler))

            logging.info("NEW NSRT CREATED!")
        self._nsrts = new_nsrts

    def get_interaction_requests(self):
        requests = []
        explorer = create_explorer(
            "partial_planning",
            self._initial_predicates,
            self._initial_options,
            self._types,
            self._action_space,
            self._train_tasks,
            self._get_current_nsrts(),
            self._option_model)

        metrics: Metrics = defaultdict(float)

        if CFG.lifelong_burnin_period is None or self._online_learning_cycle > 0:
            num_tasks = CFG.interactive_num_requests_per_cycle
        else:
            num_tasks = CFG.lifelong_burnin_period
        first_train_task = self._next_train_task
        self._next_train_task += num_tasks
        # Get the next tasks in the sequence
        total_time = 0
        for train_task_idx in range(first_train_task, self._next_train_task):
            query_policy = self._create_none_query_policy()

            explore_start = time.perf_counter()
            act_policy, termination_fn, skeleton = explorer.get_exploration_strategy(
                train_task_idx, CFG.timeout)
            explore_time = time.perf_counter() - explore_start
            requests.append(InteractionRequest(train_task_idx, act_policy, query_policy, termination_fn, skeleton))

            total_time += explore_time
        num_unsolved = explorer.metrics["num_unsolved"]
        num_solved = explorer.metrics["num_solved"]
        num_total = num_unsolved + num_solved
        assert num_total == num_tasks
        avg_time = total_time / num_tasks
        metrics["num_solved"] = num_solved
        metrics["num_unsolved"] = num_unsolved
        metrics["num_total"] = num_tasks
        metrics["avg_time"] = avg_time
        metrics["min_num_samples"] = explorer.metrics["min_num_samples"]
        metrics["max_num_samples"] = explorer.metrics["max_num_samples"]
        metrics["min_num_skeletons_optimized"] = explorer.metrics["min_num_skeletons_optimized"]
        metrics["max_num_skeletons_optimized"] = explorer.metrics["max_num_skeletons_optimized"]
        metrics["num_solve_failures"] = num_unsolved

        for metric_name in [
            "num_samples", "num_skeletons_optimized", "num_nodes_expanded",
            "num_nodes_created", "num_nsrts", "num_preds", "plan_length",
            "num_failures_discovered"
        ]:
            total = explorer.metrics[f"total_{metric_name}"]
            metrics[f"avg_{metric_name}"] = (
                total / num_solved if num_solved > 0 else float("inf"))
        total = explorer.metrics["total_num_samples_failed"]
        metrics["avg_num_samples_failed"] = total / num_unsolved if num_unsolved > 0 else float("inf")

        logging.info(f"Tasks solved: {int(num_solved)} / {num_tasks}")
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
            state_annotations = []
            atoms_prev_state = utils.abstract(result.states[0], self._initial_predicates)
            for state, ground_nsrt in zip(result.states[1:], result.skeleton):
                atoms_state = utils.abstract(state, self._initial_predicates)
                expected_atoms = utils.apply_operator(ground_nsrt, atoms_prev_state)
                state_annotations.append(all(a.holds(state) for a in expected_atoms))
                atoms_prev_state = atoms_state

            traj = LowLevelTrajectory(result.states, result.actions)
            traj_list.append(traj)
            annotations_list.append(state_annotations)
            skeleton_list.append(result.skeleton)
        self._update_samplers(traj_list, annotations_list, skeleton_list)
        self._online_learning_cycle += 1

    def _update_samplers(self, trajectories: List[LowLevelTrajectory], annotations_list: List[Any],
                         skeletons: List[Any]) -> None:
        logging.info("\nUpdating the samplers...")
        if self._online_learning_cycle == 0:
            self._initialize_ebm_samplers()
        logging.info("Featurizing the samples...")

        for ebm, nsrt, replay in zip(self._ebms, self._nsrts, self._replay):
            # Get the data corresponding to the current NSRT
            states = []
            actions = []
            for traj, annotations, skeleton in zip(trajectories, annotations_list, skeletons):
                for state, action, annotation, ground_nsrt in zip(traj.states[:-1], traj.actions, annotations,
                                                                  skeleton):
                    option = action.get_option()
                    # Get this NSRT's positive (successful) data only
                    if annotation > 0:
                        if nsrt.name == ground_nsrt.name:
                            x = _featurize_state(state, ground_nsrt.objects)
                            a = option.params
                            states.append(x)
                            actions.append(a)

            states_arr = np.array(states)
            actions_arr = np.array(actions)

            logging.info(f"{nsrt.name}: {states_arr.shape[0]} samples")
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
                replay[0].extend(states)
                replay[1].extend(actions)
                self._save_dict[nsrt.name] = {
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

        torch.save(self._save_dict, f"{CFG.results_dir}/{utils.get_config_path_str()}__checkpoint.pt")

@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler:
    """A convenience class for holding the models underlying a learned
    sampler."""
    _name: str
    _ebm: DiffusionRegressor
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption
    _nsrts: List[NSRT]
    _original_sampler: NSRTSampler

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object],
                skeleton: List[Any]) -> Array:
        """The sampler corresponding to the given models.
        May be used as the _sampler field in an NSRT.
        """
        if not self._ebm.is_trained:
            logging("TRAIN ME TRAIN ME TRAIN ME")
            return self._original_sampler(state, goal, rng, objects, skeleton)
        else:
            logging("NEW SAMPLER!!!!!!!!!!!!!!!!!!!!!!!!!")
        x_lst: List[Any] = []
        sub = dict(zip(self._variables, objects))
        for var in self._variables:
            x_lst.extend(state[sub[var]])
        x = np.array(x_lst)
        assert (x == state.vec(objects)).all()

        params = np.array(self._ebm.predict_sample(x, rng),
                         dtype=self._param_option.params_space.dtype)

        return params