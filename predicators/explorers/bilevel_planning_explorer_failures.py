"""An explorer that uses bilevel planning with NSRTs."""

from typing import List, Optional, Set
from collections import defaultdict

from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.explorers.base_explorer import BaseExplorer
from predicators.option_model import _OptionModelBase
from predicators.planning import sesame_plan
from predicators.settings import CFG
from predicators.structs import Metrics, NSRT, ExplorationStrategy, \
    ParameterizedOption, Predicate, Task, Type


class BilevelPlanningExplorerFailures(BaseExplorer):
    """BilevelPlanningExplorer implementation.

    This explorer is abstract: subclasses decide how to use the _solve
    method implemented in this class, which calls sesame_plan().
    """

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task], nsrts: Set[NSRT],
                 option_model: _OptionModelBase) -> None:

        super().__init__(predicates, options, types, action_space, train_tasks)
        self._nsrts = nsrts
        self._option_model = option_model
        self._num_calls = 0
        self.initialize_metrics()

    @property
    def metrics(self) -> Metrics:
        return self._metrics.copy()

    def _solve(self, task: Task, timeout: int) -> ExplorationStrategy:

        # Ensure random over successive calls.
        self._num_calls += 1
        seed = self._seed + self._num_calls
        # Note: subclasses are responsible for catching PlanningFailure and
        # PlanningTimeout and handling them accordingly.
        plan_list, metrics, traj_list, skeleton_list = sesame_plan(
            task,
            self._option_model,
            self._nsrts,
            self._predicates,
            self._types,
            timeout,
            seed,
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon,
            allow_noops=CFG.sesame_allow_noops,
            use_visited_state_set=CFG.sesame_use_visited_state_set,
            return_skeleton=True,
            return_all_failed_refinements=True)
        self._save_metrics(metrics)
        termination_function = task.goal_holds

        return plan_list, termination_function, skeleton_list, traj_list

    def initialize_metrics(self, metrics: Optional[Metrics] = None) -> None:
        if metrics is None:
            metrics = defaultdict(float)
        self._metrics = metrics
    
    def _save_metrics(self, metrics: Metrics) -> None:
        for metric in [
                "num_samples", "num_skeletons_optimized",
                "num_failures_discovered", "num_nodes_expanded",
                "num_nodes_created", "plan_length"
        ]:
            self._metrics[f"total_{metric}"] += metrics[metric]
        for metric in [
                "num_samples",
                "num_skeletons_optimized",
        ]:
            self._metrics[f"min_{metric}"] = min(
                metrics[metric], self._metrics[f"min_{metric}"])
            self._metrics[f"max_{metric}"] = max(
                metrics[metric], self._metrics[f"max_{metric}"])
        self._metrics["num_solved"] += 1

        if len(CFG.behavior_task_list) != 1:
            env = get_or_create_env(CFG.env)
            subenv_name = CFG.behavior_task_list[env.task_list_indices[env.task_num]]

            for metric in [
                    "num_samples", "num_skeletons_optimized",
                    "num_failures_discovered", "num_nodes_expanded",
                    "num_nodes_created", "plan_length"
            ]:
                self._metrics[f"env_{subenv_name}_total_{metric}"] += metrics[metric]
            for metric in [
                    f"num_samples",
                    f"num_skeletons_optimized",
            ]:
                self._metrics[f"env_{subenv_name}_min_{metric}"] = min(
                    metrics[metric], self._metrics[f"env_{subenv_name}_min_{metric}"])
                self._metrics[f"env_{subenv_name}_max_{metric}"] = max(
                    metrics[metric], self._metrics[f"env_{subenv_name}_max_{metric}"])
            self._metrics[f"env_{subenv_name}_num_solved"] += 1