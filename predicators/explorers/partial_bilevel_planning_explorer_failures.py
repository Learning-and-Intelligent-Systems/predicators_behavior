"""An explorer that explores by solving tasks with bilevel planning."""

from typing import List, Set

from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.explorers.bilevel_planning_explorer_failures import \
    BilevelPlanningExplorerFailures
from predicators.explorers.random_options_explorer import RandomOptionsExplorer
from predicators.planning import PlanningFailure, PlanningTimeout
from predicators.structs import ExplorationStrategy
from predicators.settings import CFG

class PartialBilevelPlanningExplorer(BilevelPlanningExplorerFailures):
    """PartialBilevelPlanningExplorer implementation."""

    @classmethod
    def get_name(cls) -> str:
        return "partial_planning_failures"

    def get_exploration_strategy(self, train_task_idx: int,
                                 timeout: int) -> ExplorationStrategy:
        task = self._train_tasks[train_task_idx]
        try:
            return self._solve(task, timeout)
        except (PlanningFailure, PlanningTimeout) as e:
            # print(f'Failed to refine plan, {len(e.info["partial_refinements"])} partial partial_refinements')
            skeleton_list, partial_plan_list, partial_traj_list = list(map(list, zip(*(e.info["partial_refinements"]))))
            # When the policy finishes, an OptionExecutionFailure is raised
            # and caught, terminating the episode.
            termination_function = lambda _: False

            self._save_failure_metrics(task)
            # partial_plan_list = [p[:-1] for p in partial_plan_list]
            # partial_traj_list = [t[:-1] for t in partial_traj_list]

            return partial_plan_list, termination_function, skeleton_list, partial_traj_list     # DO return the failed action

    def _save_failure_metrics(self, task) -> None:
        self._metrics["total_num_samples_failed"] += CFG.sesame_max_samples_total
        self._metrics["num_unsolved"] += 1

        if len(CFG.behavior_task_list) != 1:
            env = get_or_create_env(CFG.env)
            subenv_name = CFG.behavior_task_list[env.task_list_indices[env.task_num]]
    
            self._metrics[f"env_{subenv_name}_total_num_samples_failed"] += CFG.sesame_max_samples_total
            self._metrics[f"env_{subenv_name}_num_unsolved"] += 1            