"""An explorer that explores by solving tasks with bilevel planning."""

from typing import List, Set

from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.explorers.bilevel_planning_explorer import \
    BilevelPlanningExplorer
from predicators.explorers.random_options_explorer import RandomOptionsExplorer
from predicators.planning import PlanningFailure, PlanningTimeout
from predicators.structs import ExplorationStrategy
from predicators.settings import CFG

class PartialBilevelPlanningExplorer(BilevelPlanningExplorer):
    """PartialBilevelPlanningExplorer implementation."""

    @classmethod
    def get_name(cls) -> str:
        return "partial_planning"

    def get_exploration_strategy(self, train_task_idx: int,
                                 timeout: int) -> ExplorationStrategy:
        task = self._train_tasks[train_task_idx]
        try:
            return self._solve(task, timeout)
        except (PlanningFailure, PlanningTimeout) as e:
            # print(f'Failed to refine plan, {len(e.info["partial_refinements"])} partial partial_refinements')
            skeleton, partial_plan, partial_traj = e.info["partial_refinements"][0]
            partial_metrics = e.info["partial_metrics"]
            # When the policy finishes, an OptionExecutionFailure is raised
            # and caught, terminating the episode.
            termination_function = lambda _: False

            self._save_failure_metrics(task, partial_metrics)
            return partial_plan[:-1], termination_function, skeleton, partial_traj[:-1]     # dont return the failed action

    def _save_failure_metrics(self, task, partial_metrics) -> None:
        self._metrics["total_num_samples_failed"] += CFG.sesame_max_samples_total
        self._metrics["total_sampling_time_failed"] += partial_metrics["sampling_time"]
        self._metrics["num_unsolved"] += 1

        if len(CFG.behavior_task_list) != 1:
            env = get_or_create_env(CFG.env)
            subenv_name = CFG.behavior_task_list[env.task_list_indices[env.task_num]]
    
            self._metrics[f"env_{subenv_name}_total_num_samples_failed"] += CFG.sesame_max_samples_total
            self._metrics[f"env_{subenv_name}_total_sampling_time_failed"] += partial_metrics["sampling_time"]      
            self._metrics[f"env_{subenv_name}_num_unsolved"] += 1      
