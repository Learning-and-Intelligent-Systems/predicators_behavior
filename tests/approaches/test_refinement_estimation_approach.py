"""Test cases for the refinement cost estimation--based approach class."""

from predicators import utils
from predicators.approaches.refinement_estimation_approach import \
    RefinementEstimationApproach
from predicators.envs.narrow_passage import NarrowPassageEnv
from predicators.settings import CFG


def _policy_solves_task(policy, task, simulator):
    """Helper method used in this file, copied from test_oracle_approach.py."""
    traj = utils.run_policy_with_simulator(policy,
                                           simulator,
                                           task.init,
                                           task.goal_holds,
                                           max_num_steps=CFG.horizon)
    return task.goal_holds(traj.states[-1])


def test_refinement_estimation_approach():
    """Tests for RefinementEstimationApproach class."""
    args = {
        "env": "narrow_passage",
        "refinement_estimator": "oracle",
    }
    # Default to 2 train and test tasks, but allow them to be specified in
    # the extra args too.
    if "num_train_tasks" not in args:
        args["num_train_tasks"] = 2
    if "num_test_tasks" not in args:
        args["num_test_tasks"] = 2
    utils.reset_config(args)
    env = NarrowPassageEnv(use_gui=False)
    train_tasks = env.get_train_tasks()
    test_tasks = env.get_test_tasks()
    approach = RefinementEstimationApproach(env.predicates, env.options,
                                            env.types, env.action_space,
                                            train_tasks)
    assert approach.get_name() == "refinement_estimation"
    assert not approach.is_learning_based
    for task in train_tasks:
        policy = approach.solve(task, timeout=500)
        assert _policy_solves_task(policy, task, env.simulate)
    for task in test_tasks:
        policy = approach.solve(task, timeout=500)
        assert _policy_solves_task(policy, task, env.simulate)
