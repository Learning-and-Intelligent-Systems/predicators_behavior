import pandas as pd
import numpy as np
import pickle
from itertools import product
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

num_seeds = 4#8
num_iters_task = 5
burnin = 48
num_tasks_per_cycle_list = [48]
# lifelong_approaches = ['distill','retrain-scratch', 'finetune', 'retrain-balanced']
lifelong_approaches = ['retrain-balanced']#, 'retrain-scratch', 'finetune']
model_types = ['geometry+', 'specialized', 'generic']

model_map = {'geometry+': 'Mixture', 'default': 'Hand-crafted', 'specialized': 'Specialized', 'generic': 'Generic'}
approach_map = {'finetune': 'Finetune', 'retrain-balanced': 'Replay', 'retrain-scratch': 'Retrain', '': ''}


task_order=np.array([[2, 8, 4, 9, 1, 6, 7, 3, 0, 5],
                    [2, 9, 6, 4, 0, 3, 1, 7, 8, 5],
                    [4, 1, 5, 0, 7, 2, 3, 6, 9, 8],
                    [5, 4, 1, 2, 9, 6, 7, 0, 3, 8],
                    [3, 8, 4, 9, 2, 6, 0, 1, 5, 7],
                    [9, 5, 2, 4, 7, 1, 0, 8, 6, 3],
                    [8, 1, 7, 0, 6, 5, 2, 4, 3, 9],
                    [8, 5, 0, 2, 1, 9, 7, 3, 6, 4]])

task_list=['boxing_books_up_for_storage', 'collecting_aluminum_cans', 'locking_every_door', 'locking_every_window', 'organizing_file_cabinet', 'polishing_furniture', 'putting_leftovers_away', 're-shelving_library_books', 'throwing_away_leftovers', 'unpacking_suitcase']
train_scene_list=['Benevolence_1_int', 'Benevolence_2_int', 'Pomaria_0_int', 'Rs_int', 'Pomaria_0_int', 'Rs_int', 'Ihlen_1_int',
    'Pomaria_1_int', 'Ihlen_1_int', 'Ihlen_1_int']
num_behavior_tasks = len(task_list)
assert task_order.shape[1] == num_behavior_tasks

for n_tasks  in num_tasks_per_cycle_list:
    successes = {}
    samples = {}
    time = {}
    for model in model_types:
        # for approach in lifelong_approaches:
        if model == 'geometry+':
            approach = 'retrain-balanced'
        else:
            approach = 'retrain-scratch'
        successes[model, approach] = np.full((num_seeds, num_iters_task * num_behavior_tasks), np.nan)
        samples[model, approach] = np.full((num_seeds, num_iters_task * num_behavior_tasks), np.nan)
        time[model, approach] = np.full((num_seeds, num_iters_task * num_behavior_tasks), np.nan)

        final_iter = num_iters_task * num_behavior_tasks
        for seed in range(num_seeds):
            n_iter = 0
            for shuffled_index in range(num_behavior_tasks):
                task_index = task_order[seed, shuffled_index]
                task_name = task_list[task_index] + '-' + train_scene_list[task_index]
                for n_iter_task in range(num_iters_task):
                    if model in ['specialized', 'generic']:
                        fname = f'results_lifelong_5/specialized_{model == "specialized"}/behavior__lifelong_sampler_learning__{seed}______{shuffled_index}-{task_name}-lifelong__{n_iter_task}.pkl'
                    else:
                        fname = f'results_lifelong_5/geometry+/behavior__lifelong_sampler_learning_mix__{seed}______{shuffled_index}-{task_name}-lifelong__{n_iter_task}.pkl'
                    try: 
                        with open(fname, 'rb') as f:
                            r = pickle.load(f)
                        solved = r['results']['num_solved']
                        unsolved = r['results']['num_unsolved']
                        samples_solved = r['results']['avg_num_samples'] * solved if solved > 0 else 0
                        samples_unsolved = r['results']['avg_num_samples_failed'] * unsolved if unsolved > 0 else 0
                        successes[model, approach][seed, n_iter] = solved
                        samples[model, approach][seed, n_iter] = samples_solved + samples_unsolved
                        time[model, approach][seed, n_iter] = r['results']['avg_time']
                    except FileNotFoundError:
                        print(fname)
                        if n_iter < final_iter:
                            final_iter = n_iter
                        break
                    n_iter += 1
        print(f'{approach}, {model}: {final_iter} iters done')
        successes[model, approach][:, final_iter:] = np.nan
        samples[model, approach][:, final_iter:] = np.nan
        time[model, approach][:, final_iter:] = np.nan

    
    successes['default',''] = successes['geometry+', 'retrain-balanced'][:, ::num_iters_task].repeat(num_iters_task, axis=1)
    samples['default',''] = samples['geometry+', 'retrain-balanced'][:, ::num_iters_task].repeat(num_iters_task, axis=1)
    time['default', ''] = time['geometry+', 'retrain-balanced'][:, ::num_iters_task].repeat(num_iters_task, axis=1)

    # for model, approach in zip(['geometry+', 'default'], ['retrain-balanced', '']):
    for model, approach in zip(['geometry+', 'specialized', 'generic', 'default'], ['retrain-balanced', 'retrain-scratch', 'retrain-scratch', '']):
        model_legend = model_map[model]
        approach_legend = approach_map[approach]
        label = model_legend + ('+' + approach_legend if approach_legend else '')
        plt.figure(0)
        plt.plot(successes[model,approach].mean(axis=0), label=label, linewidth=4)
        plt.figure(1)
        plt.plot(samples[model,approach].mean(axis=0).cumsum(axis=0), successes[model,approach].mean(axis=0), label=label, linewidth=4)
        plt.figure(2)
        plt.plot(samples[model,approach].mean(axis=0).cumsum(axis=0), successes[model,approach].mean(axis=0).cumsum(axis=0), label=label, linewidth=4)
        plt.figure(3)
        div = np.full(samples[model, approach].shape[1], n_tasks)
        div[0] = burnin
        plt.plot(samples[model,approach].mean(axis=0).cumsum(axis=0), samples[model,approach].mean(axis=0) / div, label=label, linewidth=4)
        plt.figure(4, figsize=(18,6))
        plt.plot(np.array([0]), np.array([0]), label=f'{model_legend}+{approach_legend}', linewidth=4)



    plt.figure(0)
    plt.legend()
    plt.xlabel('# training iters')
    plt.ylabel('# solved tasks per iter')
    plt.tight_layout()
    plt.savefig(f'results_lifelong_5/avg_solved_per_iter_{n_tasks}_rebuttal.pdf')
    plt.close()
    plt.figure(1)
    plt.legend()
    plt.xlabel('# compute units')
    plt.ylabel('# solved tasks per iter')
    plt.ticklabel_format(axis='x', scilimits=(3,3))
    plt.tight_layout()
    plt.savefig(f'results_lifelong_5/avg_solved_per_sample_{n_tasks}_rebuttal.pdf')
    plt.close()
    plt.figure(2)
    plt.legend(fontsize=14)
    plt.xlabel('# samples')
    plt.ylabel('# cumulative solved')
    # plt.ylim(bottom=0)
    plt.ticklabel_format(axis='x', scilimits=(3,3))
    plt.tight_layout()
    plt.savefig(f'results_lifelong_5/total_solved_{n_tasks}_rebuttal.pdf')
    plt.close()
    plt.figure(3)
    plt.legend()
    plt.xlabel('# compute units')
    plt.ylabel('# samples per attempted task')
    plt.ticklabel_format(axis='x', scilimits=(3,3))
    plt.savefig(f'results_lifelong_5/avg_samples_{n_tasks}_rebuttal.pdf')
    plt.tight_layout()
    plt.close()
    plt.figure(4, figsize=(18,6))
    plt.legend(ncol=2)
    plt.savefig('results_lifelong_5/legend_lifelong_rebuttal.pdf')
    plt.close()
    print({model_approach: samples[model_approach][:, 1:].mean() / n_tasks for model_approach in samples})

    print({model_approach: time[model_approach][:, 1::].mean() / n_tasks for model_approach in time})