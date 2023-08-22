import pickle
import numpy as np
import pandas as pd

indices = range(10)
tasks = ["locking_every_door", "throwing_away_leftovers", "organizing_file_cabinet", "unpacking_suitcase", "collecting_aluminum_cans", "putting_leftovers_away", "re-shelving_library_books", "locking_every_window", "boxing_books_up_for_storage", "polishing_furniture"]
scenes = ["Pomaria_0_int", "Ihlen_1_int", "Pomaria_0_int", "Ihlen_1_int", "Benevolence_2_int", "Ihlen_1_int", "Pomaria_1_int", "Rs_int", "Benevolence_1_int", "Rs_int"] 

total_oracle = 0
total_lifelong = 0

index = pd.MultiIndex.from_product([['oracle', 'lifelong'], tasks], names=['approach', 'task'])
results_df = pd.DataFrame(index=index, columns=['time_solved', 'solved', 'samples', 'time_failed', 'failed', 'time_all'])

for idx, task, scene in zip(indices, tasks, scenes):
    with open(f'results_timing/oracle/behavior__lifelong_sampler_learning_mix__0______{idx}-{task}-{scene}-lifelong__0.pkl', 'rb') as f:
        oracle = pickle.load(f)['results']
    num_solved_oracle = oracle['num_solved']
    sampling_time_oracle = oracle['avg_sampling_time']
    samples_solved_oracle = oracle['avg_num_samples']
    samples_unsolved_oracle = oracle['avg_num_samples_failed']
    num_unsolved_oracle = oracle['num_unsolved']
    sampling_time_unsolved_oracle = oracle['avg_sampling_time_failed']
    avg_time_oracle = (((sampling_time_oracle * num_solved_oracle) if num_solved_oracle > 0 else 0) + ((sampling_time_unsolved_oracle * num_unsolved_oracle ) if num_unsolved_oracle > 0 else 0)) / (num_solved_oracle + num_unsolved_oracle)
    results_df.loc[('oracle', task), 'time_solved'] = sampling_time_oracle
    results_df.loc[('oracle', task), 'solved'] = num_solved_oracle
    results_df.loc[('oracle', task), 'samples'] = (((samples_solved_oracle * num_solved_oracle) if num_solved_oracle > 0 else 0) + ((samples_unsolved_oracle * num_unsolved_oracle) if  num_unsolved_oracle > 0 else 0)) / (num_solved_oracle + num_unsolved_oracle)
    results_df.loc[('oracle', task), 'time_failed'] = sampling_time_unsolved_oracle
    results_df.loc[('oracle', task), 'failed'] = num_unsolved_oracle
    results_df.loc[('oracle', task), 'time_all'] = avg_time_oracle
    
    try:
        with open(f'results_timing/lifelong/behavior__lifelong_sampler_learning_mix__0______{idx}-{task}-{scene}-lifelong__0.pkl', 'rb') as f:
            lifelong = pickle.load(f)['results']
        num_solved_lifelong = lifelong['num_solved']
        sampling_time_lifelong = lifelong['avg_sampling_time']
        num_unsolved_lifelong = lifelong['num_unsolved']
        sampling_time_unsolved_lifelong = lifelong['avg_sampling_time_failed']
        avg_time_lifelong = (((sampling_time_lifelong * num_solved_lifelong) if num_solved_lifelong > 0 else 0) + ((sampling_time_unsolved_lifelong * num_unsolved_lifelong ) if num_unsolved_lifelong > 0 else 0)) / (num_solved_lifelong + num_unsolved_lifelong)
        samples_solved_lifelong = lifelong['avg_num_samples']
        samples_unsolved_lifelong = lifelong['avg_num_samples_failed']

        results_df.loc[('lifelong', task), 'time_solved'] = sampling_time_lifelong
        results_df.loc[('lifelong', task), 'solved'] = num_solved_lifelong
        results_df.loc[('lifelong', task), 'samples'] = (((samples_solved_lifelong * num_solved_lifelong) if num_solved_lifelong > 0 else 0) + ((samples_unsolved_lifelong * num_unsolved_lifelong) if  num_unsolved_lifelong > 0 else 0)) / (num_solved_lifelong + num_unsolved_lifelong)
        results_df.loc[('lifelong', task), 'time_failed'] = sampling_time_unsolved_lifelong
        results_df.loc[('lifelong', task), 'failed'] = num_unsolved_lifelong
        results_df.loc[('lifelong', task), 'time_all'] = avg_time_lifelong
    except:
        pass

    results_df.replace([np.inf], np.nan, inplace=True)

print(results_df)
print(results_df.groupby(level=0).mean())