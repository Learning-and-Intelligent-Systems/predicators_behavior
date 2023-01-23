import glob
import dill as pkl
import re

results = {"pnad_search": [],  "cluster_and_search": [], "cluster_and_intersect": [], "pred_error": [], "cluster_and_search_random": []}
for filename in glob.glob("saved_approaches/*numNSRTs.pkl"):
    exp_name = filename.split("__")[-1].split(".")[0]
    try:
        with open(filename, 'rb') as f:
            data = pkl.load(f)
        if "satellites" in filename and "simple" in filename:
            if "pnad_search" in exp_name or "pnadsearch" in exp_name:
                results["pnad_search"].append(data)
            if "cluster_and_search" in exp_name and "random" not in exp_name:
                results["cluster_and_search"].append(data)
            if "cluster_and_intersect" in exp_name:
                results["cluster_and_intersect"].append(data)
            if "pred_error" in exp_name:
                results["pred_error"].append(data)
                print(exp_name)
            if "cluster_and_search" in exp_name and "random" in exp_name:
                results["cluster_and_search_random"].append(data)
            
    except Exception as e:
        if "random" in filename and "50demo" in filename:
            print(filename)
            print(exp_name)
            print("error")
            import ipdb; ipdb.set_trace()
        pass

import numpy as np
for key, val in results.items():
    print(key, len(val), np.mean(val), np.std(val))