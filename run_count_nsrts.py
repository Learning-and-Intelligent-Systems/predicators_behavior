import glob
import dill as pkl
import re

paths_to_approaches = ["/home/wbm3/Dropbox (MIT)/predicators_ignore_effs_ijcai23/experiment_data/aaai_domains/saved_approaches/*.NSRTs",
    "/home/wbm3/Dropbox (MIT)/predicators_ignore_effs_ijcai23/experiment_data/behavior_domains/saved_approaches/*/*.NSRTs"]
print("Running...")
print()
results = {"pnad_search": [],  "cluster_and_search": [], "cluster_and_intersect": [], "pred_error": [], "cluster_and_search_random": []}
for path in paths_to_approaches:
    for filename in glob.glob(path):
        exp_name = filename.split("__")[-1].split(".")[0]
        try:
            with open(filename, 'rb') as f:
                data = pkl.load(f)
                nsrt_names = re.findall(r"\bNSRT-\w+", str(data))
            if "50demo" in filename:
                if "pnad_search" in exp_name or "pnadsearch" in exp_name:
                    results["pnad_search"].append(len(nsrt_names))
                if "cluster_and_search" in exp_name and "random" not in exp_name:
                    results["cluster_and_search"].append(len(nsrt_names))
                if "cluster_and_intersect" in exp_name:
                    results["cluster_and_intersect"].append(len(nsrt_names))
                if "pred_error" in exp_name:
                    results["pred_error"].append(len(nsrt_names))
                if "cluster_and_search" in exp_name and "random" in exp_name:
                    results["cluster_and_search_random"].append(len(nsrt_names))
                    print(exp_name)
                
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