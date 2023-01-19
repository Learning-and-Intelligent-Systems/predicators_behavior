import glob
import dill as pkl
import re

paths_to_approaches = ["/home/wbm3/Dropbox (MIT)/predicators_ignore_effs_ijcai23/experiment_data/behavior_domains/saved_approaches/*/*.NSRTs", 
    "/home/wbm3/Dropbox (MIT)/predicators_ignore_effs_ijcai23/experiment_data/aaai_domains/saved_approaches/*.NSRTs"]
print("Running...")
for path in paths_to_approaches:
    for filename in glob.glob(path):
        exp_name = filename.split("__")[-1].split(".")[0]
        try:
            with open(filename, 'rb') as f:
                data = pkl.load(f)
                nsrt_names = re.findall(r"\bNSRT-\w+", str(data))
            print(exp_name, len(nsrt_names))
        except Exception as e:
            print(exp_name, e)
            import ipdb; ipdb.set_trace()