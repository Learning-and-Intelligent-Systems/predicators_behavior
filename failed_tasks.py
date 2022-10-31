import glob

tasks = {"sorting-books": [], "throwing-away-leftovers": [], "re-shelving-library-books": []}
for filename in glob.glob("logs/behavior*"):
    print("#"*30)
    print(filename)
    bad_ids = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Task Num" in line:
                # print(i)
                # print(line.replace("\n", ""))
                # print(lines[i+1].replace("\n", ""))
                # print(lines[i+2])
                task_instance_id = int(line.split(": ")[-1])
                bad_ids.append(task_instance_id)
    print(bad_ids)
    for key in tasks.keys():
        if key in filename:
            tasks[key].append(bad_ids)

for task in tasks.items():
    all_ids = []
    remove_ids = []
    for i in range(30):
        i_in = True
        for id_list in task[1]:
            if i not in id_list:
                i_in = False
            all_ids += id_list
        if i_in:
            remove_ids.append(i)
    all_ids = set(all_ids)
    print(task[0], remove_ids)
