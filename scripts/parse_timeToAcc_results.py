"""
This file parses the results of the timeToAcc experiments and creates a DataFrame with the results.

For ImageNet for fair comparison across methods, in order to mitigate runtime variance from using a shared cluster, we calculate the minimum optimal timeToAcc per method.
For more details on the above calculateion read the paper Appendix or 'imagenet tta result' section of `visualize_results.ipynb`
"""

DATASET="CIFAR10" # ImageNet
OUT_PATH = f"outputs/timeToAcc-{DATASET}/"

from tqdm import tqdm
import statistics
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

def get_test_acc_per_epoch(file_names, descriptors, seeds=3):
    accs, epochs, time, loss, modes, avg_time, std_time, avg_acc, std_acc, median_time = [], [], [], [], [], [], [], [], [], []
    current_avg_time = []
    current_median_time = []
    current_std_time = []
    current_avg_acc = []
    current_min_acc = []
    current_max_acc = []
    current_std_acc = []
    pretrain_times = []
    offset = 0
    pretrain_coreset_time = 0
    # check min,med,avg per batch and pretrain
    times_per_batch = []

    min_per_batch_time = 0
    # per_method_pretrain_dict, per_batch_time_dict = {}, {}
    for file_path, current_mode in zip(file_names, descriptors):
        # print(f"file_path = {file_path}")
        with open(file_path) as f:
            lines = f.readlines()
        epoch = 1
        current_time = []
        current_acc = []
        for line in lines:
            if line.startswith("Test acc:"):
                dubrovnik = line.split()
                accs.append(float(dubrovnik[-1]))
                current_acc.append(float(dubrovnik[-1]))
                epochs.append(epoch)
                epoch += 1
                modes.append(current_mode)
                offset += 1
            elif line.startswith("Epoch:"):
                dubrovnik = line.split()
                train_loss = float(dubrovnik[-4][1:-1])
                loss.append(train_loss)
                num_batches = dubrovnik[1]
                num_batches = int(num_batches[num_batches.find("/")+1 : num_batches.find("]", num_batches.find("/")+1)])
                batch_time = float(dubrovnik[4][1:-1])
                times_per_batch.append(batch_time)
                # print(f"({file_path}) Epoch, time = {batch_time}")
                min_per_batch_time = min(min_per_batch_time, batch_time)
                if len(current_time) == 0 and pretrain_coreset_time != 0:
                    current_time.append(num_batches * batch_time + pretrain_coreset_time)
                else:
                    current_time.append(num_batches * batch_time)
            elif line.startswith("Time for subset selection:"):
                dubrovnik = line.split()
                pretrain_coreset_time = float(dubrovnik[-1])
                pretrain_times.append(pretrain_coreset_time)
                # print(f"({file_path}) Time for subset selection: = {float(dubrovnik[4][1:-1])}")
        current_time = np.array(current_time).cumsum()
        time.extend(list(current_time))
        current_avg_time.append(current_time)
        current_std_time.append(current_time)
        current_median_time.append(current_time)
        current_avg_acc.append(current_acc)
        current_std_acc.append(current_acc)
        current_min_acc.append(current_acc)
        current_max_acc.append(current_acc)


    # eval_df only used for imagenet in order to get statistics for min,avg,median time per batch and time for pretraining/subset selection. 
    table = {
        'file_path': str(file_names[0].split('/')[-1]), # any file name from batch (batch is the different seeds)
        'min per batch': min(times_per_batch),
        'avg per batch': statistics.mean(times_per_batch),
        'median per batch': statistics.median(times_per_batch),
        'min pretrain': min(pretrain_times), 
        'avg pretrain':statistics.mean(pretrain_times),
        'median pretrain':statistics.median(pretrain_times),
    }
    eval_df = pd.DataFrame.from_dict(table, orient='index').T

    # pretrain_avg_time = np.average(np.array(pretrain_times), axis=0)
    # pretrain_std_time = np.std(np.array(pretrain_times), axis=0)
    
    current_avg_time = np.average(np.array(current_avg_time), axis=0)
    current_std_time = np.std(np.array(current_std_time), axis=0)
    current_median_time = np.median(np.array(current_median_time), axis=0)

    current_avg_acc = np.average(np.array(current_avg_acc), axis=0)
    current_std_acc = np.std(np.array(current_std_acc), axis=0)
    current_min_acc = np.min(np.array(current_min_acc), axis=0)
    current_max_acc = np.max(np.array(current_max_acc), axis=0)

    avg_acc.extend(list(np.tile(current_avg_acc, reps=seeds)))
    std_acc.extend(list(np.tile(current_std_acc, reps=seeds)))
    avg_time.extend(list(np.tile(current_avg_time, reps=seeds)))
    std_time.extend(list(np.tile(current_std_time, reps=seeds)))
    median_time.extend(list(np.tile(current_median_time, reps=seeds)))
    result = {"Accuracy (avg %)": current_avg_acc, "Accuracy (min %)": current_min_acc, "Accuracy (max %)": current_max_acc, "Accuracy (std %)": current_std_acc,"Epochs": epochs[:len(epochs)//3], "Method": modes[:len(epochs)//3], "Average time (s)": current_avg_time, "Std time (s)": current_std_time, "Median time (s)": current_median_time}
    df = pd.DataFrame.from_dict(result, orient='index')

    return result, df, eval_df



def preprocess(files_in_dir):
    batches = {}
    for file_path in files_in_dir:
        file = file_path.split("/")[-1]
        split_file = file.split("_")
        prefix = split_file[1:-1]
        str_prefix = '_'.join(prefix)

        if str_prefix in batches:
            batches[str_prefix].append(file_path)
        else:
            batches[str_prefix] = [file_path]

    batch_list = list(batches.values())
    desc = []
    for batch in batch_list:
        tmp = []
        for f in batch:
            file = f.split("/")[-1]
            split_file = file.split("_")
            dataset = split_file[4]
            model = split_file[5]
            method = split_file[6]
            fraction = float(split_file[-2])
            seed = int(split_file[-1].split('.')[0])
            prefix = split_file[1:-1]
            tmp.append('_'.join(split_file[1:]))
        desc.append(tmp)

    return batch_list, desc



def merge_and_process_imagenet_dfs(final_eval_df, final_df, DATASET):
    assert DATASET=="ImageNet"
    df3_imgnet = final_eval_df
    df3_imgnet.set_index('file_path').rename_axis('Method').sort_index()

    imagenet_min_per_batch = df3_imgnet['min per batch'].min()
    df3_imgnet['global_min_per_batch'] = imagenet_min_per_batch
    print(imagenet_min_per_batch)

    df = final_df 
    # NEW Time (s) will hold the minimum potential runtime, reported ImageNet time to acc experiments
    df['NEW Time (s)'] = None
    df['Fraction'] = None
    df['Fraction'] = df['Method'].map(lambda x: '0.'+x.split('.')[-2].split('_')[0])
    df['Fraction'] = df['Fraction'].replace('0.ImageNet', np.nan)
    df['Fraction'] = df['Fraction'].astype(float)
    df['Fraction'] = df['Fraction'].fillna(1.0) # only NA in for FULL method which is frac=1

    df_copy = df.copy(deep=True)
    df3_imgnet_copy = df3_imgnet.copy(deep=True)

    df_copy['Method'] = df_copy['Method'].str.slice(stop=-6)
    df3_imgnet_copy['file_path'] = df3_imgnet_copy['file_path'].str.slice(stop=-6)
    df3_imgnet_copy['file_path'] = df3_imgnet_copy['file_path'].str.split('_', n=1, expand=True)[1]
    merged_df = pd.merge(df_copy, df3_imgnet_copy, left_on='Method', right_on='file_path', how='left')
    merged_df = merged_df.drop('file_path', axis=1)


    cols_to_convert = merged_df.columns[merged_df.columns.get_loc('Fraction')+1:]
    merged_df[cols_to_convert] = merged_df[cols_to_convert].astype(float)

    # imagenet_batch_size = 256
    imagenet_fracs_to_batches_per_epoch = {
        0.01: 51,
        0.05: 250,
        0.1: 501,
        1.0: 5005
    }

    def calculate_new_time(row):
        batches_per_epoch = imagenet_fracs_to_batches_per_epoch.get(row['Fraction'])
        if batches_per_epoch is None:
            return np.nan
        return imagenet_min_per_batch * batches_per_epoch * row['Epochs'] + row['min pretrain']

    def calculate_total_batches(row):
        batches_per_epoch = imagenet_fracs_to_batches_per_epoch.get(row['Fraction'])
        if batches_per_epoch is None:
            return np.nan
        return batches_per_epoch * row['Epochs']

    def calculate_batches_per_epoch(row):
        batches_per_epoch = imagenet_fracs_to_batches_per_epoch.get(row['Fraction'])
        if batches_per_epoch is None:
            return np.nan
        return batches_per_epoch

    # calculate the "NEW Time (s)" column
    merged_df['batch per epoch'] = merged_df.apply(calculate_batches_per_epoch, axis=1)
    merged_df['total batches'] = merged_df.apply(calculate_total_batches, axis=1)

    merged_df['NEW Time (s)'] = merged_df.apply(calculate_new_time, axis=1)
    merged_df = merged_df.dropna() # drops rows if any nan (shouldnt)
    merged_df['NEW Time (s)'] = merged_df['NEW Time (s)'].round().astype(int)

    return merged_df

if __name__ == "__main__":
    
    files_in_dir = [join(OUT_PATH, f) for f in listdir(OUT_PATH) if isfile(join(OUT_PATH, f))]
    batch_list, desc = preprocess(files_in_dir)

    problematic_batches, all_res, all_dfs, eval_tables = [], [], [], []
    for batch, descr in tqdm(zip(batch_list, desc)):
        try:
            assert len(batch) == len(descr) == 3
            batch_res, batch_df, eval_df = get_test_acc_per_epoch(batch, descr, seeds=3)
            all_res.append(batch_res)
            all_dfs.append(batch_df)
            eval_tables.append(eval_df)
        except:
            problematic_batches.append(batch)
            continue

    print(f"problematic_batches = {problematic_batches}")

    final_df = pd.concat(all_dfs, axis=1)
    final_df = final_df.T.reset_index()
    final_df = final_df.astype({'Accuracy (avg %)':float,'Accuracy (std %)':float, 'Epochs': int, 'Average time (s)': int, 'Std time (s)': int})
    print(f"{DATASET}_final_df = {final_df}")
    final_df.to_pickle(f'./outputs/{DATASET}_final_df.pkl')
    
    # output file of problematic_batches
    # with open("problematic_batches.txt", "w") as f:
    #     for item in problematic_batches:
    #         f.write(str(item) + "\n")

    if DATASET=="ImageNet":
        final_eval_df = pd.concat(eval_tables, axis=0)
        print(f"eval_{DATASET}_df = {final_eval_df}")

        merged_df = merge_and_process_imagenet_dfs(final_eval_df, final_df, DATASET)
        merged_df.to_pickle(f'./merged_df_{DATASET}_df.pkl') 

        print(f"./outputs/merged_df_{DATASET}_df = {merged_df}")
    
    print("Finished!")
