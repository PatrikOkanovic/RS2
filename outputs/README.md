# Outputs

We save the output files from the experiments in this folder.

We provide scripts for parsing the output files and creating `pandas.DataFrame`. We provide a script for parsing: [`parse_timeToAcc_results.py`](../parse_timeToAcc_results.py) and a notebook [`timeToAcc_table.ipynb`](../timeToAcc_table.ipynb).

When executing this reproducibility package, it is possible to observe slight variations in the runtime results compared 
to those reported in our paper. These differences may arise due to variances in hardware configurations, the initialization 
of random seeds, and other factors that influence the execution environment. However, it is important to note that the 
relative order of methods and the gaps between them should remain consistent. Any variations caused by random noise are 
expected to average out over multiple runs.

## Analyzing output files
An output file from the experiments has the following structure:
```train
dataset: CIFAR10, model: ResNet18, selection: UniformNoReplacement, num_ex: 1, epochs: 200, fraction: 0.1, seed: 1, lr: 0.1, save_path: , resume: , device: cuda
...
(should be unordered) subset[:10]: [45573 39021 36044 45885 19497 27415 30257 11455 22672 43982]
Time for subset selection: time_subset
Epoch: [epoch_num][last_minibatch/total_minibatches]       Time time_last_minibatch (time_avg_minibatch)      Loss loss_last_minibatch (loss_avg_minibatch)    Prec@1 acc_last_minibatch (acc_avg_minibatch)
Time.time(): train_time
Test acc: * Prec@1 test_acc
...
len(subset):  num_datapoints
All together took: total_time
```

First line describes a subset of hyperparameters, showing which dataset is used,
`fraction` represents selection ratio _r_, and `selection` states the method used, e.g., Random. 

Then we show the output lines repeated for each epoch.
Line `(should be unordered) subset[:10]: [45573...]` shows first ten indices selected
for the current round for training.
The explanation for the placeholders are the following:

- `time_subset`: time needed for subset selection
- `epoch_num`: current epoch
- `last_minibatch`: index of the last minibatch starting from 0 to 
- `total_minibatches`: number of total minibatches
- `time_last_minibatch`: time needed for training on the last minibatch
- `time_avg_minibatch`: average time of training on all the minibatches in the current round
- `loss_last_minibatch`: training loss for the last minibatch
- `loss_avg_minibatch`: average training loss on all the minibatches in the current round
- `acc_last_minibatch`: training accuracy on the last minibatch
- `acc_avg_minibatch`: average training accuracy on all the minibatches in the current round
- `test_acc`: test accuracy after training on the subset for the current round


Finally, last two lines show number of datapoints that is used per-round for training in `num_datapoints`, and `total_time`
measures time needed for subset selection, training on all epochs, and evaluating the performance all together.