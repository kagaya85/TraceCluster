# Trace Cluster

A mircoservice trace cluster tool based on Graph Neural Network

## Dependencies

please check [requirements.txt](./requirements.txt)

## How to use

### Preprocess

edit file path variant `data_path_list` in preprocess.py

run script:

```shell
# use skywalking data
python preprocess.py
```

arguments:
```
--wechat
use wechat data

--embedding glove
use globe embedding (default bert)

--normalize zscore
use z-score normalize (default minmax)
```

### Training

run script:
```python
# use the newest raw preprocessed datafile
python train.py
```

arguments:
```
--wechat
use the newest wechat raw preprocessed datafile

--dataset filename
use other data file of filename
```
please check the source code file(train.py), to see other training auguments

### Cluster

run script:
```shell
# run cluster methof of TraceClusterDataset
# support DBSCAN/CEDAS/DenStream/CluStream.py
python cluster_method_name.py
```

### CleanUp

run commands:
```shell
# use this command when first use
chmod u+x ./tool/cleanup.sh

# and use this script
./tool/cleanup.sh
```