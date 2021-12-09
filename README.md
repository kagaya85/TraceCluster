# Trace Cluster

A mircoservice trace cluster tool based on Graph Neural Network

## Dependencies

Please check [requirements.txt](./requirements.txt)

## How to use

### Preprocess

Edit file path variant `data_path_list` in preprocess.py

**Notice:** for wechat dataset, need to edit the `mm_data_path_list` for call graph file and `mm_trace_root_list` for click stream file

Run script:

```shell
# use defalut skywalking dataset
python preprocess.py
```

Arguments:
```
--embedding glove
use globe embedding, default bert

--normalize zscore
use z-score normalize, default minmax

--cores [number]
config parallel processing core numbers, default depends on the mechine cores

--wechat
use wechat dataset, default skywalking dataset

	--use-request
	use request to replace the ossid/cmdid to ossname/cmdname by querying cmdb, only avalibale for `--wechat`
```

When use wechat dataset, also need to add a `secrets` directory to includes `api.yaml` for cmdb api URL and `cache.json` for service name local cache file, cache file will be updated when using `--use-request` arguments.

### Training

`TraceClusterDataset` will check the `data/preprocessed` directory automatically and use the newest changed file

Run script:
```python
# use the newest raw preprocessed data file
python train.py
```

Arguments:
```
--wechat
use the newest wechat raw preprocessed datafile

--dataset [filename]
use other data file of filename
```
Please check the source code file(train.py), to see other training auguments

### Cluster

Run script:
```shell
# run cluster method on TraceClusterDataset
# support DBSCAN/CEDAS/DenStream/CluStream.py
python [cluster_method_name].py
```

### Clean Up

Run commands:
```shell
# use this command when first use
chmod u+x ./tool/cleanup.sh

# and use this script to clean up processed data and weights files
./tool/cleanup.sh
```