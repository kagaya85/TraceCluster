# Trace Cluster

A mircoservice trace cluster tool based on Graph Neural Network

## Dependencies

Please check [requirements.txt](./requirements.txt)

## How to use

### Preprocess

Data preprocess phase will output preprocessed file under `data/preprocessed` directory.

**Notice:** Edit `preprocess.py` file and add raw data path in variant `data_path_list`. For wechat dataset, you need to edit the `mm_data_path_list` for call graph file and `mm_trace_root_list` for click stream file

Run script:

```shell
# use defalut skywalking dataset
python preprocess.py
```

Arguments:
```
--embedding glove
use glove embedding, default bert

--normalize zscore
use z-score normalize, default minmax

--cores [number]
config parallel processing core numbers, default depends on the mechine cores

--max-num [number]
set max saved trace number per-file to control the preprocessed file size, default number 100000

--wechat
use wechat dataset, default skywalking dataset

	--use-request
	use request to replace the ossid/cmdid to ossname/cmdname by querying cmdb, only avalibale for `--wechat`
```

When use wechat dataset, also need to add a `secrets` directory to includes `api.yaml` for cmdb api URL and `cache.json` for service name local cache file, cache file will be updated when using `--use-request` arguments.

### Training

Training phase, `TraceClusterDataset` will check the `data/preprocessed` directory automatically and use the newest changed file. Before training model, it will save processed file under `data/processed` as pytorch `pt` file, you may need to check [this](#clean-up) to know how to clean up when change dataset.

Run script:
```python
# use the newest raw preprocessed data file
python train.py
```

Arguments:
```
--wechat
use the newest wechat raw preprocessed datafile

--dataset [dirpath]
use other preprocessed data dirpath, eg. /data/TraceCluster/preprocessed
```
Please check the source code file(train.py), to see other training auguments

### Cluster

Cluster phase can use multiple cluster methods on `TraceClusterDataset`, it will output a T-SNE image under `img`.

Run script:
```shell
# run cluster method on TraceClusterDataset
# support DBSCAN/CEDAS/DenStream/CluStream.py
python [cluster_method_name].py
```

### Clean Up

Use script to clean up processed data

Run commands:
```shell
# use this command when first use
chmod u+x ./tool/cleanup.sh

# and use this script to clean up processed data and weights files
./tool/cleanup.sh
```