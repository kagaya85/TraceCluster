# Trace Cluster

A Trace cluster tool based on Graph Neural Network

## Dependencies

TODO

## How to use

### Preprocess

edit file path variant `data_path_list` in preprocess.py

run script:

```shell
# use skywalking data
python preprocess.py

# use wechat data
python preprocess.py --wechat

# use globe embedding (default bert)
python preprocess.py --embedding glove

# use z-score normalize (default minmax)
python preprocess.py --normalize zscore
```

### Train model

```python
python train.py
```

### Cluster

```shell
python cluster_method_name.py
```