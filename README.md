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
```

### Train model

```python
python train.py
```
