# Trace Cluster

A Trace cluster tool based on Graph Neural Network

### 8.31

#### Task

预处理：

- [x] span id 映射
- [x] word embedding
  - [x] 转小写
  - [x] 分隔符
  - [x] url参数替换
- [x] 响应时间归一化：Z-score

Dataset：
<!-- - [ ] 替换 InMemoryDataset -> Dataset -->
- [x] GraphCL Dataset graph enhanced

#### Other

- 思考trace聚类效果的衡量标准

### 9.1
####

- [x] dbscan
- [x] 预处理并行
- [x] 微信数据适配


### 11.18

- [ ] embedding与build graph并行
- [x] 重构目录结构

### 12.7

- [x] 优化输入输出
- [x] 完善Readme

### 12.13
- [x] 提取embedding到单独的文件中