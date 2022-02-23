import torch
import json
from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd

from utils import read_json
from xcluster.src.python.xcluster.models.PNode import PNode
from xcluster.src.python.xcluster.utils.Graphviz import Graphviz
from torch_geometric.loader import DataLoader
from simclr import SIMCLR
from aug_dataset_mem import TraceDataset
from torch.utils.data import Subset



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_dataset(embedding, dataset_path):
    global traceID_list
    traceID_fr = open('./newData/test_traceID.txt', 'r')
    S = traceID_fr.read()
    traceID_fr.close()
    traceID_list = [traceID for traceID in S.split(', ')]

    idx_fr = open('./newData/test_idx_url_y.txt', 'r')
    S = idx_fr.read()
    idx_fr.close()
    test_idx = [int(index_item) for index_item in S.split(', ')]

    dataset = []

    if embedding == "original":
        print("embedding method:", embedding)

        traces = read_json('./newData/preprocessed/data.json')

        # Get the traces
        traces_data = {}
        api_class = {}
        for traceID in traceID_list:
            traces_data[traceID] = traces[traceID]
            api_class[traceID] = "abnormal" if traces[traceID]["abnormal"]==1 else "normal"

        # with open(path, 'r') as f:
        #     traces_data = json.load(f)

        print("Finish reading traces.")

        # 所有操作的字母表
        alphabet = set()

        # 第一次遍历：得到字母表
        for trace_id, trace in traces_data.items():
            for start_vertex_id, edges in trace["edges"].items():
                for edge in edges:
                    if edge["operation"]:
                        alphabet.add(edge["operation"])

        print("Finish building alphabet, length: %d" % len(alphabet))

        # 第二次遍历：得到每个trace的operation列表，计算字母表中每个操作出现的次数，作为每个trace对应的array
        for trace_id, trace in traces_data.items():
            # 统计trace中每个operation
            c = []
            for start_vertex_id, edges in trace["edges"].items():
                for edge in edges:
                    c.append(edge["operation"])

            vector = []
            for a in alphabet:
                vector.append(c.count(a))

            dataset.append((np.array(vector), api_class[trace_id], trace_id))

        print("Finish building dataset, length: %d" % len(dataset))
    
    elif embedding == "ourMethod":
        print("embedding method:", embedding)

        # init dataset
        dataset_all = TraceDataset(root=dataset_path, aug='none')    # shuffle
        print("Finish reading traces.")

        # load the pre-trained model
        with open(args.model_path + 'train_info.json', "r") as f:  # file name not list
            model_info = json.load(f)
            num_layers = model_info['num_layers']
            output_dim = model_info['output_dim']
            # num_node_features = model_info['num_node_features']
            # num_edge_features = model_info['num_edge_features']
            gnn_type = model_info['gnn_type']
            pooling_type = model_info['pooling_type']
            aug = model_info['aug']
            train_idx = model_info['train_idx']
            # test_idx = model_info['test_idx']
        print("Finish loading model info.")

        eval_dataset = Subset(dataset_all, test_idx)

        # init dataloader
        dataloader = DataLoader(eval_dataset, batch_size=1)    # eval_dataset

        model = SIMCLR(num_layers=num_layers, input_dim=dataset_all.num_node_features, output_dim=output_dim,
                   num_edge_attr=dataset_all.num_edge_features, gnn_type=gnn_type,
                   pooling_type=pooling_type).to(device)
        model.load_state_dict(torch.load(args.model_path + '{}_{}'.format(gnn_type, pooling_type) + ".model"))
        model.eval()
        print("Finish loading model.")

        for data in tqdm(dataloader):
            data = data.to(device)
            vector = model(data.x, data.edge_index, data.edge_attr, data.batch).tolist()[0]
            dataset.append((np.array(vector), "abnormal" if data[0].y.cpu().numpy()[0]==1 else "normal", data[0].trace_id))
        print("Finish building dataset, length: %d" % len(dataset))

    return dataset


def arguments():
    parser = argparse.ArgumentParser(description="PERCH argumentes")
    parser.add_argument('--embedding', dest='embedding',
                        help='select the embedding method of traces, eg. original and ourMethod', type=str, default='original')
    parser.add_argument('--dataset', dest='dataset',
                        help='use other preprocessed data dirpath, eg. ./newData or ./newData/outfiles2022-02-13_21-04-20', default="./newData")
    parser.add_argument('--model_path', dest='model_path',
                        default='./newData/model/', help='weights save path')

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()

    dataset = create_dataset(embedding=args.embedding, dataset_path=args.dataset)

    root = PNode(exact_dist_thres=10)

    for i, pt in enumerate(dataset):
        root = root.insert(pt, collapsibles=None, L=float('inf'))

    print("Finish creating cluster tree.")
    Graphviz.write_tree('./WSET/tree.dot', root)

    results = {}
    # 实验一：计算每个trace被采样的概率
    for leaf in root.leaves():
        trace_id = leaf.pts[0][2]
        label = leaf.pts[0][1]
        p = 1
        n = leaf
        while n is not None:
            p = p / (len(n.siblings()) + 1)
            n = n.parent
        # If p is no less than the random number, PERCH samples the trace
        if p >= np.random.uniform(0, 1):
            # Sample the trace
            sample_res = "Sample"
        # Otherwise
        else:
            # Drop the trace
            sample_res = "Drop"
        results[trace_id] = (label, p, sample_res)

    # 记录实验结果 trace_id    label    p    sample_res
    # open test result
    time_str = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    res_f = open('./WSET/result_PERCH_' + args.embedding + '_' + time_str + '.txt', 'w')
    for traceID in traceID_list:
        res_content = traceID + '\t' + results[traceID][0] + '\t' + str(results[traceID][1]) + '\t' + results[traceID][2] + '\n'
        res_f.write(res_content)
    res_f.close()

    print("Done !")