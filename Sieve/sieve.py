import torch
import json
import argparse
import numpy as np
import random
import os
from tqdm import tqdm
from STVProcess import embedding_to_vector, load_dataset, process_one_trace

from torch_geometric.loader import DataLoader
import rrcf

from simclr import SIMCLR
from aug_dataset_mem import TraceDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def arguments():
    parser = argparse.ArgumentParser(description="Sieve argumentes")
    parser.add_argument('--embedding', dest='embedding',
                        help='select the embedding method of traces, eg. STV and ourMethod', type=str, default='STV')
    parser.add_argument('--dataset', dest='dataset',
                        help='use other preprocessed data dirpath, eg. ./newData or ./newData/outfiles2022-02-13_21-04-20', default="./newData/outfiles2022-02-13_21-04-20")
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--model_path', dest='model_path',
                        default='./newData/20epoch_CGConv_add_new_anomaly_view_0.5_new_data/', help='weights save path')
    # ========================================
    # Set tree parameters
    # ========================================
    parser.add_argument('--num_trees', type=int, default=2)     # 50
    parser.add_argument('--tree_size', type=int, default=128)
    parser.add_argument('--windowSize_k', type=int, default=50)
    parser.add_argument('--threshold_h', type=int, default=0.3)

    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1,
                        help='batch size')    # 128

    return parser.parse_args()




def main():
    # ========================================
    # Set global parameters
    # ========================================
    isMaintenanceStage = False
    # Create a list to store attention score of each point
    avg_attentionScore = []
    forest = []
    X = []    # before path vector or after

    all_path = []
    all_trace = {}
    normal_trace = {}
    abnormal_trace = {}
    # url_trace = {}




    args = arguments()
    batch_size = args.batch_size
    tree_size = args.tree_size
    num_trees = args.num_trees
    windowSize_k = args.windowSize_k
    threshold_h = args.threshold_h


    if args.embedding == 'STV':
        print("embedding method:", args.embedding)

        dataloader = load_dataset(args.dataset)    # trace list
    
    elif args.embedding == 'ourMethod':
        print("embedding method:", args.embedding)
        
        # dataroot = os.path.join(os.path.dirname(
        #     os.path.realpath(__file__)), '.', 'data')

        # init dataset
        dataset = TraceDataset(root=args.dataset, aug='none')    # shuffle

        # init dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size)

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

        model = SIMCLR(num_layers=num_layers, input_dim=dataset.num_node_features, output_dim=output_dim,
                   num_edge_attr=dataset.num_edge_features, gnn_type=gnn_type,
                   pooling_type=pooling_type).to(device)
        model.load_state_dict(torch.load(args.model_path + '{}_{}'.format(gnn_type, pooling_type) + ".model"))
        model.eval()

    # open test result
    res_f = open('./Sieve/result.txt', 'w')

    print('Start !')
    index = 0
    for data in tqdm(dataloader):
        index = index + 1
        if args.embedding == 'STV':
            res_content = data['trace_id'] + '\t' + str(data['trace_bool']) + '\t' + data['error_trace_type']
            # ========================================
            # Path vector encoder
            # ========================================
            # all_trace 需要在维护阶段每次清空，每次仅保留当前的 trace 信息。这里的 all_trace 也就是 X，下一步需要确定 embedding 的位置 ！！！！！！
            if isMaintenanceStage == True:
                all_trace = {}
            all_path, all_trace, normal_trace, abnormal_trace = process_one_trace(data, all_path, all_trace, normal_trace, abnormal_trace)

        elif args.embedding == 'ourMethod':
            res_content = data[0].trace_id + '\t' + str(data[0].y.numpy()[0]) + '\t' + data[0].root_url
            
            data = data.to(device)

            newNode = model(data.x, data.edge_index, data.edge_attr, data.batch).tolist()[0]    # 每个图的特征均表示为一个 tensor

        # A new trace come ...
        if isMaintenanceStage == False:
            if args.embedding == 'ourMethod':
                X.append(newNode)
            if len(X) == tree_size or len(all_trace) == tree_size:
                # ========================================
                # Path vector encoder
                # ========================================
                if args.embedding == 'STV':
                    X = embedding_to_vector(all_trace, all_path)

                # ========================================
                # Construction stage
                # ========================================
                # Create a forest
                for _ in range(num_trees):
                    tree = rrcf.RCTree(X=X, embedding_method=args.embedding)
                    forest.append(tree)
                isMaintenanceStage = True
        elif isMaintenanceStage == True:
            # ========================================
            # Path vector encoder
            # ========================================
            if args.embedding == 'STV':
                newNode = np.array(embedding_to_vector(all_trace, all_path)[0])

            # Sum attention score of all RRCTs
            scoreSum = 0
            for i, tree in enumerate(forest):
                # ========================================
                # Maintenance stage
                # ========================================
                # Remove the oldest leaf
                tree.forget_point(min(tree.leaves.keys()))
                # Insert new path vector
                # No new dimension
                if len(newNode) == tree.ndim:
                    tree.insert_point(newNode, index=max(tree.leaves.keys())+1)
                # Has new dimension
                else:
                    # Extend leaves
                    for leaf_key, leaf_value in tree.leaves.items():
                        tree.leaves[leaf_key].x = np.append(tree.leaves[leaf_key].x, [-1]*(len(newNode)-len(tree.leaves[leaf_key].x)))
                        tree.leaves[leaf_key].b = tree.leaves[leaf_key].x.reshape(1, -1)
                
                    tree.ndim = len(newNode)

                    # Extend internal nodes
                    tree._get_bbox_top_down(tree.root)

                    # Build a new root
                    node = tree.root
                    parent = node.u
                    leaf = rrcf.Leaf(x=newNode, i=max(tree.leaves.keys())+1, d=0)
                    branch = rrcf.Branch(q=tree.ndim-1, p=-0.5, l=node, r=leaf,
                                         n=(leaf.n + node.n))
                    # Set parent of new leaf and old branch
                    node.u = branch
                    leaf.u = branch
                    # Set parent of new branch
                    branch.u = parent
                    # If a new root was created, assign the attribute
                    tree.root = branch
                    # Increment depths below branch
                    tree.map_leaves(branch, op=tree._increment_depth, inc=1)
                    # Increment leaf count above branch
                    tree._update_leaf_count_upwards(parent, inc=1)
                    # Update bounding boxes
                    tree._tighten_bbox_upwards(branch)
                    # Add leaf to leaves dict
                    tree.leaves[max(tree.leaves.keys())+1] = leaf
            
                # ========================================
                # Calculate attention score
                # ========================================
                if max(tree.leaves[leaf_index].d for leaf_index in tree.leaves.keys()) == 0:
                    scoreSum = scoreSum + 0
                else:
                    scoreSum = scoreSum + max(tree.leaves[leaf_index].d for leaf_index in tree.leaves.keys()) / tree.leaves[max(tree.leaves.keys())].d

            # Final score of a path vector
            scoreAvg = scoreSum / len(forest)

            # ========================================
            # Biased sampler
            # ========================================
            # A sliding window containing k most recent scores and the current score
            if len(avg_attentionScore) == windowSize_k+1:
                avg_attentionScore.pop(0)
            avg_attentionScore.append(scoreAvg)
            # Calculates the variance vark of the past k scores and the variance vark+1 of the k + 1 scores
            if len(avg_attentionScore) == windowSize_k+1:
                Var_k = np.var(avg_attentionScore[:-1])
                Var_k1 = np.var(avg_attentionScore)
                # Difference degree exceeds a threshold h
                if Var_k1 - Var_k > threshold_h:
                    p = 1 / (1 + np.exp(2*np.mean(avg_attentionScore)-avg_attentionScore[-1]))
                # Otherwise ...
                elif Var_k1 - Var_k <= threshold_h:
                    if not(np.any(avg_attentionScore)):
                        # avg_attentionScore is all zeros
                        p = -1
                    else:
                        p = avg_attentionScore[-1] / np.sum(avg_attentionScore)
                # If p is no less than the random number, Sieve samples the trace
                if p >= np.random.uniform(0, 1):
                    # Sample the trace
                    # print("Sample the trace ...")
                    res_content = res_content + '\t' + str(p) + '\t' + "Sample" + '\n'
                    res_f.write(res_content)
                # Otherwise
                else:
                    # Drop the trace
                    # print("Drop the trace ...")
                    res_content = res_content + '\t' + str(p) + '\t' + "Drop" + '\n'
                    res_f.write(res_content)
            # else:
                # Sliding window is not full
                # print("Sliding window is not full, missing {} attention score ...".format(windowSize_k+1-len(avg_attentionScore)))
    
    res_f.close()
    print("Done !")

if __name__ == '__main__':
    main()