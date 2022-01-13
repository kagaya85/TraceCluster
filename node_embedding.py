from ge import Node2Vec, DeepWalk
import networkx as nx
import json
import sys
sys.path.append(r'./deepwalk')
from deepwalk import __main__


def read_json(file):
    with open(file, 'r', encoding='utf8') as fp:
        dic = json.load(fp)
    return dic


def build_graph(file_name):
    graph = {}
    graph["vertices"] = ["start"]
    graph["edges"] = {}
    error_count = 0
    data = read_json(file_name)
    count = 0
    for trace_id, trace in data.items():
        try:
            print(count, trace_id)
            count += 1
            vertices = trace["vertexs"]
            v_map = {}
            for index, n in vertices.items():
                if index == "0":
                    v_map[index] = n
                    continue
                name = n[1]
                v_map[index] = name
                if name not in graph["vertices"]:
                    graph["vertices"].append(name)
            edges = trace['edges']
            for v, l in edges.items():
                name = v_map[v]
                if name not in graph["edges"]:
                    graph["edges"][name] = {}
                for adj in l:
                    v2 = str(adj["vertexId"])
                    name2 = v_map[v2]
                    if name2 not in graph["edges"][name]:
                        graph["edges"][name][name2] = 1
                    else:
                        graph["edges"][name][name2] += 1
        except:
            print("error")
            error_count += 1
            continue
    print('error_count:', error_count)
    return graph


def make(dic, weighted=False):
    vertices = dic["vertices"]
    count = 0
    v_map = {}
    for v in vertices:
        v_map[v] = count  # key 为 api
        count += 1
    with open("./experiment/vertices_map.json", 'w', encoding='utf8') as fp:
        json.dump(v_map, fp)
    edges = dic["edges"]
    if weighted:
        file = "./experiment/edges_weighted.txt"
    else:
        file = "./experiment/edges.txt"
    with open(file, 'w', encoding='utf8') as fp:
        for f, s in edges.items():
            for ss in s.keys():
                i1 = str(v_map[f])
                i2 = str(v_map[ss])
                if weighted:
                    fp.write(i1 + " " + i2 + " " + str(s[ss]) + "\n")  # 带权重
                else:
                    fp.write(i1 + " " + i2 + "\n")  # 不带权重
    return file


def processEmbeddingFileToJSONFile(node_filename, filename, output_filename):
    mp = {}
    result = {}
    num = []
    with open(node_filename, 'r') as nodef:
        node = json.load(nodef)
        for k, v in node.items():
            mp[v] = k
    with open(filename, 'r') as f:
        f.readline()
        for s in f.readlines():
            tmp = re.split(' |\n', s)
            num.append(int(tmp[0]))
            result[mp[int(tmp[0])]] = [float(k) for k in tmp[1:-1]]
    print(len(result))
    num.sort()
    tmp = 0
    for i, id in enumerate(num):
        if i+tmp != id:
            print(i+tmp)
            tmp += 1
    print(num)
    with open(output_filename, 'w') as of:
        json.dump(result, of)


def node2Vec_embedding(edgelist_filename, output_filename, weighted=False):
    if weighted is False:
        G = nx.read_edgelist(edgelist_filename, create_using=nx.DiGraph(), nodetype=None)
    else:
        G = nx.read_edgelist(edgelist_filename, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = Node2Vec(G, walk_length=20, num_walks=15, p=1, q=1, workers=1, use_rejection_sampling=0)
    model.train(embed_size=20, window_size=5)
    embeddings = model.get_embeddings()
    mp = {}
    with open('./experiment/vertices_map.json', 'r') as f:
        dic = json.load(f)
        for k, v in dic.items():
            mp[str(v)] = k
    ans = {}
    for k, v in embeddings.items():
        ans[mp[k]] = v.tolist()
    with open(output_filename, 'w') as of:
        json.dump(ans, of)


def deepwalk_embedding(edgelist_filename, output_filename):
    # Deepwalk未实现带权重，且对于有向图而言。
    G = nx.read_edgelist(edgelist_filename, create_using=nx.DiGraph(), nodetype=None)  # read graph
    deep_walk = DeepWalk(G, walk_length=20, num_walks=15)
    deep_walk.train(embed_size=20, window_size=5)
    embeddings = deep_walk.get_embeddings()
    mp = {}
    with open('./experiment/vertices_map.json', 'r') as f:
        dic = json.load(f)
        for k, v in dic.items():
            mp[str(v)] = k
    ans = {}
    for k, v in embeddings.items():
        ans[mp[k]] = v.tolist()
    with open(output_filename, 'w') as of:
        json.dump(ans, of)


if __name__ == '__main__':
    file_name = "/tmp/pycharm_project_683/data/preprocessed/trainticket/bert_2022-01-12_10-36-30/0.json"
    weighted = False
    if weighted:
        str = ""
    else:
        str = "_weighted"
    # edgelist_file = './experiment/edges_1.txt'

    graph = build_graph(file_name)
    edgelist_file = make(graph, weighted=weighted)
    node2Vec_embedding(edgelist_filename=edgelist_file,
                       output_filename='./experiment/node_embedding/embedding' + str +'_node2Vec.json', weighted=weighted)
    deepwalk_embedding(edgelist_filename=edgelist_file,
                       output_filename='./experiment/node_embedding/embedding' + str +'_deepwalk.json')
    # # 调用第三方库deepwalk生成结点embedding
    # __main__.process(input="./experiment/edges_1.txt", output="./experiment/embeddingss", format='edgelist',
    #                  representation_size=20, number_walks=80, walk_length=20, window_size=5, undirected=False)
    # # 将deepwalk处理好的原始数据转为json文件，并对应相应api
    # processEmbeddingFileToJSONFile("./experiment/vertices_map.json", "./experiment/embeddingss",
    #                                "./experiment/embeddingss.json")
