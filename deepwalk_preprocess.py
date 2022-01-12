import json
import random
import re
import sys
import networkx as nx
from gensim.models import Word2Vec
sys.path.append(r'./deepwalk')
import deepwalk


class DeepWalker:
    def __init__(self, G):
        """
        :param G:
        """
        self.G = G

    def deep_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_neighbors = list(self.G.neighbors(cur))
            if len(cur_neighbors) > 0:
                walk.append(random.choice(cur_neighbors))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        nodes = list(self.G.nodes())
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deep_walk(
                    walk_length=walk_length, start_node=v))
        return walks


# def chooseNeighbor(v, graphs, layers_alias, layers_accept, layer):
#     v_list = graphs[layer][v]
#
#     idx = alias_sample(layers_accept[layer][v], layers_alias[layer][v])
#     v = v_list[idx]
#
#     return v


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = DeepWalker(graph)
        self.sentences = self.walker.simulate_walks(num_walks, walk_length)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings


def read_json(file):
    with open(file, 'r', encoding='utf8')as fp:
        dic = json.load(fp)
    return dic


def build_graph():
    graph = {}
    graph["vertices"] = ["start"]
    graph["edges"] = {}
    error_count = 0
    file_name = "/tmp/pycharm_project_683/data/preprocessed/trainticket/bert_2022-01-12_10-36-30/0.json"
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
    print(error_count)
    return graph


def make(dic):
    vertices = dic["vertices"]
    count = 0
    error_count = 0
    v_map = {}
    for v in vertices:
        v_map[v] = count                # key 为 api
        count += 1
    with open("./experiment/vertices_map.json", 'w', encoding='utf8')as fp:
        json.dump(v_map, fp)
    edges = dic["edges"]
    file1 = "./experiment/edges_1.txt"
    file2 = "./experiment/edges_2.txt"
    with open(file1, 'w', encoding='utf8')as fp1:
        with open(file2, 'w', encoding='utf8')as fp2:
            for f, s in edges.items():
                for ss in s.keys():
                    i1 = str(v_map[f])
                    i2 = str(v_map[ss])
                    fp1.write(i1 + " " + i2 + "\n")                         # 不带权重(edge_1)
                    fp2.write(i1 + " " + i2 + " " + str(s[ss]) + "\n")      # 带权重(edge_2)


def processEmbeddingFileToJSONFile(node_filename, filename, output_filename):
    nodeEmbedding = {}
    mp = {}
    result = {}
    with open(node_filename, 'r') as nodef:
        node = json.load(nodef)
        for k, v in node.items():
            mp[v] = k
    with open(filename, 'r') as f:
        f.readline()
        for s in f.readlines():
            tmp = re.split(' |\n', s)
            result[mp[int(tmp[0])]] = [float(k) for k in tmp[1:-1]]
    with open(output_filename, 'w') as of:
        json.dump(result, of)


if __name__ == "__main__":
    # processEmbeddingFileToJSONFile("./experiment/vertices_map.json", "./experiment/embeddings", "./experiment/embeddings.json")
    # sys.exit(0)
    # make()
    # exit(0)
    # edges = [['a', 'b', 3], ['c', 'd', 4], ['e', 'f', 5], ['a', 'c', 6]]
    # G = nx.DiGraph()
    #
    # for edge in edges:
    #     G.add_edge(edge[0], edge[1], weight=edge[2])
    gra = build_graph()
    make(gra)
    G = nx.read_edgelist('./experiment/edges_1.txt', create_using=nx.DiGraph(), nodetype=None)  # read graph
    # 调用第三方库deepwalk生成结点embedding
    deepwalk.process(input="./experiment/edges_1.txt", output="./experiment/embeddingss", format="edgelist", representation_size=20, number_walks=15, walk_length=20, window_size=5, undirected=False)
    # 将deepwalk处理好的原始数据转为json文件，并对应相应api
    processEmbeddingFileToJSONFile("./experiment/vertices_map.json", "./experiment/embeddingss", "./experiment/embeddingss.json")

    # deep_walk = DeepWalk(G, walk_length=10, num_walks=80)
    # deep_walk.train(embed_size=20, window_size=5)
    # embeddings = deep_walk.get_embeddings()
    # for key, value in embeddings.items():
    #     embeddings[key] = value.tolist()
    # print(embeddings)
    # with open("./experiment/embedding.json", 'w', encoding='utf8')as fp:
    #     json.dump(embeddings, fp)
