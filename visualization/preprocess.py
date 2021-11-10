from os import stat
from time import daylight
from py2neo import Node, Relationship, Graph, Subgraph
from tqdm import tqdm
import json

preprocessed_file = '../data/preprocessed/wechat/2021-11-10_17-38-57.json'

node_label = 'Operation'
edge_label = 'Call'


def main():
    # load data
    f = open(preprocessed_file, 'r')
    print(f"load preprocessed from {preprocessed_file}")
    data = json.load(f)
    f.close()

    graph = Graph('http://localhost:7474',
                  auth=('neo4j', 'password'))

    graph.delete_all()

    # insert data
    for traceid, trace in tqdm(data.items()):
        # insert edges and nodes
        nodes = []
        relationships = []
        for fromid, edges in trace["edges"].items():
            for edge in edges:
                sub = edge['peer'].split('/', 1)

                from_node = Node(
                    node_label, id=fromid,
                    traceid=traceid, service=sub[0], name=sub[1]
                )
                to_node = Node(
                    node_label, id=edge['vertexId'],
                    traceid=traceid, service=edge['service'], name=edge['operation']
                )

                call = Relationship(
                    from_node, edge_label, to_node,
                    start_time=edge['startTime'], duration=edge['duration'],
                    is_error=edge['isError']
                )
                nodes.extend([from_node, to_node])
                relationships.append(call)

        graph.create(Subgraph(nodes, relationships))


if __name__ == "__main__":
    main()
