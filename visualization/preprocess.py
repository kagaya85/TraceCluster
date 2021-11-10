from os import stat
from time import daylight
from py2neo import Node, Relationship, Graph
import json

preprocessed_file = '../data/preprocessed/wechat/2021-11-10_11-49-49.json'

node_label = 'operation'
edge_label = 'call'


def main():
    # load data
    f = open(preprocessed_file, 'r')
    print(f"load preprocessed from {preprocessed_file}")
    data = json.load(f)
    f.close()

    graph = Graph('http://localhost:7474',
                  username='neo4j', password='password')

    graph.delete_all()

    # insert data
    for traceid, trace in data.items():
        # insert edges and nodes
        sub = []
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
                    from_node, 'CALL', to_node,
                    start_time=edge['startTime'], duration=edge['duration'],
                    is_error=edge['isError']
                )
                sub.append(from_node, to_node, call)

        graph.create(py2neo.Subgraph(sub))


if __name__ == "__main__":
    main()
