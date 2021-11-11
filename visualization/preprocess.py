from py2neo import Node, Relationship, Graph, Subgraph
from tqdm import tqdm
import json

preprocessed_file = '../data/preprocessed/wechat/2021-11-10_18-18-19.json'

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
    print("start insert to Neo4j...")
    for traceid, trace in tqdm(data.items()):
        # insert edges and nodes
        nodes = []
        relationships = []
        for fromid, edges in trace["edges"].items():
            for edge in edges:
                sub = edge['peer'].split('/', 1)

                from_node = Node(
                    node_label, id=edge['parentSpanId'],
                    traceid=traceid, service=sub[0], name=sub[1]
                )

                has_node = False
                for node in nodes:
                    if node['traceid'] == from_node['traceid'] and node['id'] == from_node['id']:
                        has_node = True
                        from_node = node
                        break

                if not has_node:
                    nodes.append(from_node)

                to_node = Node(
                    node_label, id=edge['vertexId'],
                    traceid=traceid, service=edge['service'], name=edge['operation']
                )

                has_node = False
                for node in nodes:
                    if node['traceid'] == to_node['traceid'] and node['id'] == to_node['id']:
                        has_node = True
                        to_node = node
                        break

                if not has_node:
                    nodes.append(to_node)

                call = Relationship(
                    from_node, edge_label, to_node,
                    from_cmd_id=edge['parentSpanId'], to_cmd_id=edge['spanId'],
                    start_time=edge['startTime'], duration=edge['duration'],
                    is_error=edge['isError']
                )
                relationships.append(call)

        graph.create(Subgraph(nodes, relationships))


if __name__ == "__main__":
    main()
