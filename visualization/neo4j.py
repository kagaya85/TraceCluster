from py2neo import Node, Relationship, Graph, Subgraph
from tqdm import tqdm
import json

preprocessed_file = '../data/preprocessed/wechat/2021-11-15_16-41-38.json'

node_label = 'Operation'
edge_label = 'Call'


def main():
    # load data
    f = open(preprocessed_file, 'r')
    print(f"load preprocessed data from {preprocessed_file}")
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
                    node_label, id=fromid,
                    name=sub[0], spanid=edge['parentSpanId'],
                    traceid=traceid, service=sub[0], operation=sub[1]
                )

                n = check_node_exists(from_node, nodes)
                if n == None:
                    nodes.append(from_node)
                else:
                    from_node = n

                to_node = Node(
                    node_label, id=edge['vertexId'],
                    name=edge['service'], spanid=edge['spanId'],
                    traceid=traceid, service=edge['service'], operation=edge['operation']
                )

                n = check_node_exists(to_node, nodes)
                if n == None:
                    nodes.append(to_node)
                else:
                    to_node = n

                call = Relationship(
                    from_node, edge_label, to_node,
                    service=edge['service'], peer=edge['peer'],
                    start_time=edge['startTime'], duration=edge['duration'],
                    is_error=edge['isError']
                )
                relationships.append(call)

        graph.create(Subgraph(nodes, relationships))


def check_node_exists(target, node_list):
    '''
    用于判断结点是否已经存在在图中，可以修改判断方法来调整trace中结点的粒度（接口、服务）
    '''
    for node in node_list:
        if node['traceid'] == target['traceid'] \
                and node['service'] == target['service']:
            # and node['operation'] == target['operation']:
            return node

    return None


if __name__ == "__main__":
    main()
