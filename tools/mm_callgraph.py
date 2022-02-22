from py2neo import Node, Relationship, Graph, Subgraph
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

normal_data_path = '/data/TraceCluster/raw/wechat/trace_mmfindersynclogicsvr/call_graph_2022-01-17_23629.csv'
normal_cs_path = '/data/TraceCluster/raw/wechat/trace_mmfindersynclogicsvr/click_stream_2022-01-17_23629.csv'
data_path = '/data/TraceCluster/raw/wechat/trace_mmfindersynclogicsvr/call_graph_2022-01-18_23629.csv'
cs_path = '/data/TraceCluster/raw/wechat/trace_mmfindersynclogicsvr/click_stream_2022-01-18_23629.csv'

def gen_span_key(span) -> str:
    return '/'.join([str(span['CallerOssID']), str(span['CallerCmdID']), str(span['CalleeOssID']), str(span['CalleeCmdID'])])


def generate_graph(clickstream: DataFrame, spandata: DataFrame) -> list:
    traceId_list = []
    graph = {}
    for _, root in clickstream.iterrows():
        traceId_list.append(root['GraphIdBase64'])

    data = spandata.groupby(['GraphIdBase64'])
    print('calculating graph...')
    for traceId, spans in tqdm(data):
        if traceId not in traceId_list:
            continue

        for _, span in spans.iterrows():
            cost_time = span['CostTime']
            exist_span = graph.get(gen_span_key(span))

            if exist_span != None:
                exist_span['count'] += 1
                exist_span['cost_time_total'] += cost_time
                if cost_time > exist_span['cost_time_max']:
                    exist_span['cost_time_max'] = cost_time
                if cost_time < exist_span['cost_time_min']:
                    exist_span['cost_time_min'] = cost_time
                exist_span['not_zero_network_ret_count'] += 0 if span['NetworkRet'] == 0 else 10
                exist_span['not_zero_service_ret_count'] += 0 if span['ServiceRet'] == 0 else 1
            else:
                graph[gen_span_key(span)] = {
                    'CallerOssID': span['CallerOssID'],
                    'CallerCmdID': span['CallerCmdID'],
                    'CalleeOssID': span['CalleeOssID'],
                    'CalleeCmdID': span['CalleeCmdID'],
                    'count' : 1,
                    'cost_time_total': cost_time,
                    'cost_time_max': cost_time,
                    'cost_time_min': cost_time,
                    'not_zero_network_ret_count': 0 if span['NetworkRet'] == 0 else 1,
                    'not_zero_service_ret_count': 0 if span['ServiceRet'] == 0 else 1,
                    }

    return list(graph.values())


def check_node_exists(target, node_list):
    '''
    用于判断结点是否已经存在在图中，可以修改判断方法来调整trace中结点的粒度（接口、服务）
    '''
    for node in node_list:
        if node['ossid'] == target['ossid'] \
                and node['cmdid'] == target['cmdid']:
            return node

    return None



def main():
    print('loadding normal data...')
    normal_data_df = pd.read_csv(normal_data_path)
    normal_cs_df = pd.read_csv(normal_cs_path)

    print('loadding data...')
    data_df = pd.read_csv(data_path)
    cs_df = pd.read_csv(cs_path)

    normal_graph = generate_graph(normal_cs_df, normal_data_df)
    chaos_graph = generate_graph(cs_df, data_df)

    print('normal edges count:', len(normal_graph))
    print('chaos edges count:', len(chaos_graph))

    graph = Graph('bolt://fdse.icu:7687', auth=('neo4j', 'password'))
    graph.delete_all()

    def insert(data, label='normal'):
        nodes = []
        relations = []

        node_label = 'Operation'
        edge_label = 'Call'

        print('start insert into neo4j...')
        for span in tqdm(data):
            from_node = Node(node_label, name = span['CallerOssID'], 
                ossid = span['CallerOssID'], cmdid = span['CallerCmdID'], type = label
                )
            n = check_node_exists(from_node, nodes)
            if n == None:
                nodes.append(from_node)
            else:
                from_node = n

            to_node = Node(node_label, name = span['CalleeOssID'],
                ossid = span['CalleeOssID'], cmdid = span['CalleeCmdID'], type = label
                )
            n = check_node_exists(to_node, nodes)
            if n == None:
                nodes.append(to_node)
            else:
                to_node = n
            
            call = Relationship(from_node, edge_label, to_node,
                count=span['count'], cost_time_mean=span['cost_time_total']/span['count'],
                cost_time_max = span['cost_time_max'], cost_time_min = span['cost_time_min'],
                not_zero_network_ret_count = span['not_zero_network_ret_count'],
                not_zero_service_ret_count = span['not_zero_service_ret_count'],
                )
            relations.append(call)

        graph.create(Subgraph(nodes, relations))

    insert(normal_graph)
    insert(chaos_graph, 'chaos')

    print('inserted')

    
if __name__ == '__main__':
    main()
