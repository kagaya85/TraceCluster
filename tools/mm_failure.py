import pandas as pd

# Ds,Hour,LogID,Uin,IfSuccess,TimeStamp,GraphIdBase64,CgiID,CallerOssID,CallerCmdID,CallerNodeID,CalleeOssID,CalleeCmdID,CalleeNodeID,NetworkRet,ServiceRet,CostTime,ErrMsg,Crc
filepath = '/data/TraceCluster/raw/wechat/trace_mmfindersynclogicsvr/call_graph_2022-01-18_23629.csv'
# filepath = '/data/TraceCluster/raw/wechat/trace_mmfindersynclogicsvr/call_graph_2022-01-17_23629.csv'
clickstream = '/data/TraceCluster/raw/wechat/trace_mmfindersynclogicsvr/click_stream_2022-01-18_23629.csv'


def f(x):
    return x.max()-x.min()


def main():
    df = pd.read_csv(filepath)
    print(f'load data from {filepath}')

    print('processing...')
    targetdf = df.loc[df['CallerOssID'] == 23629]

    # callee count
    # res = targetdf.groupby('CalleeOssID').size(
    # ).sort_values(ascending=False).head(15)
    # print(res)

    # code
    # res = targetdf.groupby('NetworkRet').size().sort_values(ascending=False)
    # print(res)
    # res = targetdf.groupby('ServiceRet').size().sort_values(ascending=False)
    # print(res)

    # CostTime
    # res = df.groupby('CallerOssID').agg(
    #     min=pd.NamedAgg(column='CostTime', aggfunc='min'),
    #     max=pd.NamedAgg(column='CostTime', aggfunc='max'),
    #     mean=pd.NamedAgg(column='CostTime', aggfunc='mean'),
    #     var=pd.NamedAgg(column='CostTime', aggfunc='var'),
    #     range=pd.NamedAgg(column='CostTime', aggfunc=f)
    # ).sort_values(['var', 'range'], ascending=False).head(10)
    # print(res)

    # trace
    cdf = pd.read_csv(clickstream)
    cresdf = cdf.sort_values(['CostTime']).head(20)
    print(cresdf[['CallerOssID', 'TimeStamp',
          'GraphIdBase64', 'CostTime', 'RetCode']])
    res = cresdf.merge(df[['GraphIdBase64']], on='GraphIdBase64').groupby(
        'GraphIdBase64').size().sort_values(ascending=True)
    print(res)
    # res = df.groupby('GraphIdBase64').size().sort_values(
    #     ascending=False).to_frame().merge(cdf, on='GraphIdBase64').head(50)
    # print(res)
    # res = df.groupby('GraphIdBase64').size().sort_values(
    #     ascending=True).to_frame().merge(cdf, on='GraphIdBase64').head(50)
    # print(res)

    # df.loc[(df['GraphIdBase64'] == 'COfdmI8GEOsBGIm31VwgpZULKIkL') | (df['GraphIdBase64'] == 'CO3dmI8GEAkY3MvaXCCmlwsoqww='), ['IfSuccess', 'TimeStamp', 'GraphIdBase64', 'CgiID', 'CallerOssID',
    #    'CallerCmdID', 'CallerNodeID', 'CalleeOssID', 'CalleeCmdID', 'CalleeNodeID', 'NetworkRet', 'ServiceRet', 'CostTime', 'ErrMsg']].to_csv('result.csv')

    print('finished')


if __name__ == '__main__':
    main()
