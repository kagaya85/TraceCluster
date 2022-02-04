import pandas as pd

# filepath = '/data/TraceCluster/raw/wechat/trace_mmfindersynclogicsvr/click_stream_2022-01-18_23629.csv'
filepath = '/data/TraceCluster/raw/wechat/trace_mmfindersynclogicsvr/click_stream_2022-01-17_23629.csv'


def f(x):
    return x.max()-x.min()


def main():
    df = pd.read_csv(filepath)
    print(f'load data from {filepath}')

    print('processing...')

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
    res = df.groupby('CallerOssID').agg(
        min=pd.NamedAgg(column='CostTime', aggfunc='min'),
        max=pd.NamedAgg(column='CostTime', aggfunc='max'),
        mean=pd.NamedAgg(column='CostTime', aggfunc='mean'),
        var=pd.NamedAgg(column='CostTime', aggfunc='var'),
        range=pd.NamedAgg(column='CostTime', aggfunc=f)
    ).sort_values(['var', 'range'], ascending=False).head(10)
    print(res)

    print('finished')


if __name__ == '__main__':
    main()
