import pandas as pd

# filepath = '/data/TraceCluster/raw/trainticket/chaos/2022-01-11_00-00-00_24h_traces.csv'
filepath = '/data/TraceCluster/raw/trainticket/normal/2022-01-11_00-00-00_12h_traces.csv'


def f(x):
    return x.max()-x.min()


def main():
    df = pd.read_csv(filepath)
    print(f'load data from {filepath}')

    df['Duration'] = df['EndTime'] - df['StartTime']

    print('processing...')
    df.loc[df['ParentSpan'] == '-1'].groupby('URL').agg(
        min=pd.NamedAgg(column='Duration', aggfunc='min'),
        max=pd.NamedAgg(column='Duration', aggfunc='max'),
        mean=pd.NamedAgg(column='Duration', aggfunc='mean'),
        var=pd.NamedAgg(column='Duration', aggfunc='var'),
        range=pd.NamedAgg(column='Duration', aggfunc=f)
    ).sort_values(['var', 'range'], ascending=False).to_csv('result.csv')
    print('finished')


if __name__ == '__main__':
    main()
