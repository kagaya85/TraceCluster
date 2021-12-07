import pandas as pd
from tqdm import tqdm

filename = 'data/raw/wechat/12-3/call_graph_2021-12-03_23629.csv'


def main():
    df = pd.read_csv(filename)
    count = 0
    for idx, row in df.iterrows():
        if row['CallerNodeID'] == row['CalleeNodeID']:
            count = count + 1
            print(row)
    print(
        f'total num: {len(df)}, cycle num: {count}, account: {count/len(df)}')


if __name__ == '__main__':
    main()
