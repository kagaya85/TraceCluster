import pandas as pd
import numpy as np
import time

datapath = '/data/TraceCluster/raw/trainticket/chaos/2022-02-28_00-00-00_10h_traces.csv'
savepath_prefix = '/data/TraceCluster/raw/trainticket/rootchaos/'
delta = 3600000 # ms = 1h

def main():
	data_type = {'StartTime': np.uint64, 'EndTime': np.uint64}
	data = pd.read_csv(datapath, dtype=data_type).drop_duplicates().dropna()

	start = int(round(time.mktime(time.strptime('2022-02-28 00:00:00', '%Y-%m-%d %H:%M:%S'))*1000))
	print(start)
	
	while True:
		p = data[(data['StartTime'] >= start) & (data['StartTime'] < start + delta)]
		if len(p) == 0:
			break
		
		path = savepath_prefix + f'{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start//1000))}_1h_traces.csv'
		p.to_csv(path)
		print(f'{len(p)} spans saved in {path}')
		start = start + delta



if __name__ == '__main__':
	main()