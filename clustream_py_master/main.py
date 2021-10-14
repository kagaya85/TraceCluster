#!/usr/bin/env python
from CluStream import CluStream

import numpy as np
import pandas as pd
from time import time
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("dataset",help=".csv dataset to test")
parser.add_argument("-c","--chunksize",type=int,default=2000)
parser.add_argument("-m","--microclusters",type=int,default=2000)
parser.add_argument("-H","--window-range",type=int,default=1000)
args=parser.parse_args()



if __name__=="__main__":
	model=CluStream(m=args.microclusters,h=args.window_range)
	t=0
	total_time=0
	for chunk in pd.read_csv(args.dataset,chunksize=args.chunksize,dtype=np.float32):
		for datapoint in chunk.values:
			start=time()
			model.offline_cluster(datapoint,t)
			if t>model.m: #ignore initialization
				total_time+=1//(time()-start)
			t+=1
	print(total_time/((t-model.m)),"avg points per second")
