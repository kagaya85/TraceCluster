"""
This code is adapdated from the source code of moa, that can be found in the following address
https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/clusterers/clustream/WithKmeans.java
"""
from CluStream_master.Kernel import Kernel
import numpy as np
import itertools
from sklearn.cluster import KMeans

# from multiprocessing.dummy import Pool

# def min(iterable,key=lambda t:t):
# 	ans=-float("inf")
# 	for result in Pool().imap(key,iterable):

from multiprocessing.dummy import Pool

def multiprocess_min(iterable, key=lambda t:t):
    ans=float("inf")
    for item in pool.imap(key,iterable):
    	if item<ans:
    		ans=item
    return ans

pool=Pool()
# FILE=open("log.txt","w")
class CluStream:
	"""
	CluStream data stream clustering algorithm implementation

	Args:
		h (int): Range of the window
		m (int): Maximum number of micro kernels to use
		t (int): Multiplier for the kernel radius
	Attributes:
		kernels (list of clustream.Kernel) : microclusters
		time_window (int) : h
		m (int): m
		t (int): t
	"""
	def __init__(self,h=10000000000000000,m=800,t=2):    # h=1000, m=100
		self.kernels=[]
		self.time_window=h
		self.m=m
		self.t=t
	def offline_cluster(self,datapoint,timestamp):
		"""
		offline clustering of a datapoint

		Args:
			datapoint (ndarray): single point from the dataset
			timestamp (int): timestamp of the datapoint
		"""
		if len(self.kernels)!=self.m:
			#0. Initialize
			self.kernels.append(Kernel(datapoint,timestamp,self.t,self.m))
			return
		centers=[kernel.center for kernel in self.kernels] #TODO :faster computing with caching
		#1. Determine closest kernel
		closest_kernel_index,min_distance=min(
			((i,np.linalg.norm(center-datapoint)) for i,center in enumerate(centers)),
			key=lambda t:t[1]
			)
		closest_kernel=self.kernels[closest_kernel_index]
		closet_kernel_center=centers[closest_kernel_index]
		# 2. Check whether instance fits into closest_kernel
		if closest_kernel.n==1:
			# Special case: estimate radius by determining the distance to the
			# next closest cluster
			radius=min(( #distance between the 1st closest center and the 2nd
				np.linalg.norm(center-closet_kernel_center) for center in centers if not center is closet_kernel_center
			))
		else:
			radius=closest_kernel.get_radius()
		if min_distance<radius:
			# Date fits, put into kernel and be happy
			closest_kernel.insert(datapoint,timestamp)
			# print(f"{timestamp} fits",file=FILE)
			return
		# 3. Date does not fit, we need to free
		# some space to insert a new kernel

		threshold = timestamp - self.time_window # Kernels before this can be forgotten
		# 3.1 Try to forget old kernels
		oldest_kernel_index=next((
			i for i,kernel in enumerate(self.kernels) if kernel.get_relevance_stamp() < threshold
		),None)
		if oldest_kernel_index!=None:
			# print(f"{timestamp} forgot old kernel",file=FILE)
			self.kernels[oldest_kernel_index]=Kernel(datapoint,timestamp,self.t,self.m)
			return

		# 3.2 Merge closest two kernels
		# print(f"{timestamp} merge closest kernel",file=FILE)
		combination_indexes=itertools.combinations(range(len(centers)),2)
		closest_a,closest_b,dist=min(
			((i,j,np.linalg.norm(centers[j]-centers[i])) for i,j in combination_indexes),
			key=lambda t:t[-1]
		)

		self.kernels[closest_a].add(self.kernels[closest_b])
		self.kernels[closest_b]=Kernel(datapoint,timestamp,self.t,self.m)

	def predict(self, X=None):
		cluster_centers = list(map((lambda i: i.get_center()), self.kernels))
        #centers_weights = list(map((lambda i: i.get_weight()), self.micro_clusters))
		kmeans = KMeans(n_clusters=5, random_state=1)
		y_pred = kmeans.fit_predict(X=cluster_centers, y=None)
		return y_pred
