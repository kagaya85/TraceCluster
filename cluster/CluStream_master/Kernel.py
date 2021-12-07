"""
This code is adapdated from the source code of moa, that can be found in the following address
https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/clusterers/clustream/ClustreamKernel.java
"""
import numpy as np
RADIUS_FACTOR = 1.8

EPSILON = 0.00005
MIN_VARIANCE = 1e-50
class Kernel:
	"""
	MicroCLuster implementation
	
	Args:
		datapoint (ndarray): Single point from the dataset
		timestamp (int): Time of the datapoint
		t (int): Multiplier for the kernel radius
		m (int): Maximum number of micro kernels to use
	Attributes:
		n (int): Number of points in the cluster
		ls (ndarray): Linear sum of the points
		ss (ndarray): Squared sum of the points
		lst (int): Linear sum of the timestamps
		sst (int): Squared sum of the timestamps
	"""
	def __init__(self,datapoint,timestamp,t,m):
		self.t=t
		self.m=m

		self.n=1
		self.ls=datapoint
		self.ss=np.square(datapoint)
		self.lst=timestamp
		self.sst=np.square(timestamp)
		self.center=self.get_center()
	
	def insert(self,datapoint,timestamp):
		self.n+=1
		self.ls+=datapoint
		self.ss+=np.square(datapoint)
		self.lst+=timestamp
		self.sst+=np.square(timestamp)
		self.center=self.get_center()

	def add(self,other):
		assert self.ls.shape==other.ls.shape
		self.n+=other.n
		self.ls+=other.ls
		self.ss+=other.ss
		self.lst+=other.lst
		self.sst+=other.sst
		self.center=self.get_center()

	def get_relevance_stamp(self):
		if self.n<(2*self.m):
			return self.get_mu_time()
		return self.get_mu_time() + self.get_sigma_time() * self.get_quantile( (self.m)/(2*self.n) );

	def get_mu_time(self):
		return self.lst/self.n

	def get_sigma_time(self):
		temp=self.lst/self.n
		return np.sqrt(self.sst/self.n - np.square(temp))

	def get_quantile(self,z):
		assert  0<=z<=1
		return np.sqrt( 2 ) * self.inverse_error( 2*z - 1 );

	def get_radius(self):
		if self.n==0:
			return 0
		return self.get_deviation()*RADIUS_FACTOR

	def get_deviation(self):
		variance=self.get_variance_vector()
		return np.mean(np.sqrt(variance))

	def get_center(self):
		if self.n==1:
			return self.ls
		return self.ls/self.n

	def get_inclusion_probability(self,datapoint):
		if self.n==1:
			distance=np.linalg.norm(datapoint)
			if distance<EPSILON:
				return 1
		else:
			dist=self.calc_normalized_distance(datapoint)
			if dist<= self.get_radius():
				return 1
		return 0

	def get_variance_vector(self):
		res=np.zeros(len(self.ls))
		for i in range(len(self.ls)):
			ls=self.ls[i]
			ss=self.ss[i]

			ls_div_squared=np.square(ls/self.n)
			ss_div=ss/self.n

			res[i]=ss_div-ls_div_squared
			
			# Due to numerical errors, small negative values can occur.
            # We correct this by settings them to almost zero.
			# if -EPSILON<=res[i] <=0.0:
			if res[i] <=0.0: #i get negative values
				res[i]=MIN_VARIANCE
		return res

	def calc_normalized_distance(self,datapoint):
		# variance=self.get_variance_vector()

		return np.sqrt(np.sum(np.square(self.center-datapoint)
			#/variance
		))

	def inverse_error(self,x):
		z = np.sqrt(np.pi) * x;
		res = z / 2;

		z2 = np.square(z)
		z_prod = z * z2 # z^3
		res += (1.0 / 24) * z_prod

		z_prod *= z2;  # z^5
		res += (7.0 / 960) * z_prod

		z_prod *= z2;  # z^7
		res += (127 * z_prod) / 80640

		z_prod *= z2;  # z^9
		res += (4369 * z_prod) / 11612160

		z_prod *= z2;  # z^11
		res += (34807 * z_prod) / 364953600

		z_prod *= z2;  # z^13
		res += (20036983 * z_prod) / 797058662400

		return res