import matplotlib.pyplot as plt 
import json
import pickle
from scipy.io import loadmat
import pdb 
import numpy as np
'''
rcnn_car = pickle.load(open('/home/tiankun/pro/pytorch-FPN/output/res101/DETRACvoc_test/default/res101_faster_rcnn_iter_120000/car_pr.pkl','rb'))
rcnn_bus = pickle.load(open('/home/tiankun/pro/pytorch-FPN/output/res101/DETRACvoc_test/default/res101_faster_rcnn_iter_120000/bus_pr.pkl','rb'))
rcnn_van = pickle.load(open('/home/tiankun/pro/pytorch-FPN/output/res101/DETRACvoc_test/default/res101_faster_rcnn_iter_120000/van_pr.pkl','rb'))
rcnn_motor = pickle.load(open('/home/tiankun/pro/pytorch-FPN/output/res101/DETRACvoc_test/default/res101_faster_rcnn_iter_120000/motor_pr.pkl','rb'))

with open("yolo_recall.json",'r') as load_f:
    RTK = json.load(load_f)

with open("yolo_precision.json",'r') as load_f:
    PTK = json.load(load_f)
'''
'''
# Faster curve
plt.figure(1)
plt.title('PR Curve')
plt.plot(rcnn_car['rec'],rcnn_car['prec'],color='red',label='car',linewidth=3)
plt.plot(rcnn_bus['rec'],rcnn_bus['prec'],color='limegreen',label='bus',linewidth=3)
plt.plot(rcnn_van['rec'],rcnn_van['prec'],color='skyblue',label='van',linewidth=3)
plt.plot(rcnn_motor['rec'],rcnn_motor['prec'],color='orange',label='motor',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show() 
'''
'''
# YOLO curve
plt.figure(1)
plt.title('PR Curve')
plt.subplot(1,3,1)
plt.plot(RTK['car'],PTK['car'])
plt.subplot(1,3,2)
plt.plot(RTK['bus'],PTK['bus'])
plt.subplot(1,3,3)
plt.plot(RTK['van'],PTK['van'])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
'''
'''
exp1 = loadmat('exp1.mat')
exp3 = loadmat('exp3.mat')
exp5 = loadmat('exp5.mat')
exp5sub1 = loadmat('exp5sub1.mat')
exp5sub2 = loadmat('exp5sub2.mat')
exp6 = loadmat('exp6.mat')

x=np.linspace(0.7,1,30)

plt.figure(1)
plt.title('PRW performance')
plt.plot(x,exp1['map'][::-1],color='red',label='exp1',linewidth=3)
plt.plot(x,exp3['map'][::-1],color='limegreen',label='exp3',linewidth=3)
plt.plot(x,exp5['map'][::-1],color='skyblue',label='exp5',linewidth=3)
plt.plot(x,exp6['map'][::-1],color='orange',label='exp6',linewidth=3)
plt.legend()
plt.xlabel('det_thresh')
plt.ylabel('mAP')
plt.show() 
pdb.set_trace()


plt.figure(2)
plt.title('PR Curve')
plt.plot(RTK['car'],PTK['car'],color='red',label='car',linewidth=3)
plt.plot(RTK['bus'],PTK['bus'],color='limegreen',label='bus',linewidth=3)
plt.plot(RTK['van'],PTK['van'],color='skyblue',label='van',linewidth=3)
plt.plot(RTK['motor'],PTK['motor'],color='orange',label='motor',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()   



#contrast curve
print(rcnn_motor['ap'])
plt.figure(3)
plt.title('Motor PR Curve')
plt.plot(rcnn_motor['rec'],rcnn_motor['prec'],color='red',label='Faster_RCNN',linewidth=3)
plt.plot(RTK['motor'],PTK['motor'],color='limegreen',label='YOLO',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()  

plt.figure(4)
plt.title('Car PR Curve')
plt.plot(rcnn_car['rec'],rcnn_car['prec'],color='red',label='Faster_RCNN',linewidth=3)
plt.plot(RTK['car'],PTK['car'],color='limegreen',label='YOLO',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()  

plt.figure(5)
plt.title('Bus PR Curve')
plt.plot(rcnn_bus['rec'],rcnn_bus['prec'],color='red',label='Faster_RCNN',linewidth=3)
plt.plot(RTK['bus'],PTK['bus'],color='limegreen',label='YOLO',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()  

plt.figure(6)
plt.title('Van PR Curve')
plt.plot(rcnn_van['rec'],rcnn_van['prec'],color='red',label='Faster_RCNN',linewidth=3)
plt.plot(RTK['van'],PTK['van'],color='limegreen',label='YOLO',linewidth=3)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()  
'''

# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,2,1)
# ax1 = fig.add_subplot(2,2,1)
# ax2 = fig.add_subplot(1,2,1)

# original Sieve
oSieve = []
abnormal = []
x = []
file_oSieve = open("./WSET/result_Sieve_STV_2022-02-22_15-36-15.txt",  "r")
listOfLines = file_oSieve.readlines()
file_oSieve.close()
for idx, line in enumerate(listOfLines):
    oSieve.append(float(line.strip().split('\t')[2]))
    x.append(idx+1)
    if line.split('\t')[1] == "1":
        abnormal.append(idx+1)

# modified Sieve
mSieve = []
file_mSieve = open("./WSET/result_Sieve_ourMethod_2022-02-22_15-26-40.txt",  "r")
listOfLines = file_mSieve.readlines()
file_mSieve.close()
for line in listOfLines:
    mSieve.append(float(line.strip().split('\t')[3]))

# original PERCH
oPERCH = []
file_oPERCH = open("./WSET/result_PERCH_original_2022-02-23_14-40-32.txt",  "r")
listOfLines = file_oPERCH.readlines()
file_oPERCH.close()
for idx, line in enumerate(listOfLines):
    if idx >= 178:
        oPERCH.append(float(line.strip().split('\t')[2]))

# modified PERCH
mPERCH = []
file_mPERCH = open("./WSET/result_PERCH_ourMethod_2022-02-23_14-35-11.txt",  "r")
listOfLines = file_mPERCH.readlines()
file_mPERCH.close()
for idx, line in enumerate(listOfLines):
    if idx >= 178:
        mPERCH.append(float(line.strip().split('\t')[2]))


figall=plt.figure(1)
fig = figall.add_subplot(1,1,1)
fig.grid(True)
fig.plot(x,oSieve,color='purple',label='oSieve',linewidth=1,alpha=0.6,marker='d')
fig.plot(x,mSieve,color='limegreen',label='mSieve',linewidth=1,alpha=0.6,marker='8')
fig.plot(x,oPERCH,color='orange',label='oPERCH',linewidth=1,alpha=0.6,marker='s')
fig.plot(x,mPERCH,color='blue',label='mPERCH',linewidth=1,alpha=0.6,marker='*')
fig.vlines(abnormal, 0, 1, linestyles='dashed', colors='red')
legend = fig.legend()
# fig.legend(loc=1,bbox_to_anchor=(1.05,1.0))
fig.set_xlabel('samples')
fig.set_ylabel('P')
figall.savefig("./WSET/result.png")
figall.show()