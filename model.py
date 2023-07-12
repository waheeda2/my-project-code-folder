import time
start = time.time()
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data =pd.read_csv("train.csv").values
clf=KNeighborsClassifier(n_neighbors=3)

#trianing dataset
xtrain =data[0:40000,1:]
train_lebel=data[0:40000,0]
clf.fit(xtrain,train_lebel)

#test dataset
xtest=data[40000:,1:]
actual_lebel=data[40000:,0]

def hdr_predition(index):
	index=int(index,10)
	predition = clf.predict([xtest[index]])
	return predition

	#Accuracy
def hdr_accuracy():

	p=clf.predict(xtest)
	count=0
	for i in range(0,1000):
		count+=1 if p[i]==actual_lebel[i] else 0
	accuracy = (count/1000)*100
	return accuracy

def hdr_predition_time():

	end = time.time()
	predition_time = end - start
	return predition_time

def hdr_img(index):
	index=int(index,10)
	d=xtest[index]
	d.shape=(28,28)
	pt.imshow(255-d,cmap='gray')
	seq=str(index)
	pt.savefig('static/uploads/pic'+seq+'.png')
	return 'static/uploads/pic'+seq+'.png'
