
import numpy as np
import pandas as pd

"""
def euclid_distance(a,b):
	a=np.array(a)
	b=np.array(b)
	return np.sqrt((np.square(a-b)))
"""
def DBSCAN3():
	data=np.loadtxt("Wine.csv",delimiter=',')
	db = DBSCAN(eps=4, min_samples=20).fit(data)
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	clusters = [data[labels == i] for i in range(n_clusters_)]
	outliers = data[labels == -1]
	return (clusters,outliers,len(clusters))
def euclid_distance1(a,b):
	a=np.array(a)
	b=np.array(b)
	return np.sqrt(np.sum(np.square(a-b),axis=1))

visited=(np.array([False for i in range(811)]))
#data=np.random.randint(-100,100,size=(100,2)).tolist()
#data=np.array(pd.read_csv("/home/jayanth/Documents/DMG_Project/datasets/Sales.csv").iloc[:,0:4]).tolist()
data=np.array(d3).tolist()
#data1=np.arange(50).tolist()
#data2=np.arange(100,150).tolist()
#data=data1+data2
e=10
minpts=5
clusters=[]
outliers=[]
#distance_func=np.vectorize(euclid_distance)


def region_fun(point):
	if len(point)==1:
		distances=euclid_distance(point,data)
	else:
		distances=euclid_distance1(point,data)
	return distances<=e

def expand_cluster(point,neighbourhood):
	clusters[-1].append(point)
	print(clusters[-1])
	#print("Clusters :",clusters)
	for i in neighbourhood:
		print(i)
		print("check :",data.index(i))
		if visited[data.index(i)]==False:
			
			neighbourhood_1=np.array(data)[region_fun(i)]
			if len(neighbourhood_1)>minpts:

				visited[data.index(i)]=True
				#neighbourhood=neighbourhood+neighbourhood_1.tolist()
				expand_cluster(i,neighbourhood_1.tolist())
			else:
				
				check=0
				visited[data.index(i)]=True
				clusters[-1].append(i)
				print(clusters[-1])
				
				#for j in clusters:
				#	if j.count(i)==1:
				#		check=1
				#		break
				#if(check==0):
				#	visited[data.index(i)]=True
				#	clusters[-1].append(i)
				#	#print(clusters)

			


def DBSCAN():
	count=0
	for i in range(len(data)):
		if visited[i]==False:
			
			
			neighbourhood=np.array(data)[region_fun(data[i])]
			print("*****************",neighbourhood)
			if len(neighbourhood)>minpts:
				print("visited Point :",data[i])
				visited[i]=True
				clusters.append([])
				print("neighbourhood :",neighbourhood)
				expand_cluster(data[i],neighbourhood.tolist())
				
	for i in range(len(data)):
		if(visited[i]==False):
			outliers.append(data[i])
	for i in clusters:
		print(i)
		count=count+len(i)
		print()
		print()
		print()
	print("Outliers :",outliers);
	print()
	print(count)
	print(len(outliers))
	print(len(clusters))
