import numpy as np
import pandas as pd
def euclid_distance(a,b):
	a=a[:a.size-1]
	b=b[:b.size-1]
	a=np.array(a)
	b=np.array(b)
	return np.sqrt(np.sum(np.square(a-b)))
def manhattan_distance(a,b):
	a=a[:a.size-1]
	b=b[:b.size-1]
	a=np.array(a)
	b=np.array(b)
	return np.sum(np.abs(a-b))
def mean(cluster):
	#cluster=np.array(cluster)
	return np.mean(cluster,axis=0)
def median(cluster):
	return np.median(cluster,axis=0)
def representative_function(distance_function,centroid_function):
	#Input=np.loadtxt('car_evaluation',delimiter=',')
	data=np.loadtxt("CarDataSet.csv",delimiter=',') 
	#Input=data[:,:data[0].size-1]
	Input=data
	dimension=Input[0].size-1
	print("Enter number of clusters to make")
	clusters_count=int(input())
	clusters=[[] for i in range(clusters_count)]
	#representatives=np.random.randint(-100,100,size=(clusters_count,dimension))
	representatives=Input[:clusters_count].copy()
	#representatives=[[1],[11],[28]]
	#print(representatives)
	temp=[]
	iter=1
	while(not np.array_equal(temp,representatives)):
		#print("Representatives :",representatives)
		clusters=[[] for i in range(clusters_count)]
		#print('hiii')
		temp=representatives.copy()
		print("Iteration ",iter)
		iter=iter+1
		for i in Input:
			#print("Input : ",i)
			arr=[]
			for j in representatives:
				arr.append(distance_function(i,j))
				#print("Distance from ",j, ":",arr[representatives.index(j)])
			clusters[arr.index(min(arr))].append(i)
			#print("Point ",i,"goes to the cluster with representative ",representatives[arr.index(min(arr))])
		#print("Clusters after above iteration" ,clusters)
		count=0
		for i in clusters:
			if len(i)>0:
				representatives[count]=centroid_function(i)
			count=count+1
				#print()	
	print("\nFinal Clusters")
	for i in clusters:
		print(i)
		print()
	return clusters
print("Enter 1 for k-Mean\nEnter 2 for K-Median")
z=int(input())
if z==1:
	representative_function(euclid_distance,mean)
else:
	representative_function(manhattan_distance,median)
