
import numpy as np
import pandas as pd





def entropy(clusters,data1):
	clusters_entropy=[]
	q=len(set(data1["Outcome"]))
	print(q)
	for i in clusters:
		i=pd.DataFrame(i)
		clusters_entropy.append(0)
		print(clusters_entropy)
		nr=len(i)
		for j in range(q):
			nr1=len(i[data1["Outcome"]==j])
			print(nr1)
			if(nr1!=0):
				clusters_entropy[-1]=clusters_entropy[-1]+(nr1/nr)*(np.log(nr1/nr))
		clusters_entropy[-1]=clusters_entropy[-1]*-1/np.log(q)
		print("final : ",clusters_entropy)
	total_entropy=0
	j=0
	len_data=len(data)
	for i in clusters:
		total_entropy+=(len(i)/len_data)*clusters_entropy[j]
		j+=1
	return total_entropy


def purity(clusters,data1):
	total_purity=0
	len_data=len(data)
	q=len(set(data1["Outcome"]))
	for i in clusters:
		i=pd.DataFrame(i)
		nr=len(i)
		nr1=[]
		for j in range(q):
			nr1.append(len(i[data1["Outcome"]==j]))
		_max=np.max(nr1)
		total_purity+=_max/len_data
	return total_purity







