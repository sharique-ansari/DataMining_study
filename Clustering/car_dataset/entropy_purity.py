
import numpy as np
import pandas as pd





def entropy(clusters):
	clusters_entropy=[]
	q=len(set(data["Species"]))
	print(q)
	for i in clusters:
		clusters_entropy.append(0)
		print(clusters_entropy)
		nr=len(i)
		for j in range(q):
			nr1=len(i[data["Species"]==j])
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


def purity(clusters):


