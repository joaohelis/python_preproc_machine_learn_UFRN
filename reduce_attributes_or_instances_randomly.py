import numpy as np
import random
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import pylab as pl

"""
	Reduce attributes or instances randomly from the dataset, and then, verify the K-NN accuracy after such modifications
"""

def reduce(np_array, reduction_rate, reduction_type='col'):
	# reduction_type='row' or 'col' this parametter can be 'row' or 'col'
	reduction = int(reduction_rate * len(np_array)) if reduction_type == 'row' else int(reduction_rate * np_array.shape[1])
	for i in range(reduction):
		if reduction_type == 'row':
			np_array = np.delete(np_array, random.randint(0, len(np_array)-1), 0)  
		else:
			np_array = np.delete(np_array, random.randint(0, np_array.shape[1]-2), axis=1)
	return np_array

dataset_name = "covertype_reduced_normalized.csv"
dataset = np.loadtxt(open(dataset_name,"rb"), delimiter=",")

reduction_rates = [float(i)/10 for i in range(0, 10)]
results = []
for reduction_rate in reduction_rates:
	reduced_dataset = reduce(dataset, reduction_rate, reduction_type='col')
	clf = KNeighborsClassifier(1)
	X = reduced_dataset[:,:-1]
	Y = reduced_dataset[:,-1]
	accuracy = cross_val_score(clf, X, Y, cv=10).mean()
	results.append([reduction_rate, accuracy])

results = pd.DataFrame(results, columns=["reduction_rate", "accuracy"])

results.to_csv("tables/reduce_attributes_randomly.csv", sep=';')

print results

pl.plot(results.reduction_rate, results.accuracy, '-o')
pl.title("Accuracy with reducing the number of database attributes randomly\n")
pl.xlabel('(%) Reduction Rate')
pl.ylabel('K-NN Accuracy')
pl.show()