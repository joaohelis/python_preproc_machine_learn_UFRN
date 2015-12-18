import pandas as pd
import pylab as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

"""
	Chose the better value of K for the K-NN classifier
"""

dataset_name = "dataset/covertype_reduced_normalized.csv"
dataset = np.loadtxt(open(dataset_name,"rb"),delimiter=",")

X = dataset[:,:-1]
Y = dataset[:,-1]

results = []
for n in range(1, dataset.shape[1], 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    accuracy = cross_val_score(clf, X, Y, cv=10).mean()
    print("Neighbors: %d, Accuracy: %3f" % (n, accuracy))
    results.append([n, accuracy])

results = pd.DataFrame(results, columns=["n", "accuracy"])

results.to_csv("tables/better_k_to_knn.csv", sep=';')

pl.title("Accuracy with Increasing K\n")
pl.plot(results.n, results.accuracy, '-o')
pl.ylabel('K-NN Accuracy'), pl.xlabel('k')
pl.show()