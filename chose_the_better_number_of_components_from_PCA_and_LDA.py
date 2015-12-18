# -*- coding: utf-8 -*-

from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd
import pylab as pl
import random

from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier

"""
	Chose the better number of components for for representing the dataset from PCA and LDA methods
	- Verify this number according K-NN accuracy
"""

dataset_name = "dataset/covertype_reduced_normalized.csv"  
dataset = np.loadtxt(open(dataset_name,"rb"), delimiter=",")
X = dataset[:,:-1]
Y = dataset[:,-1]

results = []
for n in range(1, dataset.shape[1]):
	reduction_method = LDA(n_components=n) 
	X_r2 = reduction_method.fit(X, Y).transform(X)
	#reduction_method = PCA(n_components=n)
	#X_r2 = reduction_method.fit(X).transform(X)
	clf = KNeighborsClassifier(n_neighbors=1)
	accuracy = cross_val_score(clf, X_r2, Y, cv=10).mean()
	results.append([n, accuracy])
	print("Components: %d, Accuracy: %3f" % (n, accuracy))

results = pd.DataFrame(results, columns=["n", "accuracy"])

results.to_csv("tables/knn_accuracy_with_lda_reduction.csv", sep=';')

print results

pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing the number of components - LDA\n")
pl.xlabel('Number of components')
pl.ylabel('K-NN Accuracy')
pl.show()