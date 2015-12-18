import numpy as np
import pylab as pl
import pandas as pd
import sys
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import random

attributes = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
			  'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
			  'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
			  'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4',
			  'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 
			  'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 
			  'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 
			  'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 
			  'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 
			  'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 
			  'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39', 'Soil_Type_40', 'Cover_Type']

def attributes_corrcoef_sum(dataset, attributes):
	dataset = np.transpose(dataset)
	corr = np.corrcoef(dataset)
	attr_corrcoef = [0 for i in range(len(corr))]
	for i in range(0, len(corr)-1):
		for j in range(i + 1, len(corr)):
			correlation = 0
			if str(corr[i][j]) != 'nan':
				correlation = abs(corr[i][j])
			attr_corrcoef[i] += correlation
			attr_corrcoef[j] += correlation
	attributes_corrcoef = [[attr_corrcoef[i], attributes[i], i] for i in range(len(attr_corrcoef))]
	attributes_corrcoef.sort(reverse=True)
	return attributes_corrcoef

def print_corrcoef(attr_corrcoef):
	print "--------------------------------------------------------"
	sys.stdout.write("{:<5}{:<40}{:<51}\n".format("ID", "Attribute Name", "Corrcoef SUM"))
	print "--------------------------------------------------------"
	for attr in attr_corrcoef:
		sys.stdout.write("{:<5}{:<39}{:<51}\n".format(attr[2], attr[1], attr[0]))
	print "--------------------------------------------------------"

def reduce_more_correlated_attributes(np_array, attr_corrcoef, choice_rate=1.0):
	reduction = int(np_array.shape[1] * choice_rate)
	choiced_attributes = []
	for i in range(np_array.shape[1] - reduction):
		choiced_attributes.append(np_array[:,i])
	return np.array(choiced_attributes).transpose()

dataset_name = "covertype_reduced_normalized.csv"
dataset = np.loadtxt(open(dataset_name,"rb"), delimiter=",")

X = dataset[:,:-1]
Y = dataset[:,-1]

corrcoef = attributes_corrcoef_sum(X, attributes[:-1])
print_corrcoef(corrcoef)

reduction_rates = [float(i)/10 for i in range(0, 9)]
results = []
for reduction_rate in reduction_rates:
	reduced_dataset = reduce_more_correlated_attributes(X, corrcoef, reduction_rate)
	clf = KNeighborsClassifier(1)
	accuracy = cross_val_score(clf, reduced_dataset, Y, cv=10).mean()
	results.append([reduction_rate, accuracy])

results = pd.DataFrame(results, columns=["reduction_rate", "accuracy"])

results.to_csv("tables/more_correlated_attributes.csv", sep=';')

print results

pl.plot(results.reduction_rate, results.accuracy, '-o')
pl.title("Accuracy with reducing the must correlated attributes\n")
pl.xlabel('(%) Reduction Rate')
pl.ylabel('K-NN Accuracy')
pl.show()