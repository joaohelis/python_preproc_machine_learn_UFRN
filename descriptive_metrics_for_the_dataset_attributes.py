import numpy as np
import pandas as pd

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

dataset_name = "covertype_reduced_normalized.csv"
dataset = np.loadtxt(open(dataset_name,"rb"), delimiter=",")

df = pd.DataFrame(dataset[:,:-1])

descriptive_metrics = df.describe().transpose()
descriptive_metrics['attribute_name'] = attributes[:-1]

print descriptive_metrics

descriptive_metrics.to_csv("tables/descriptive_metrics.csv", sep=';')
