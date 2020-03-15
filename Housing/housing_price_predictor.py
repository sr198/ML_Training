#Python program to predict housing price from the California Housing Price data

import tarfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

#Collect data
#data: dataset/housing.tgz -> contains housing.csv

#Extract the tgz file
def extract_housing_csv( housing_path ):
    tgz_path = os.path.join( housing_path, "housing.tgz") 
    housing_tar = tarfile.open( tgz_path )
    if not os.path.isfile( os.path.join( housing_path, "housing.csv" )):
        housing_tar.extractall( path=housing_path )
        housing_tar.close()
    else:
        print( "File alredy exists!") 

#Read csv file and load the data
def load_housing_data_from_csv( housing_path ):
    csv_file = os.path.join( housing_path, "housing.csv" )
    #return a pandas DataFrame object
    return pd.read_csv( csv_file )

def analyze_data_frame( data_frame:pd.DataFrame ):
    #print( "Head:" )
    #print( data_frame.head() )
    print( "Info: ")
    print( data_frame.info() )
    #print( "Describe: " )
    #print( data_frame.describe() ) 

#Plot some graphs from the data frame
def graph_my_data( data_frame ):
    data_frame.hist( bins=50, figsize=(20,15))

#Split data into training/test data 
def simple_split_traing_test_set( data_frame:pd.DataFrame, test_ratio ):
    shuffled_indices = np.random.permutation( len(data_frame ))
    test_set_size = int( len(data_frame) * test_ratio )
    test_indices = shuffled_indices[:test_set_size]
    train_incides = shuffled_indices[test_set_size:]
    return data_frame.iloc[train_indices], data_frame.iloc[test_indices]

#We will use scikit-learn's train_test_split method
def split_training_test_set( data_frame ):
    return train_test_split( data_frame, test_size=0.2, random_state=42 )

def stratified_split_training_test_set( data_frame ):
    data_frame["income_cat"] = np.ceil(data_frame["median_income"]/ 1.5)
    data_frame["income_cat"].where(data_frame["income_cat"] < 5, 5.0, inplace=True )
    split = StratifiedShuffleSplit( n_splits=1, test_size=0.2, random_state=42 )
    for train_index, test_index in split.split( data_frame, data_frame["income_cat"]):
        strat_train_set = data_frame.loc[train_index]
        strat_test_set = data_frame.loc[test_index]
    return strat_train_set, strat_test_set

#correlation
def draw_correlation_matrix( data_frame ):
    correlation_attributes=["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix( data_frame[correlation_attributes], figsize=(12,8), alpha=0.1 )
#Cleanup

#Apply ML algorithms

#Training

#Evaluation

#Main program
extract_housing_csv( "Housing/dataset" )
housing_data_frame = load_housing_data_from_csv( "Housing/dataset")
analyze_data_frame( housing_data_frame )
graph_my_data( housing_data_frame )
strat_train_set, strat_test_set = stratified_split_training_test_set( housing_data_frame )
train_set = strat_train_set.copy()
train_set.plot( kind="scatter", x="longitude", y="latitude" )
train_set.plot( kind="scatter", x="longitude", y="latitude",alpha=0.1,
s=train_set["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True )
draw_correlation_matrix( train_set )
plt.show()