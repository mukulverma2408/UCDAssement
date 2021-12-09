# Python Code for UCD Assement
# Author - Mukul Verma
# Using this project I would like to compare the two most popular online video streaming service provider
# in India which is Netflix and Amazon Prime Videos, I will use this project
# to analyze the content on these two platform
# the conclusion we can draw from the analysis.
# Source of Data - Kaggle
# Netflix Data = https://www.kaggle.com/shivamb/netflix-shows
# Amazon Prime Video Data = https://www.kaggle.com/shivamb/amazon-prime-movies-and-tv-shows
# OMDB API Data = http://www.omdbapi.com/
# PIMA India Diabetes Dataset = https://www.kaggle.com/uciml/pima-indians-diabetes-database
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

def create_dateframe(file):
    """This function creates Pandas DataFrame"""
    df = pd.read_csv(file)
    return df

def contains_missing_values(dataframe):
    """Function to check if dataframe contains any missing values"""
    missing_val = dataframe.isnull().sum()
    return missing_val

def plot_histogram(dataframe):
    """Function to create histogram for dataframe passed"""
    dataframe.isna().sum().plot(kind='bar')
    plt.show()

def compare_release_year(dataframe1, dataframe2):
    """Function to create histogram and compare release_year column b/w two dataframe"""
    dataframe1['release_year'].hist(bins=50, alpha=0.5)
    dataframe2['release_year'].hist(bins=50, alpha=0.5)
    plt.legend(['Netflix', 'Amazon'])
    plt.show()

def get_movie(title, year='') :
    """Function to return movie details from OMDB API, year is an optional argument"""
    API_KEY = 'f92fb218'
    Info = requests.get('http://www.omdbapi.com/?apikey='+API_KEY+'&t='+title+'&y='+year).json()
    return Info

def count_entries(file_name, chunk_size, colname):
    """Function to return a dictionary with counts of occurrences as value for key"""
    counts_dict = {}
    for chunk in pd.read_csv(file_name, chunksize=chunk_size):
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1
    return counts_dict

 #Code to demenstorate loading data from csv file
df_netflix = create_dateframe('/home/mukul/PycharmProjects/UCDAssement/data/netflix_titles.csv')
df_amazon = create_dateframe('/home/mukul/PycharmProjects/UCDAssement/data/amazon_prime_titles.csv')
#Code to demonstrate basic operations on pandas dataframe like find missing & duplicate
"""
print(df_netflix.info())
print(df_amazon.info())
missing_netflix = contains_missing_values(df_netflix)
missing_amazon = contains_missing_values(df_amazon)
print('Missing Values in Netflix Dataframe : ',  missing_netflix)
print('Missing values in Amazon Dataframe : ', missing_amazon)
duplicates_amazon = df_amazon.duplicated(subset=['title'], keep=False)
print(df_amazon[duplicates_amazon])
duplicates_netflix = df_netflix.duplicated(subset=['title'], keep=False)
print(df_netflix[duplicates_netflix])
"""
 #Code to demenostrate merging dataframe and removing duplicate
"""
df_combined = pd.concat([df_netflix, df_amazon])
print(df_combined.info())
duplicates_combined = df_combined.duplicated(subset=['title'], keep=False)
print(df_combined[duplicates_combined].info())
#After combining the dataset we have found that there are 778 common titles in both the dataframe, now dropping duplicates
unique_movie = df_combined.drop_duplicates(subset=['title','director','cast'])
print(unique_movie.info())
"""

 #Code to built histogram to check for missing values within two dataframe
"""
plot_histogram(df_netflix)
plot_histogram(df_amazon)
"""
 #Code to built a histogram to compare release year between two dataset
"""
compare_release_year(df_netflix, df_amazon)
"""
 #Code to demonstrate loading data from API
"""
movie_list = ['Fast', 'Jungle', 'Godfather']
for i in movie_list:
    movie_info = get_movie(i)
    for i in movie_info:
        print(i + ': ', movie_info[i])
"""
 #Code to demonstrate creating function, dictionary, using iterators
"""
result_counts = count_entries('./data/netflix_titles.csv', 10, 'release_year')
print(result_counts)
"""

