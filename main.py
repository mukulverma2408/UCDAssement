# Python Code for UCD Assement
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)


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


def get_movie(title, year=''):
    """Function to return movie details from OMDB API, year is an optional argument"""
    API_KEY = 'f92fb218'
    Info = requests.get('http://www.omdbapi.com/?apikey=' + API_KEY + '&t=' + title + '&y=' + year).json()
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


def plot_corr(df):
    """Function to plot correlation matrix"""
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='viridis', linewidths=.1)
    plt.show()


# Code to demenstorate loading data from csv file
df_netflix = create_dateframe('/home/mukul/PycharmProjects/UCDAssement/data/netflix_titles.csv')
df_amazon = create_dateframe('/home/mukul/PycharmProjects/UCDAssement/data/amazon_prime_titles.csv')
# Code to demonstrate basic operations on pandas dataframe like find missing & duplicate
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
# Code to demonstrate merging dataframe and removing duplicate
"""
df_combined = pd.concat([df_netflix, df_amazon])
print(df_combined.info())
duplicates_combined = df_combined.duplicated(subset=['title'], keep=False)
print(df_combined[duplicates_combined].info())
#After combining the dataset we have found that there are 778 common titles in both the dataframe, now dropping duplicates
unique_movie = df_combined.drop_duplicates(subset=['title','director','cast'])
print(unique_movie.info())
"""

# Code to built histogram to check for missing values within two dataframe

# plot_histogram(df_netflix)
# plot_histogram(df_amazon)

# Code to built a histogram to compare release year between two dataset

# compare_release_year(df_netflix, df_amazon)

# Code to demonstrate loading data from API
"""
movie_list = ['Fast', 'Jungle', 'Godfather']
for i in movie_list:
    movie_info = get_movie(i)
    for i in movie_info:
        print(i + ': ', movie_info[i])
"""
# Code to demonstrate creating function, dictionary, using iterators
"""
result_counts = count_entries('./data/netflix_titles.csv', 10, 'release_year')
print(result_counts)
"""
# Code to demonstrate Machine Learning

diabetes_data = create_dateframe('/home/mukul/PycharmProjects/UCDAssement/data/diabetes.csv')
# print(diabetes_data.head())
# print(diabetes_data.info())
# print(diabetes_data.describe())
# From above analysis it's found tha there are some missing values and some outliers.

# Replace all missing values with NaN
diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
# print(diabetes_data.describe())

total_mising = diabetes_data.isnull().sum().sort_values(ascending=False)
# print(total_mising)

# diabetes_data.hist(figsize= (20,20))
# plt.show()

# Replace NaN values of columns as per distribution
diabetes_data['Glucose'].fillna(diabetes_data['Glucose'].mean(), inplace=True)
diabetes_data['BloodPressure'].fillna(diabetes_data['BloodPressure'].mean(), inplace=True)
diabetes_data['SkinThickness'].fillna(diabetes_data['SkinThickness'].mean(), inplace=True)
diabetes_data['Insulin'].fillna(diabetes_data['Insulin'].median(), inplace=True)
diabetes_data['BMI'].fillna(diabetes_data['BMI'].median(), inplace=True)

new_missing = diabetes_data.isnull().sum().sort_values(ascending=False)
# print('Check if there is any missing data after Imputing')
# print(new_missing)

# Code to draw a plot and check if there are any correlated values in dataset
# plot_corr(diabetes_data)
# plt.show()

# Plotting relationship b/w Age & Diabetes
#sns.countplot(x='Age',hue = 'Outcome', data=diabetes_data)
#plt.show()


#Splitting Data in Train & Test set
X_train, X_test, y_train, y_test = train_test_split(diabetes_data,diabetes_data['Outcome'], test_size=0.30, random_state=25)
neighbors = np.arange(1,40)
test_scores = np.empty(len(neighbors))
train_scores = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    train_scores[i] = knn.score(X_train, y_train)
    test_scores[i] = knn.score(X_test, y_test)

plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_scores, label = 'Testing Accuracy')
plt.plot(neighbors, train_scores, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
#plt.show()

#From Above plot we can conclude that the best result is obtained at k=14
#Train the model with best value of k
knn = KNeighborsClassifier(n_neighbors = 14)
knn.fit(X_train, y_train)
#print(knn.score(X_test, y_test))

# Code to demonstrate HyperParameter Tuning
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=2)
knn_cv.fit(X_train, y_train)
print('Best Params :- ', end='')
print(knn_cv.best_params_)
print('Best Score :- ', end='' )
print(knn_cv.best_score_)

