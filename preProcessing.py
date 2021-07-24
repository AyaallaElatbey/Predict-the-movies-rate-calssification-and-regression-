#Importing libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
import random
import secrets

import os
def getdata():
    #Importing dataset
    df = pd.read_csv('Movies_testing.csv')
    # print(df.shape)
    #drop the const. coulm
    # df = df.loc[:, df.apply(pd.Series.nunique) != 1]
    df.drop(labels=['Title','Country','Language','Directors'], axis=1, inplace=True)
    df.drop(labels=['Type'], axis=1, inplace=True)
    # df.drop(labels=['Hulu', 'Prime Video', 'Disney+','Netflix'], axis=1, inplace=True)
    #drop rows that none data in these coulmns
    # df = df.dropna(subset=['IMDb', 'Runtime', 'Country','Language','Directors','Genres','Year',])
    # replace nan to str nan
    age=['+18','+16','0','+10','+12','all']
    df['Age'] = df['Age'].replace(np.nan, secrets.choice(age))
    rotten =['88%','90%','99%','8%','68%','58%','48%','38%','18%','0']
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].replace(np.nan, secrets.choice(rotten))
    #lablel to num
    number = preprocessing.LabelEncoder()
    # df['Directors'] = number.fit_transform(df['Directors'])
    df['Age'] = number.fit_transform(df['Age'])
    df['Rotten Tomatoes'] = number.fit_transform(df['Rotten Tomatoes'])
    # df['Country'] = number.fit_transform(df['Country'])
    # df['Directors'] = number.fit_transform(df['Directors'])
    # df['Genres'] = number.fit_transform(df['Genres'])
    # df['Language'] = number.fit_transform(df['Language'])

    # one_hot encoding
    df1 = df
    GF = df1.Genres.str.split(r'\s*,\s*', expand=True).apply(pd.Series.value_counts, 1) .iloc[:, 1:].fillna(0, downcast='infer')
    df = pd.concat([df1, GF.reindex(df.index)], axis=1, join='inner')

    # CF = df.Country.str.split(r'\s*,\s*', expand=True).apply(pd.Series.value_counts, 1).iloc[:, 1:].fillna(0,downcast='infer')
    # df2 = pd.concat([df1, CF.reindex(df1.index)], axis=1, join='inner')
    #
    # LF = df.Language.str.split(r'\s*,\s*', expand=True).apply(pd.Series.value_counts, 1).iloc[:, 1:].fillna(0,downcast='infer')
    # df = pd.concat([df1, LF.reindex(df1.index)], axis=1, join='inner')


    #scaling the data
    float_array = pd.Series(df['Year']).values.astype(float).reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(float_array)
    df_normalized = pd.DataFrame(scaled_array)
    df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    float_array = pd.Series(df['Age']).values.astype(float).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(float_array)
    df_normalized = pd.DataFrame(scaled_array)
    df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    float_array = pd.Series(df['Rotten Tomatoes']).values.astype(float).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(float_array)
    df_normalized = pd.DataFrame(scaled_array)
    df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    float_array = pd.Series(df['Runtime']).values.astype(float).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(float_array)
    df_normalized = pd.DataFrame(scaled_array)
    df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    # float_array = pd.Series(df['Country']).values.astype(float).reshape(-1, 1)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # scaled_array = min_max_scaler.fit_transform(float_array)
    # df_normalized = pd.DataFrame(scaled_array)
    # df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    # float_array = pd.Series(df['Genres']).values.astype(float).reshape(-1, 1)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # scaled_array = min_max_scaler.fit_transform(float_array)
    # df_normalized = pd.DataFrame(scaled_array)
    # df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    # float_array = pd.Series(df['Language']).values.astype(float).reshape(-1, 1)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # scaled_array = min_max_scaler.fit_transform(float_array)
    # df_normalized = pd.DataFrame(scaled_array)
    # df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')

    # float_array = pd.Series(df['Directors']).values.astype(float).reshape(-1, 1)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # scaled_array = min_max_scaler.fit_transform(float_array)
    # df_normalized = pd.DataFrame(scaled_array)
    # df = pd.concat([df_normalized, df.reindex(df_normalized.index)], axis=1, join='inner')


    #make the rate last column
    rate = df['IMDb']
    df['IMDb_rate']=rate
    # drop all unuse cloumns
    # df.drop(labels=['IMDb', 'Age', 'Rotten Tomatoes', 'Runtime','Title','Language','Country','Genres','Directors','Year',], axis=1, inplace=True)

    df.drop(labels=['Genres'], axis=1, inplace=True)

    df= df.dropna(0)
    print(df.shape)
    # print(df.info())
    df.to_csv('PreprocecedMovies_testing.csv')

getdata()
# def preProcessing():
#     if (os.path.exists("PreprocecedMovies_testing.csv")): # If you have already created the dataset:
#         data= pd.read_csv("PreprocecedMovies_testing.csv")
#
#     else:
#         getdata()
#         data= pd.read_csv("PreprocecedMovies_testing.csv")
#     return data