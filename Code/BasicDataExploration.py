import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import re
import seaborn as sns

class BasicDataExploration:
    def __init__(self , data):
        self.data = data

    def basic_data_exploration(self , data):
        print(data.shape)
        print(data.head())
        # checking for null values
        print(data.isnull().sum())
        # droping the null values
        data = data.dropna()

        # after removing null values checking again the shape and other info
        print(data.shape)
        print(data.isnull().sum())
        print(data.describe())
        print(data.info())

    def basic_data_visualization(self , data):
        print(data.is_duplicate.value_counts())
        print(data.is_duplicate.value_counts()/data.is_duplicate.value_counts().sum()*100)

        qids = np.append(data.qid1.values , data.qid2.values)
        print("total number of qids are",qids.shape)

        print("total no of unique question in data set: ", len(set(qids)))

        # plotting the occurance of question
        occurences = np.bincount(qids)
        plt.figure(figsize=(10,5)) 
        plt.hist(occurences, bins=range(0,160))
        plt.yscale('log')
        plt.xlabel('Number of times question repeated')
        plt.ylabel('Number of questions')
        plt.title('Question vs Repeatition')
        plt.show()
        print("Minimun occurences of any question: " , np.min(occurences))
        print("Maximum occurences of any question: " , np.max(occurences))