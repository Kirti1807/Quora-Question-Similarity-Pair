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

    def basic_data_exploration(self):
        print(self.data.shape)
        print(self.data.head())
        # checking for null values
        print(self.data.isnull().sum())
        # droping the null values
        self.data.dropna(inplace=True)

        # after removing null values checking again the shape and other info
        print(self.data.shape)
        print(self.data.isnull().sum())
        # print(data.describe())
        # print(data.info())
        self.data.reset_index(drop=True , inplace=True)
        # removind rows in which question having float valus
        
        for i in range(len(self.data)):
            if (type(self.data['question1'][i]) != str) or (type(self.data['question2'][i]) != str):
                self.data.drop(i , inplace=True)
        
        # self.data.drop(self.data[((type(self.data['question1']) != str) or (type(self.data['question2']) != str))].iloc[0] , inplace = True) 

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


# if __name__ == "__main__": 
#     data = pd.read_csv(r'D:\ML_Projects\Quora-Question-Similarity-Pair\data\train.csv') 
#     data_exp = BasicDataExploration(data) 
#     data_exp.basic_data_exploration() 
