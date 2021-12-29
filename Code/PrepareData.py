import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import re
import seaborn as sns
from BasicDataExploration import BasicDataExploration
from DataPreprocessing import TextPreprocessing
from FeatureEngineering import FeatureEngineering
from ExplorateryDataAnalysis import EDA
from Vectorization import Vectorizer
from sklearn.model_selection import train_test_split

class PrepareData:
    def __init__(self, data):
        self.data = data

    def run_prepare_data(self):
        
        # Basic data exploration
        bde = BasicDataExploration(self.data)
        bde.basic_data_exploration()
        bde.basic_data_visualization(self.data)

        # Data preprocessing 
        tp = TextPreprocessing(self.data)
        data_ = tp.text_preprocessing(self.data , 'question1')
        data__ = tp.text_preprocessing(data_ , 'question2')

        # I am a disco dancer and I like to dance. 
        # kya bat hai sir XD
        # feature engineering
        fe = FeatureEngineering(data__)
        df = fe.feature_extraction(data__)
        df = fe.add_fuzzywuzzy_features(df)
        print(df.head())

        # EDA
        '''
                if you want to see the plots of features then please remove the comments from the below line
        '''
        eda = EDA(df)
        #eda.visualize_features()
        df = eda.dropping_features()
        #eda.showing_wordcloud() 

        # Vectorization

        vt = Vectorizer(self.data)
        data = vt.text_vectorization()
        result = vt.merging_features(df , data)

        
        return result
        











