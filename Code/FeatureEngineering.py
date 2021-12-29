import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import re
import seaborn as sns
from fuzzywuzzy import fuzz

class FeatureEngineering:
    
    def __init__(self , data):
        self.data = data
        
    
    def feature_extraction(self , data):
        
        data_copy = data.copy()
        
        data_copy['length_of_q1'] = data_copy['question1'].apply(lambda text : len(text))
        
        data_copy['length_of_q2'] = data_copy['question2'].apply(lambda text : len(text))
        data_copy['length_of_q1_and_q2'] = data_copy.apply(lambda x : len(x.question1) + len(x.question2) , axis=1)   

        data_copy['length_of_q1_and_q2_difference'] = data_copy['length_of_q1'] - data_copy['length_of_q2'] 
        data_copy['length_of_q1_and_q2_difference'] = data_copy['length_of_q1_and_q2_difference'].apply(lambda x : abs(x)) 

        data_copy['total_number_of_words_in_q1'] = data_copy['question1'].apply(lambda text : len(nltk.word_tokenize(text)))
        
        data_copy['total_number_of_words_in_q2'] = data_copy['question2'].apply(lambda text : len(nltk.word_tokenize(text)))
        
        data_copy['sum_of_total_words_of_q1_and_q2'] = data_copy['total_number_of_words_in_q1'] + data_copy['total_number_of_words_in_q2']
        
        data_copy['number_of_unique_words_in_q1'] = data_copy['question1'].apply(lambda text : len(set(nltk.word_tokenize(text))))
        
        data_copy['number_of_unique_words_in_q2'] = data_copy['question2'].apply(lambda text : len(set(nltk.word_tokenize(text))))
        
        data_copy['sum_of_total_uinque_words_of_q1_and_q2'] = data_copy['number_of_unique_words_in_q1'] + data_copy['number_of_unique_words_in_q2']
        
        data_copy['ratio_of_total_unique_words_and_total_words'] = data_copy['sum_of_total_uinque_words_of_q1_and_q2']/data_copy['sum_of_total_words_of_q1_and_q2']
        
        data_copy['number_of_common_words_in_q1_and_q2'] = data_copy.apply(lambda x : len(set(x.question1.split()).intersection(set(x.question2.split()))) , axis=1)
        
        data_copy['ratio_of_common_words_of_q1q2_and_total_words'] = data_copy['number_of_common_words_in_q1_and_q2']/data_copy['sum_of_total_words_of_q1_and_q2']
        
        data_copy['ratio_of_common_words_of_q1q2_and_total_unique_words'] = data_copy['number_of_common_words_in_q1_and_q2']/data_copy['sum_of_total_uinque_words_of_q1_and_q2']
        
        data_copy['ratio_of_common_words_and_length_of_smaller_question'] = data_copy['number_of_common_words_in_q1_and_q2']/np.minimum(data_copy['length_of_q1'] , data_copy['length_of_q2'])
        
        data_copy['ratio_of_common_words_and_length_of_larger_question'] = data_copy['number_of_common_words_in_q1_and_q2']/np.maximum(data_copy['length_of_q1'] , data_copy['length_of_q2'])
        
        return data_copy
    
    def add_fuzzywuzzy_features(self , data):
        
        data_copy = data.copy()
        
        data_copy['fuzz_ratio'] = data_copy.apply(lambda x : fuzz.ratio(x.question1 , x.question2) , axis=1)
        
        data_copy['fuzz_partial_ratio'] = data_copy.apply(lambda x : fuzz.partial_ratio(x.question1 , x.question2) , axis=1)
    
        data_copy['fuzz_token_sort_ratio'] = data_copy.apply(lambda x : fuzz.token_sort_ratio(x.question1 , x.question2) , axis=1)
    
        data_copy['fuzz_token_set_ratio'] = data_copy.apply(lambda x : fuzz.token_set_ratio(x.question1 , x.question2) , axis=1)
        
        return data_copy
    