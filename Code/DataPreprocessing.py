import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import re
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessing:
    
    def __init__(self , data):
        self.data = data
        self.my_stopword = list(stopwords.words('english'))
        self.my_lemmatizer = WordNetLemmatizer()
    
    def text_preprocessing(self , data , column_name):
        
        data_copy = data.copy()
        
        data_copy[column_name] = data_copy[column_name].apply(self.lower_text)
        data_copy[column_name] = data_copy[column_name].apply(self.remove_punctuation)
        data_copy[column_name] = data_copy[column_name].apply(self.replace_numeric_to_string)
        data_copy[column_name] = data_copy[column_name].apply(self.remove_urls)
        data_copy[column_name] = data_copy[column_name].apply(self.replace_special_character_to_string_equalent)
        data_copy[column_name] = data_copy[column_name].apply(self.decontrate_words)
        data_copy[column_name] = data_copy[column_name].apply(self.remove_stopwords)
        data_copy[column_name] = data_copy[column_name].apply(self.text_lemmatization)
        
        return data_copy
        
    
    def lower_text(self , text):
        return text.lower().strip()
    
    def remove_punctuation(self , text):
        text = text.translate(str.maketrans("", "", string.punctuation)) 
        return text
    
    def replace_numeric_to_string(self, text):
        text = text.replace(',000,000,000 ', 'b ')
        text = text.replace(',000,000 ', 'm ')
        text = text.replace(',000', 'k ')
        text = re.sub(r'([0-9]+)000000000' , r'\1b', text)
        text = re.sub(r'([0-9]+)000000' , r'\1m', text)
        text = re.sub(r'([0-9]+)000' , r'\1k', text)
        return text
    
    def remove_urls(self, text):
        text = re.sub(r"http\S+", "", text)
        return text
    
    def replace_special_character_to_string_equalent(self , text):
        text = text.replace('%' , ' percent')
        text = text.replace('$' , ' dollar')
        text = text.replace('₹' , ' repee')
        text = text.replace('€' , ' euro')
        text = text.replace('@' , ' at')
        return text
    
    def remove_stopwords(self , text):
        words = text.split()
        new_words = [word for word in words if not word in self.my_stopword]
        text = ' '.join(new_words)
        return text
    
    def text_lemmatization(self , text):
        text = ' '.join([self.my_lemmatizer.lemmatize(word) for word in text.split()])
        return text
    
    def decontrate_words(self , text):
        contractions = { 
            "ain't": "am not",
            "aren't": "are not",
            "can't": "can not",
            "can't've": "can not have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'd've": "i would have",
            "i'll": "i will",
            "i'll've": "i will have",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so as",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you would",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have"
           }
        
        text_decontrate = []
        
        for word in text.split():
            if word in contractions:
                word = contractions[word]
            text_decontrate.append(word)
    
        text = ' '.join(text_decontrate)
        
        return text
    