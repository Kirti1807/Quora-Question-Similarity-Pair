import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import re
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords


from numpy import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import spacy

nlp = spacy.load('en_core_web_sm')

class Vectorizer:
    def __init__(self , data):
        self.data = data

    def text_vectorization(self):

        data_copy = self.data
        # Merge texts
        questions = list(data_copy['question1'])+list(data_copy['question2'])

        tfidf = TfidfVectorizer(lowercase=False)
        tfidf.fit_transform(questions)

        #  dict key:word and value:tf-idf score
        word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))


        vecs1 = []

        # https://github.com/noamraph/tqdm
        # tqdm is used to print the progress bar
        for qu1 in tqdm(list(data_copy['question1'])):
            doc1 = nlp(qu1) 
            # 384 is the number of dimensions of vectors 
            mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])
            for word1 in doc1:
                # Word2Vec
                vec1 = word1.vector
                # Fetch df score
                try: idf = word2tfidf[str(word1)]
                except: idf = 0
                # Compute final vec
                mean_vec1 += vec1 * idf
            mean_vec1 = mean_vec1.mean(axis=0)
            vecs1.append(mean_vec1)
        data_copy['q1_feats_m'] = list(vecs1)

        vecs2 = []
        for qu2 in tqdm(list(data_copy['question2'])):
            doc2 = nlp(qu2) 
            mean_vec2 = np.zeros([len(doc1), len(doc2[0].vector)])
            for word2 in doc2:
                # Word2Vec
                vec2 = word2.vector
                # Fetch df score
                try: idf = word2tfidf[str(word2)]
                except: idf = 0
                # Compute final vec
                mean_vec2 += vec2 * idf
            mean_vec2 = mean_vec2.mean(axis=0)
            vecs2.append(mean_vec2)
        data_copy['q2_feats_m'] = list(vecs2)

        return data_copy

    def merging_features(df,data_):
        
        df1 = df.drop(['qid1' , 'qid2' , 'question1' , 'question2'] , axis=1)
        df2 = data_.drop(['qid1' , 'qid2' , 'question1' , 'question2' , 'is_duplicate'], axis=1)

        # print(df1.shape)
        # print(df2.shape)

        df2_q1 = pd.DataFrame(df2.q1_feats_m.values.tolist() , index = df2.index)
        df2_q2 = pd.DataFrame(df2.q2_feats_m.values.tolist() , index = df2.index)

        # print(df2_q1.shape)
        # print(df2_q2.shape)

        df2_q1['id'] = df1['id']
        df2_q2['id'] = df1['id']

        df2 = df2_q1.merge(df2_q2 , on='id' , how='left')

        #df2.shape

        result = df1.merge(df2 , on='id' , how='left')

        #result.shape
        #result.head()

        return result

