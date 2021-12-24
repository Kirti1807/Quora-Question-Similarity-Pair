import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import re
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords

class EDA:
    def __init__(self , data) -> None:
        self.df = data

    def visualize_features(self):
        # 1
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='length_of_q1' , data=self.df)
        plt.title('Box Plot - Length of Question 1')
        plt.xlabel('Class label')
        plt.ylabel('Number of characters')

        plt.subplot(1,2,2)

        sns.kdeplot(x='length_of_q1' , hue='is_duplicate' , data=self.df , shade=True)
        plt.title('PDF - Length of qQuestion 1')
        plt.xlabel('Number of characters')

        plt.show()

        # 2
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='length_of_q2' , data=self.df)
        plt.title('Box Plot - Length of Question 2')
        plt.xlabel('Class label')
        plt.ylabel('Number of characters')

        plt.subplot(1,2,2)

        sns.kdeplot(x='length_of_q2' , hue='is_duplicate' , data=self.df , shade=True)
        plt.title('PDF - Length of qQuestion 2')
        plt.xlabel('Number of characters')

        plt.show()


        # 3
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='total_number_of_words_in_q1' , data=self.df)
        plt.title('Box Plot - Total number of words in question1')
        plt.xlabel('Class label')
        plt.ylabel('Total number of words in question 1')

        plt.subplot(1,2,2)

        sns.kdeplot(x='total_number_of_words_in_q1' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Total number of words in question1')
        plt.xlabel('Total number of words in question1')

        plt.show()

        # 4
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='total_number_of_words_in_q2' , data=self.df)
        plt.title('Box Plot - Total number of words in question2')
        plt.xlabel('Class label')
        plt.ylabel('Total number of words in question 2')

        plt.subplot(1,2,2)

        sns.kdeplot(x='total_number_of_words_in_q2' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Total number of words in question2')
        plt.xlabel('Total number of words in question2')

        plt.show()

        # 5
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='sum_of_total_words_of_q1_and_q2' , data=self.df)
        plt.title('Box Plot - Total number of words in both question')
        plt.xlabel('Class label')
        plt.ylabel('Sum of words of question1 and question 2')

        plt.subplot(1,2,2)

        sns.kdeplot(x='sum_of_total_words_of_q1_and_q2' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Total number of words in both question')
        plt.xlabel('Sum of words of question1 and question 2')

        plt.show()

        # 6
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='number_of_unique_words_in_q1' , data=self.df)
        plt.title('Box Plot - Total number of unique words in question1')
        plt.xlabel('Class label')
        plt.ylabel('Total number of unique words in question 1')

        plt.subplot(1,2,2)

        sns.kdeplot(x='number_of_unique_words_in_q1' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Total number of unique words in question1')
        plt.xlabel('Total number of unique words in question1')

        plt.show()

        # 7

        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='number_of_unique_words_in_q2' , data=self.df)
        plt.title('Box Plot - Total number of unique words in question2')
        plt.xlabel('Class label')
        plt.ylabel('Total number of unique words in question 2')

        plt.subplot(1,2,2)

        sns.kdeplot(x='number_of_unique_words_in_q2' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Total number of unique words in question2')
        plt.xlabel('Total number of unique words in question2')

        plt.show()

        # 8
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='sum_of_total_uinque_words_of_q1_and_q2' , data=self.df)
        plt.title('Box Plot - Total number of unique words in both question')
        plt.xlabel('Class label')
        plt.ylabel('Sum of unique words of question1 and question 2')

        plt.subplot(1,2,2)

        sns.kdeplot(x='sum_of_total_uinque_words_of_q1_and_q2' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Total number of unique words in both question')
        plt.xlabel('Sum of unique words of question1 and question 2')

        plt.show()

        # 9
        
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='ratio_of_total_unique_words_and_total_words', data=self.df)
        plt.title('Box Plot - Ratio of total unique words and total Words ')
        plt.xlabel('Class label')
        plt.ylabel('Ratio of total unique words and total words')

        plt.subplot(1,2,2)

        sns.kdeplot(x='ratio_of_total_unique_words_and_total_words', hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Ratio of total unique words and total Words ')
        plt.xlabel('Ratio of total unique words and total words')

        plt.show()

        #10

        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='number_of_common_words_in_q1_and_q2', data=self.df)
        plt.title('Box Plot - Commom words in both questions')
        plt.xlabel('Class label')
        plt.ylabel('number of common word')

        plt.subplot(1,2,2)

        sns.kdeplot(x='number_of_common_words_in_q1_and_q2', hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Commom words in both questions')
        plt.xlabel('number of common word')

        plt.show()

        #11

        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='ratio_of_common_words_of_q1q2_and_total_words' , data=self.df)
        plt.title('Box Plot - Ratio of common words and total words')
        plt.xlabel('Class label')
        plt.ylabel('Ratio of common words and total words')

        plt.subplot(1,2,2)

        sns.kdeplot(x='ratio_of_common_words_of_q1q2_and_total_words' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Ratio of common words and total words')
        plt.xlabel('Ratio of common words and total words')

        plt.show()

        #12

        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='ratio_of_common_words_of_q1q2_and_total_unique_words' , data=self.df)
        plt.title('Box Plot - Ratio of common words and total unique words')
        plt.xlabel('Class label')
        plt.ylabel('Ratio of common words and total unique words')

        plt.subplot(1,2,2)

        sns.kdeplot(x='ratio_of_common_words_of_q1q2_and_total_unique_words' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Ratio of common words and total unique words')
        plt.xlabel('Ratio of common words and total unique words')

        plt.show()

        #13

        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='ratio_of_common_words_and_length_of_smaller_question' , data=self.df)
        plt.title('Box Plot - Ratio of commons words and small length question')
        plt.xlabel('Class label')
        plt.ylabel('Ratio of commons words and small length question')

        plt.subplot(1,2,2)

        sns.kdeplot(x='ratio_of_common_words_and_length_of_smaller_question' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Ratio of commons words and small length question')
        plt.xlabel('Ratio of commons words and small length question')

        plt.show()

        #14

        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='ratio_of_common_words_and_length_of_larger_question' , data=self.df)
        plt.title('Box Plot - Ratio of commons words and large length question')
        plt.xlabel('Class label')
        plt.ylabel('Ratio of commons words and large length question')

        plt.subplot(1,2,2)

        sns.kdeplot(x='ratio_of_common_words_and_length_of_larger_question' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Ratio of commons words and large length question')
        plt.xlabel('Ratio of commons words and large length question')

        plt.show()

        #15

        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='fuzz_ratio' , data=self.df)
        plt.title('Box Plot - Fuzz Ratio')
        plt.xlabel('Class label')
        plt.ylabel('fazz ratio')

        plt.subplot(1,2,2)

        sns.kdeplot(x='fuzz_ratio' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Fuzz Ratio')
        plt.xlabel('fuzz ratio')

        plt.show()

        #16

        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='fuzz_partial_ratio' , data=self.df)
        plt.title('Box Plot - Fuzz Partial Ratio')
        plt.xlabel('Class label')
        plt.ylabel('fazz partial ratio')

        plt.subplot(1,2,2)

        sns.kdeplot(x='fuzz_partial_ratio' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Fuzz Partial Ratio')
        plt.xlabel('fuzz partial ratio')

        plt.show()

        #17

        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='fuzz_token_sort_ratio' , data=self.df)
        plt.title('Box Plot - Fuzz Token Sort Ratio')
        plt.xlabel('Class label')
        plt.ylabel('fazz token sort ratio')

        plt.subplot(1,2,2)

        sns.kdeplot(x='fuzz_token_sort_ratio' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Fuzz Token Sort Ratio')
        plt.xlabel('fuzz token sort ratio')

        plt.show()

        #18

        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)

        sns.boxplot(x='is_duplicate' , y='fuzz_token_set_ratio' , data=self.df)
        plt.title('Box Plot - Fuzz Token Set Ratio')
        plt.xlabel('Class label')
        plt.ylabel('fazz token set ratio')

        plt.subplot(1,2,2)

        sns.kdeplot(x='fuzz_token_set_ratio' , hue='is_duplicate' , data=self.df , shade=False)
        plt.title('PDF - Fuzz Token Set Ratio')
        plt.xlabel('fuzz token set ratio')

        plt.show()


    def dropping_features(self):
        #'ratio_of_common_words_of_q1q2_and_total_words' and 'ratio_of_common_words_of_q1q2_and_total_unique_words' 
        # features are giving same distribution so droping 
        # 'ratio_of_common_words_of_q1q2_and_total_unique_words' feature

        df_copy = self.df
        df_copy = df_copy.drop('ratio_of_common_words_of_q1q2_and_total_unique_words' , axis=1)

        # 'fuzz_ratio' and 'fuzz_token_sort_ratio'
        # features are giving same distribution so droping 'fuzz_ratio'

        df_copy = df_copy.drop('fuzz_ratio' , axis=1)
    
        return df_copy

    def showing_wordcloud(self):
        duplicate = self.df[self.df['is_duplicate']==1]
        non_duplicate = self.df[self.df['is_duplicate']==0]

        duplicate = np.array([duplicate['question1'] , duplicate['question2']]).flatten()
        non_duplicate = np.array([non_duplicate['question1'], non_duplicate['question2']]).flatten()

        dup_str = ' '.join(duplicate)
        non_dup_str = ' '.join(non_duplicate)

        stop_words = set(stopwords.words('english'))

        #Word cloud for duplicate pairs

        word_cloud = WordCloud(background_color='black' , max_words=len(dup_str) , stopwords=stop_words , width=600 , height=400)
        word_cloud.generate(dup_str)

        print('Word cloud for duplicate pairs')
        plt.figure(figsize=(15,8))
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis('off')

        plt.show()

        # Word cloud for non duplicate pairs
        word_cloud = WordCloud(background_color='black' , max_words=len(dup_str) , stopwords=stop_words , width=600 , height=400)
        word_cloud.generate(non_dup_str)

        print('Word cloud for non duplicate pairs')
        plt.figure(figsize=(15,8))
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis('off')

        plt.show()

    



    

    
    




