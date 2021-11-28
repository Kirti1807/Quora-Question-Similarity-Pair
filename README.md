# Quora-Question-Similarity-Pair
Predicting that 2 Quora question are similar or not \n
Data source: https://www.kaggle.com/c/quora-question-pairs \n
Python Libarary used: nltk , numpy, pandas , matplotlib , sklearn , seaborn \n
Step performed
  -> Data reading and basic data Exploration
  -> Data Preprocessing
      . Lower the text
      . Replace number to string
      . Removing Urls and hash tags
      . Replaceing some special character to their string equalent
      . decontrate word
      . Removing punctuations
      . Removing Stopword
      . Lemmatization
  -> Feature Engineering
      . Added basic features by feature extraction from text
      . Added fuzzywuzzy features
  -> Exploratery Data Analysis
      . Visualizing all features with respect to class
      . Removing some features which have same distribution
      . Visualizing Word Clous
  -> Vectorization
      . TF-IDF vector
      . Word2Vec 
  -> Model Building
      . XgBoost Model
  -> Error Analysis
      . Calculate Confusion matrix, Recall , Precision , F1-score , Log loss
      . Visualize Confusion metrix , Recall , Precision
      . Visualize Roc curve
