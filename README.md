# Quora-Question-Similarity-Pair
Predicting that 2 Quora question are similar or not <br>
Data source: https://www.kaggle.com/c/quora-question-pairs <br>
Python Libarary used: nltk , numpy, pandas , matplotlib , sklearn , seaborn <br>
Step performed <br>
  -> Data reading and basic data Exploration <br>
  -> Data Preprocessing <br>
      . Lower the text <br>
      . Replace number to string <br>
      . Removing Urls and hash tags <br>
      . Replaceing some special character to their string equalent <br>
      . decontrate word  <br>
      . Removing punctuations <br>
      . Removing Stopword <br>
      . Lemmatization <br>
  -> Feature Engineering <br>
      . Added basic features by feature extraction from text <br>
      . Added fuzzywuzzy features <br>
  -> Exploratery Data Analysis <br>
      . Visualizing all features with respect to class <br>
      . Removing some features which have same distribution <br>
      . Visualizing Word Clous <br>
  -> Vectorization <br>
      . TF-IDF vector <br>
      . Word2Vec  <br>
  -> Model Building <br>
      . XgBoost Model <br>
  -> Error Analysis <br>
      . Calculate Confusion matrix, Recall , Precision , F1-score , Log loss <br>
      . Visualize Confusion metrix , Recall , Precision <br>
      . Visualize Roc curve <br>
 
