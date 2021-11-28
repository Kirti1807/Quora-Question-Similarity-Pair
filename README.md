# Quora-Question-Similarity-Pair
### Predicting that 2 Quora question are similar or not 
### Data source: https://www.kaggle.com/c/quora-question-pairs 
### Python Libarary used: nltk , numpy, pandas , matplotlib , sklearn , seaborn 
### Step performed 
  ##### -> Data reading and basic data Exploration 
  ##### -> Data Preprocessing 
      . Lower the text 
      . Replace number to string 
      . Removing Urls and hash tags 
      . Replaceing some special character to their string equalent 
      . decontrate word 
      . Removing punctuations 
      . Removing Stopword 
      . Lemmatization 
  ##### -> Feature Engineering 
      . Added basic features by feature extraction from text 
      . Added fuzzywuzzy features <br>
  ##### -> Exploratery Data Analysis <br>
      . Visualizing all features with respect to class <br>
      . Removing some features which have same distribution <br>
      . Visualizing Word Clous <br>
  ##### -> Vectorization <br>
      . TF-IDF vector <br>
      . Word2Vec  <br>
  ##### -> Model Building <br>
      . XgBoost Model <br>
  ##### -> Error Analysis <br>
      . Calculate Confusion matrix, Recall , Precision , F1-score , Log loss <br>
      . Visualize Confusion metrix , Recall , Precision <br>
      . Visualize Roc curve <br>
 
