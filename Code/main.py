import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import re
import seaborn as sns
from ErrorAnalysis import EvaluateModel
from PrepareData import PrepareData
from ModelTraining import ModelTraining
from sklearn.model_selection import train_test_split
import joblib

def main():
    data = pd.read_csv(r'D:\ML_Projects\Quora-Question-Similarity-Pair\data\train.csv')
    pre_data = PrepareData(data)
    result = pre_data.run_prepare_data()
    # print(result.shape)
    # print(result.head())
    # print(result.isnull().sum().sum())
    result = result.dropna()
    #print(result.shape)

    # data splitting
    result = result.drop('id' , axis=1)

    X = result.drop('is_duplicate' , axis=1)
    Y = result['is_duplicate']


    #print(X.shape , Y.shape)
    x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.3 , random_state=1)
    #print(x_train.shape , x_test.shape , y_train.shape , y_test.shape)


    # ------------------------------------model evaluating-----------------------------
    mt = ModelTraining(x_train , y_train)

    
    # XgBoost Model
    xgb = mt.Xgboost_model(fine_tuning=False)
    y_predict_xgb = xgb.predict(x_test)

    evaluate_xgb = EvaluateModel(x_test , y_test , xgb)
    evaluate_xgb.evaluate_model()
    evaluate_xgb.plot_confusion_matrix(y_test , y_predict_xgb)
    evaluate_xgb.plot_roc_curve(y_test , xgb.predict_proba(x_test)[: , 1])

    # # Logistic Regression model
    # log_reg = mt.logistic_regression_model()
    # y_predict_lr = log_reg.predict(x_test)

    # evaluate_lr = EvaluateModel(x_test , y_test , log_reg)
    # evaluate_lr.evaluate_model()
    # evaluate_lr.plot_confusion_matrix(y_test , y_predict_lr)
    # evaluate_lr.plot_roc_curve(y_test , log_reg.predict_proba(x_test)[: , 1])

    # # SVM
    # svm = mt.svm_model(fine_tuning=False)
    # y_predict_svm = svm.predict(x_test)

    # evaluate_svm = EvaluateModel(x_test , y_test , svm)
    # evaluate_svm.evaluate_model()
    # evaluate_svm.plot_confusion_matrix(y_test , y_predict_svm)
    # evaluate_svm.plot_roc_curve(y_test , svm.predict_proba(x_test)[: , 1])

    # # Naive Bayes
    # nb = mt.Naive_Bayes()
    # y_predict_nb = nb.predict(x_test)

    # evaluate_nb = EvaluateModel(x_test , y_test , nb)
    # evaluate_nb.evaluate_model()
    # evaluate_nb.plot_confusion_matrix(y_test , y_predict_nb)
    # evaluate_nb.plot_roc_curve(y_test , nb.predict_proba(x_test)[: , 1])

    # # KNN
    # knn = mt.KNN_model(finetuning=False)
    # y_predict_knn = knn.predict(x_test)

    # evaluate_knn = EvaluateModel(x_test , y_test , knn)
    # evaluate_knn.evaluate_model()
    # evaluate_knn.plot_confusion_matrix(y_test , y_predict_knn)
    # evaluate_knn.plot_roc_curve(y_test , knn.predict_proba(x_test)[: , 1])


if __name__ == "__main__":
    main()









	


	
	

	