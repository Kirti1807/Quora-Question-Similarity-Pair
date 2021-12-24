import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import re
import seaborn as sns
from PrepareData import PrepareData
def main():

	data = pd.read_csv('D:\ML_Projects\Quora-Question-Similarity-Pair\data\train.csv')

	pre_data = PrepareData(data)

	x_train , x_test , y_train , y_test = pre_data.run_prepare_data()
	