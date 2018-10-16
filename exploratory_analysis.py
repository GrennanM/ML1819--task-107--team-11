# the below needs to be done for each of our two datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def boxplot(column):
    # to do: return a boxplot of each variables
    return 0

def plot_hist(column):
    # to do: plot histogram for each individual variable
    return 0

def plot_hist_two(column, y):
    # to do: plot a two way histogram for male female for each variable
    return 0

def scatter(col1, col2):
    # to do: plot a scatter plot for variables. E.g. hue vs brightness with
    # male and female colored differently
    return 0

def main():
    # load the dataset
    dataset = '/home/markg/Documents/TCD/ML/ML1819--task-107--team-11/dataset/twitter_gender_colors.csv'
    data = pd.read_csv(dataset, na_values = '?')
    print (data)

if __name__ == '__main__':
  main()


# the results above should indicate which are important variables &
# allow us to identify outliers

# see notes on repl
# to do: compute decision tree with chosen dependent variables
# return: recall precision and f1 score for decision tree

# to do: compute logisitic regression
# return: recall precision and f1
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html.

# to do: compute SVM
# return: recall precision and f1
