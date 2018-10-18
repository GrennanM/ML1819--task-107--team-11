#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def boxplot(column):
    # to do: return a boxplot of each variables
    return 0

def plotHist(column, title, x_label, y_label):
    # to do: plot histogram for each individual variable
    binwidth = [x for x in range(0,20000, 2000)]
    ex = plt.hist(column, bins=binwidth)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return plt.show()

def plotHistTwo(colA, colB, title="", x_label="", y_label="frequency"):
    # to do: plot a two way histogram for male female for each variable
    binwidth = [x for x in range(0,30000, 1000)]
    # plt.hist(colA, bins=binwidth, alpha=0.5, label = "favNumberMales")
    # plt.hist(colB, bins=binwidth, alpha=0.5, label = "favNumberFemales")
    plt.hist([colA, colB], bins=binwidth, alpha=0.5, label=["tweetCountMales", "tweetCountFemales"])
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return plt.show()

def scatter(col1, col2):
    # to do: plot a scatter plot for variables. E.g. hue vs brightness with
    # male and female colored differently
    return 0

def main():

    startTime = time.time()
    # load the dataset and remove unnecessary columns and NA row
    dataset = '/home/markg/Documents/TCD/ML/ML1819--task-107--team-11/dataset/default_color_dataset.csv'
    data = pd.read_csv(dataset, na_values = '?')
    data['tweet_location'] = data['tweet_location'].astype('category')
    data['user_timezone'] = data['user_timezone'].astype('category')
    ### change the time
    print (data.dtypes)
    # print (np.dtype(data['tweet_location']))
    # data = data.drop(['Unnamed: 10', 'Unnamed: 11'], axis=1)
    # data = data.drop(data.index[9993])

    # create a subset of males and females
    males = data[data['gender']==0]
    females = data[data['gender']==1]

    # to access specific columns
    favNumberMales = males.loc[:,'fav_number']
    favNumberFemales = females.loc[:,'fav_number']
#    plotHistTwo(favNumberMales, favNumberFemales)

    tweetCountMales = males.loc[:,'tweet_count']
    tweetCountFemales = females.loc[:,'tweet_count']
    # plotHistTwo(tweetCountMales, tweetCountFemales)

    # retweetCountMales = males.loc[:,'retweet_count']
    # retweetCountFemales = females.loc[:,'retweet_count']

    # plot a histogram
    #plot_hist(fav_number, "title", "favourited tweets", "freq")

    # to keep track of user_time
    endTIme = time.time()

    totalTime = startTime - endTIme
    print(totalTime)
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
