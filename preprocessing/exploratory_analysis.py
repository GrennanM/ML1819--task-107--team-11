#!/usr/bin/env python3 -w ignore DataConversionWarning
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from textAnalysis_ex import *
from setUp import *

def boxplot(column):
    # to do: return a boxplot of each variables
    return 0

def plotHist(column, title, x_label, y_label):
    # plots a histogram. Note: update bin width as appropriate
    binwidth = [x for x in range(0,800000, 2000)]
    ex = plt.hist(column, bins=binwidth)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return plt.show()

def plotHistTwo(colA, colB, title="", x_label="", y_label="Frequency"):
    # plots a histogram with two variables side-by-side
    # Note: update binwidth
    binwidth = [x for x in range(0,6, 1)]
    plt.hist([colA, colB], bins=binwidth, alpha=0.5, label=["Males", "Females"])
    plt.legend(loc='upper right', prop={'size': 13})
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.savefig("retweet_count.png")
    return plt.show()

def plotBar(colA, colB, title="", x_label="", y_label="Frequency"):
    return 0

def scatter(col1, col2):
    # to do: plot a scatter plot for variables. E.g. hue vs brightness with
    # male and female colored differently
    return 0

def main():
    #################### SETUP CODE ########################################
    # start time
    startTime = time.time()

#     # load the dataset
#     dataset = '/home/markg/Documents/TCD/ML/ML1819--task-107--team-11/dataset/overall_dataset.csv'
#     data = pd.read_csv(dataset, encoding='latin-1')
#
#     # reformat date column
#     data['created'] = pd.to_datetime(data['created'])
#
#     # create new columns for year and month % remove original column
#     data['year'] = pd.DatetimeIndex(data['created']).year
#     data['month'] = pd.DatetimeIndex(data['created']).month
#     data = data.drop(['created'], axis=1)
#     # data.drop(columns=['Unnamed: 0'], inplace = True)
#     # data.drop(columns = ['user_timezone', 'tweet_count', 'month', 'text_sent'], inplace=True)
#     #
#     # # reformat date column
#     # data['created'] = pd.to_datetime(data['created'])
#     #
#     # # create new columns for year and month
#     # data['year'] = pd.DatetimeIndex(data['created']).year
#     # data['month'] = pd.DatetimeIndex(data['created']).month
#     #
#     # # remove original date column
#     # data = data.drop(['created'], axis=1)
#     #
#     # # standardize numeric variables (could also consider using robust scaler here)
#     # numericVariables = ['fav_number', 'tweet_count','retweet_count', 'link_hue',
#     #  'link_sat', 'link_vue', 'sidebar_hue', 'sidebar_sat', 'sidebar_vue', 'year', 'month']
#     # scaler = preprocessing.StandardScaler()
#     # data[numericVariables] = scaler.fit_transform(data[numericVariables])
#
#     ##################### END SETUP CODE ######################################
#
#     #################### SVM MODEL ############################################
#     # # create dependent & independent variables
#     # X = data.drop(['gender', 'fav_number', 'user_timezone', 'tweet_count','retweet_count', 'link_hue',
#     # #  'link_sat', 'link_vue', 'sidebar_sat', 'sidebar_vue', 'month'], axis=1)
#     # # # X = data.drop('gender', axis=1)
#     # # y = data['gender']
#     # # # print (X.keys())
#     # #
#     # #
#     # # # split into 90% training, 10% testing
#     # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)
#     # #
#     # # # # train model (could change kernel here)
#     # # svm = SVC(C=1, gamma=0.3, kernel='rbf')
#     # # svm.fit(X_train, y_train)
#     # # #
#     # # # # # recursive feature selection using cross validation
#     # # # # rfecv = RFECV(estimator=svm, step=1, cv=StratifiedKFold(2),
#     # # # #               scoring='accuracy')
#     # # # # rfecv.fit(X, y)
#     # # # # print("Optimal number of features : %d" % rfecv.n_features_)
#     # # # # print("Feature ranking: ", rfecv.ranking_)
#     #
#     # # # recursive feature selection without cross validation
#     # rfe = RFE(svm, 3)
#     # fit = rfe.fit(X, y)
#     # print('Num Features:',fit.n_features_to_select)
#     # print("Selected Features:",fit.support_)
#     # # #
#     # # plot bar chart of feature ranking
#     # features = list(X)
#     # ranking = rfecv.ranking_
#     # plt.bar(features, ranking, align='center', alpha=0.5)
#     # plt.show()
#     #
#     # # Plot number of features VS. cross-validation scores
#     # plt.figure()
#     # plt.xlabel("Number of features selected")
#     # plt.ylabel("Cross validation score (nb of correct classifications)")
#     # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#     # plt.show()
#     # #
#     # # make predictions and print metrics
#     # y_pred = svm.predict(X_test)
#     # print(classification_report(y_test,y_pred))
#     # print(confusion_matrix(y_test,y_pred))
#     # # #
#     # # # # # cross validation to choose c and gamma
#     # C_s, gamma_s = np.meshgrid(np.logspace(-2, 1, 20), np.logspace(-2, 1, 20))
#     # scores = list()
#     # i=0; j=0
#     # for C, gamma in zip(C_s.ravel(),gamma_s.ravel()):
#     #     svm.C = C
#     #     svm.gamma = gamma
#     #     this_scores = cross_val_score(svm, X, y, cv=5)
#     #     scores.append(np.mean(this_scores))
#     # scores=np.array(scores)
#     # scores=scores.reshape(C_s.shape)
#     # fig2, ax2 = plt.subplots(figsize=(12,8))
#     # c=ax2.contourf(C_s,gamma_s,scores)
#     # ax2.set_xlabel('C')
#     # ax2.set_ylabel('gamma')
#     # fig2.colorbar(c)
#     # fig2.savefig('crossvalOverall.png')
#
#     ################## END SVM MODEL ##########################################
#
# #   # create a subset of males and females
#     males = data[data['gender']==0]
#     females = data[data['gender']==1]

    # retweetCountMales = data.loc[(data['gender'] == 0) & data['retweet_count']]
    # print (retweetCountMales.head(10))

    # to access specific columns
    # favNumberMales = males.loc[:,'fav_number']
    # favNumberFemales = females.loc[:,'fav_number']
    # plotHistTwo(favNumberMales, favNumberFemales,
    # x_label="Total Number of Tweets Favourited", title="Total Number of Tweets Favourited")

    # tweetCountMales = males.loc[:,'tweet_count']
    # tweetCountFemales = females.loc[:,'tweet_count']
    # plotHistTwo(tweetCountMales, tweetCountFemales,
    # x_label="Total Number of Tweets Posted", title="Total Number of Tweets Posted")

    # retweetCountMales = males.loc[:,'retweet_count'].value_counts()
    # retweetCountFemales = females.loc[:,'retweet_count'].value_counts()
    # print (retweetCountFemales)
    # plotHistTwo(retweetCountMales, retweetCountFemales,
    # x_label="Total Number of retweets Posted", title="Total Number of retweets Posted")

    # # plot bar char of retweet count
    # x = np.array([0, 1, 2])
    # malesRetweet = [4451, 154, 16]
    # femalesRetweet  = [5220, 117, 12]
    # width = 0.2
    # ax = plt.subplot(111)
    # rect1 = ax.bar(x, malesRetweet, width, align='center')
    # rect2 = ax.bar(x + width, femalesRetweet, width, align='center')
    # ax.set_title('Number of retweets')
    # ax.set_ylabel('Frequency')
    # ax.set_xlabel('Number of retweets')
    # ax.set_xticks(x + width / 2)
    # ax.set_xticklabels(('0', '1', '2'))
    # ax.legend( (rect1[0], rect2[0]), ('Male', 'Female') )
    # plt.savefig('retweet.png')
    # plt.show()
    # data.info()

    # to do: DATE
    # dateMales = females.loc[:,'year'].value_counts()
    x = np.array([2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008,
    2007, 2006])
    malesDate = [506, 513, 535, 659, 784, 528, 845, 205, 59, 3]
    femalesDate = [830, 728, 715, 872, 899, 494, 705, 100, 13, 0]
    width = 0.2
    ax = plt.subplot(111)
    rect1 = ax.bar(x, malesDate, width, align='center')
    rect2 = ax.bar(x + width, femalesDate, width, align='center')
    ax.set_title('Date of Registration')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Date of Registration')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels((2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008,
    2007, 2006))
    ax.legend( (rect1[0], rect2[0]), ('Male', 'Female') )
    plt.savefig('dateRegistration.png')
    plt.show()


    #################### LOGISTIC MODEL #######################################
    # create dependent & independent variables
    # X = data.drop('gender_catg', axis=1)
    # Y = data['gender_catg']
    #
    # model = LogisticRegression()
    # rfe = RFE(model, 3)
    # fit = rfe.fit(X, Y)
    # print('Num Features:',fit.n_features_to_select)
    # print("Selected Features:",fit.support_)
    #
    # # build model
    # logit_model=sm.Logit(Y,X)
    # result=logit_model.fit()
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    # logreg = LogisticRegression()
    # logreg.fit(X_train, y_train)
    #
    # y_pred = logreg.predict(X_test)
    # # print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    # print(classification_report(y_test,y_pred))
    # logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # # plt.savefig('Log_ROC')
    # plt.show()

    # to keep track of time taken
    endTIme = time.time()
    totalTime = endTIme - startTime
    print("Time taken:", totalTime)

if __name__ == '__main__':
  main()
