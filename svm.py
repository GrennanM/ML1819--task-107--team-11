#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from textAnalysis_ex import *
from setUp import *

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

def main():
        # start time
        startTime = time.time()

        # get data
        dataset = '/home/markg/Documents/TCD/ML/ML1819--task-107--team-11/cleanData.csv'
        data = pd.read_csv(dataset, encoding='latin-1')
        data.drop(columns=['Unnamed: 0'], inplace = True)
        # data.drop(columns = ['tweet_count', 'month', 'fav_number',
        # 'month', 'totalLettersName', 'link_hue', 'link_vue', 'link_sat', 'sidebar_sat',
        # 'sidebar_vue'], inplace=True)

        # create independent & dependent variables
        X = data.drop('gender_catg', axis=1)
        Y = data['gender_catg']


        # split into 90% training, 10% testing
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10)

        # train model (could change kernel here)
        svm = SVC(C=1, gamma=0.3, kernel='rbf')
        svm.fit(X_train, y_train)

        # make predictions and print metrics
        y_pred = svm.predict(X_test)
        print(classification_report(y_test,y_pred))
        print(confusion_matrix(y_test,y_pred))

        # recursive feature selection without cross validation
        # rfe = RFE(svm, 3)
        # fit = rfe.fit(X, Y)
        # print('Num Features:',fit.n_features_to_select)
        # print("Selected Features:",fit.support_)
        # print ("Feature ranking:", fit.ranking_)
        # data.info()

        # # # recursive feature selection using cross validation
        # rfecv = RFECV(estimator=svm, step=1, cv=StratifiedKFold(2),
        #               scoring='accuracy')
        # rfecv.fit(X, Y)
        # print("Optimal number of features : %d" % rfecv.n_features_)
        # print("Feature ranking: ", rfecv.ranking_)
        #
        # # plot bar chart of feature ranking
        # features = list(X)
        # ranking = rfecv.ranking_
        # plt.bar(features, ranking, align='center', alpha=0.5)
        # plt.savefig('featureRankingSVM.png')
        # plt.show()
        #
        # # Plot number of features VS. cross-validation scores
        # plt.figure()
        # plt.xlabel("Number of features selected")
        # plt.title("SVM")
        # plt.ylabel("Cross validation score")
        # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        # plt.savefig('featuresSelectionSVM.png')
        # plt.show()

        # cross validation to choose c and gamma
        # C_s, gamma_s = np.meshgrid(np.logspace(-2, 0.3, 5), np.logspace(-2, 0.3, 5))
        # scores = list()
        # i=0; j=0
        # for C, gamma in zip(C_s.ravel(),gamma_s.ravel()):
        #     svm.C = C
        #     svm.gamma = gamma
        #     this_scores = cross_val_score(svm, X, Y, cv=3)
        #     scores.append(np.mean(this_scores))
        # scores=np.array(scores)
        # scores=scores.reshape(C_s.shape)
        # fig2, ax2 = plt.subplots(figsize=(12,8))
        # c=ax2.contourf(C_s,gamma_s,scores)
        # ax2.set_xlabel('C')
        # ax2.set_ylabel('gamma')
        # fig2.colorbar(c)
        # fig2.savefig('crossvalParameterSelection2.png')

        # end time
        endTIme = time.time()
        totalTime = endTIme - startTime
        print("Time taken:", totalTime)

if __name__ == '__main__':
  main()
