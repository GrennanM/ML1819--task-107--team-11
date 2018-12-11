import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from collections import OrderedDict

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
        y = data['gender_catg']

        # split into 90% training, 10% testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)


        ############## BASELINE #################################

        # # build a Random Forest Model
        # baseline_model = RandomForestClassifier(n_estimators=1500, max_depth=None,
        #  min_samples_split=2, random_state=0)
        #
        # # fit the training model
        # baseline_model.fit(X_train, y_train)
        #
        # # predict validation data set
        # y_pred = baseline_model.predict(X_test)
        # print(classification_report(y_test,y_pred))
        # print(confusion_matrix(y_test,y_pred))

        #################### END BASELINE ##########################

        ######### Hyper-Parameter Tuning #####################################

        ###### Randomized Search #############
        #
        # # Number of trees in random forest
        # n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 2000, num = 10)]
        # # Number of features to consider at every split
        # max_features = ['sqrt', 'log2', None]
        # # Maximum number of levels in tree
        # max_depth = [int(x) for x in np.linspace(500, 1500, num = 20)]
        # max_depth.append(None)
        # # Minimum number of samples required to split a node
        # min_samples_split = [2, 5, 10]
        # # Minimum number of samples required at each leaf node
        # min_samples_leaf = [1, 2, 4]
        # # Method of selecting samples for training each tree
        # bootstrap = [True, False]
        #
        # # Create the random grid
        # random_grid = {'n_estimators': n_estimators,
        #                'max_features': max_features,
        #                'max_depth': max_depth,
        #                'min_samples_split': min_samples_split,
        #                'min_samples_leaf': min_samples_leaf,
        #                'bootstrap': bootstrap}
        #
        # # First create the base model to tune
        # rf = RandomForestClassifier()
        # # Random search of parameters, using 3 fold cross validation,
        # # search across 100 different combinations, and use all available cores
        # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
        #  n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        #
        # # Fit the random search model
        # rf_random.fit(X_train, y_train)
        #
        # print ("Best parameters:", rf_random.best_params_)

        ######### END Randomized Search ##################

        # # build a Random Forest Model using Randomized Search results
        # best_randomized_model = RandomForestClassifier(n_estimators=1333,
        #                 min_samples_split=2, min_samples_leaf=4,
        #                 max_features='sqrt', max_depth=1236,
        #                 bootstrap=True, random_state=0)
        #
        # # fit the training model
        # best_randomized_model.fit(X_train, y_train)
        #
        # # # cross validation to return metrics
        # # scores = cross_validate(best_randomized_model, X_test, y_test, cv=5,
        # #                         scoring=('recall', 'precision', 'f1'))
        # # print ("Recall: ", scores['test_recall'].mean())
        # # print ("Precision: ", scores['test_precision'].mean())
        # # print ("F1: ", scores['test_f1'].mean())
        #
        # # # predict validation data set
        # # y_pred = best_randomized_model.predict(X_test)
        # # print(classification_report(y_test,y_pred))
        ############### END RANDOMISED SEARCH MODEL ######################

        ###############  Grid Search ##########

        # # grid returned from random search
        # grid_from_random_search = {{'n_estimators': 1333, 'min_samples_split': 2,
        #  'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 1236,
        #  'bootstrap': True}}
        #
        # param_grid = {'n_estimators': [1000],
        # 'min_samples_split': [2, 3],
        # 'min_samples_leaf': [3, 4, 5],
        # 'max_features': ['sqrt'],
        # 'max_depth': [1200, 1300],
        # 'bootstrap': [True]}
        #
        # # model
        # rf = RandomForestClassifier()
        #
        # # Grid Search of parameters, using 3 fold cross validation
        # grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
        #                       cv = 3, n_jobs = -1, verbose = 2)
        #
        # # Fit the random search model
        # grid_search.fit(X_train, y_train)
        # print (grid_search.best_params_)
        ############### End Grid Search ###############

        # build model with results of grid search
        best_grid_model = RandomForestClassifier(n_estimators=2000,
                        min_samples_split=3, min_samples_leaf=3,
                        max_features='sqrt', max_depth=1200,
                        bootstrap=True, random_state=0)

        # fit the training model
        best_grid_model.fit(X_train, y_train)

        # predict validation data set
        y_pred = best_grid_model.predict(X_test)
        print(classification_report(y_test,y_pred))

        ################# End grid model ###################


        # end time
        endTIme = time.time()
        totalTime = endTIme - startTime
        print("Time taken:", totalTime)

if __name__ == '__main__':
  main()
