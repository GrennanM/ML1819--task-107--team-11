import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, RFE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from collections import OrderedDict

def main():
        # start time
        startTime = time.time()

        # get data
        dataset = '/home/markg/Documents/TCD/ML/ML1819--task-107--team-11/cleanData2.csv'
        data = pd.read_csv(dataset, encoding='latin-1')
        data.drop(columns=['Unnamed: 0'], inplace = True)

        # store independent variables for later
        names = ['fav_number', 'retweet_count','tweet_count',
        'tweet_location', 'year', 'month', 'totalLettersName', 'link_hue',
         'link_sat', 'link_vue', 'sidebar_hue', 'sidebar_sat', 'sidebar_vue']

        # create a dictionary of indices and independent variables
        nameDict = {}
        for i in range(len(names)):
            nameDict[i] = names[i]

        # create independent & dependent variables
        X = data.drop('gender_catg', axis=1)
        y = data['gender_catg']

        # split into 90% training, 10% testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

        # ############## BASELINE Decision Tree ################################
        #
        # # build decision tree
        # dt = DecisionTreeClassifier()
        # dt_fit = dt.fit(X_train, y_train)
        #
        # # predict test data set
        # y_pred = dt.predict(X_test)
        # print(classification_report(y_test,y_pred))
        #
        # ############## BASELINE Random Forest #################################

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
        #
        # # build model with results of grid search
        # best_grid_model = RandomForestClassifier(n_estimators=2000,
        #                 min_samples_split=3, min_samples_leaf=3,
        #                 max_features='sqrt', max_depth=1200,
        #                 bootstrap=True, random_state=0)
        #
        # # fit the training model
        # best_grid_model.fit(X_train, y_train)
        #
        # # predict validation data set
        # y_pred = best_grid_model.predict(X_test)
        # print(classification_report(y_test,y_pred))

        ################# End grid model ###################

    # ################ print OOB vs search criteria #############################
    #     RANDOM_STATE = 123
    #
    #     # NOTE: Setting the `warm_start` construction parameter to `True` disables
    #     # support for parallelized ensembles but is necessary for tracking the OOB
    #     # error trajectory during training.
    #     ensemble_clfs = [
    #         ("RandomForestClassifier, max_features=2",
    #             RandomForestClassifier(n_estimators=100,
    #                                    warm_start=True, max_features=2,
    #                                    oob_score=True,
    #                                    random_state=RANDOM_STATE)),
    #         ("RandomForestClassifier, max_features=3",
    #             RandomForestClassifier(n_estimators=100,
    #                                    warm_start=True, max_features=3,
    #                                    oob_score=True,
    #                                    random_state=RANDOM_STATE)),
    #         ("RandomForestClassifier, max_features=4",
    #             RandomForestClassifier(n_estimators=100,
    #                                    warm_start=True, max_features=4,
    #                                    oob_score=True,
    #                                    random_state=RANDOM_STATE))
    #     ]
    #
    #     # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    #     error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    #
    #     # Range of `n_estimators` values to explore.
    #     min_estimators = 10
    #     max_estimators = 1500 # number of trees to try
    #
    #     for label, clf in ensemble_clfs:
    #         for i in range(min_estimators, max_estimators + 1, 20):
    #             clf.set_params(n_estimators=i)
    #             clf.fit(X_train, y_train)
    #
    #             # Record the OOB error for each `n_estimators=i` setting.
    #             oob_error = 1 - clf.oob_score_
    #             error_rate[label].append((i, oob_error))
    #
    #     # Generate the "OOB error rate" vs. "n_estimators" plot.
    #     for label, clf_err in error_rate.items():
    #         xs, ys = zip(*clf_err)
    #         plt.plot(xs, ys, label=label)
    #
    #     axes = plt.gca()
    #     axes.set_ylim([0,0.4]) # sets y_axis limit to between 0 and 1
    #     plt.xlim(min_estimators, max_estimators)
    #     plt.xlabel("n_estimators")
    #     plt.ylabel("OOB error rate")
    #     plt.legend(loc="upper right")
    #     plt.show()
    #     ######################## END PLOT ###########################

        # ################### Feature Importance Plot ##################
        # # Build a forest and compute the feature importances
        # forest = ExtraTreesClassifier(n_estimators=200, min_samples_split=3,
        #                             min_samples_leaf=3, max_features=4,
        #                             max_depth=1200, bootstrap=True,
        #                               random_state=0)
        #
        # forest.fit(X_train, y_train)
        # importances = forest.feature_importances_
        # std = np.std([tree.feature_importances_ for tree in forest.estimators_],
        #              axis=0)
        # indices = np.argsort(importances)[::-1]
        #
        # # create a list of features in order of importance
        # important_features = [nameDict[i] for i in indices]
        #
        # # Plot the feature importances of the forest
        # plt.figure()
        # plt.title("Feature importance Random Forest")
        # plt.bar(range(X.shape[1]), importances[indices],
        #        color="r", yerr=std[indices], align="center")
        # plt.xticks(range(X.shape[1]), important_features, rotation=60)
        # plt.xlim([-1, X.shape[1]])
        # plt.ylabel("Relative Importance")
        # plt.show()
        # #################### End Feature Importance Plot ######################

        # #################### Recursive Feature Selection ######################
        #
        # # svm recursive feature selection
        # svm = SVC(C=1, gamma=0.3, kernel='linear')
        # svm_rfe = RFECV(estimator=svm, step=1, cv=StratifiedKFold(3),
        #                       scoring='accuracy')
        # svm_rfe.fit(X_train, y_train)
        #
        # # Random Forest recursive feature selection
        # best_grid_model = RandomForestClassifier(n_estimators=100,
        #                 min_samples_split=3, min_samples_leaf=3,
        #                 max_features='sqrt', max_depth=1200,
        #                 bootstrap=True, random_state=0)
        # rf_rfe = RFECV(estimator=best_grid_model, step=1, cv=StratifiedKFold(3),
        #               scoring='accuracy')
        # rf_rfe.fit(X_train, y_train)
        #
        # # Logistic Recursive Feature Selection
        # logistic = LogisticRegression()
        # log_rfe = RFECV(estimator=logistic, step=1, cv=StratifiedKFold(3),
        #               scoring='accuracy')
        # log_rfe.fit(X_train, y_train)
        #
        # # Decision tree Feature Selection
        # dt = DecisionTreeClassifier()
        # dt_rfe = RFECV(estimator=dt, step=1, cv=StratifiedKFold(3),
        #               scoring='accuracy')
        # dt_rfe.fit(X_train, y_train)
        #
        # # Plot number of features VS. Accuracy cross-validation scores
        # plt.figure()
        # plt.xlabel("Number of Features Selected")
        # plt.title("Recursive Feature Selection")
        # plt.ylabel("Accuracy 5-Fold Cross validation Score")
        # plt.plot(range(1, len(rf_rfe.grid_scores_) + 1), rf_rfe.grid_scores_,
        # label="Random Forest")
        # plt.plot(range(1, len(svm_rfe.grid_scores_) + 1), svm_rfe.grid_scores_,
        # label="SVM")
        # plt.plot(range(1, len(log_rfe.grid_scores_) + 1), log_rfe.grid_scores_,
        # label="Logistic Regression")
        # plt.plot(range(1, len(dt_rfe.grid_scores_) + 1), dt_rfe.grid_scores_,
        # label="Baseline Decision Tree")
        # plt.axhline(y=0.5, label="Random Classifier", color="#D3D3D3",
        # linestyle="dashed")
        # plt.legend(loc=4)
        # plt.show()
        #
        # ################ End Recursive Feature ##############################

        # #################### Test Model with 8 best features #################
        #
        # # create independent & dependent variables without worst features
        # data.drop(columns = ['retweet_count', 'sidebar_sat', 'tweet_location',
        # 'sidebar_hue', 'sidebar_vue'], inplace = True)
        # X = data.drop('gender_catg', axis=1)
        # y = data['gender_catg']
        #
        # # split into 90% training, 10% testing
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)
        #
        # # build model
        # rf_8features_model = RandomForestClassifier(n_estimators=600,
        #                 min_samples_split=3, min_samples_leaf=3,
        #                 max_features='sqrt', max_depth=1200,
        #                 bootstrap=True, random_state=0)
        #
        # # fit the model
        # rf_8features_model.fit(X_train, y_train)
        #
        # # predict validation data set
        # y_pred = rf_8features_model.predict(X_test)
        # print(classification_report(y_test,y_pred))
        #
        # ################### End Test Model with 8 features ##################

        # #################### ROC-AUC Curve #############################
        #
        # # fit svm
        # svm = SVC(C=1, gamma=0.3, kernel='rbf', probability=True)
        # svm.fit(X_train, y_train)
        #
        # # fit random forest
        # rf_model = RandomForestClassifier(n_estimators=200,
        #                 min_samples_split=3, min_samples_leaf=3,
        #                 max_features='sqrt', max_depth=1200,
        #                 bootstrap=True, random_state=0)
        # rf_model.fit(X_train, y_train)
        #
        # # fit logistic model
        # logreg = LogisticRegression()
        # logreg.fit(X_train, y_train)
        #
        # # fit baseline decision tree
        # dt = DecisionTreeClassifier()
        # dt.fit(X_train, y_train)
        #
        # plt.figure()
        # # ROC for random forest
        # rf_roc_auc = roc_auc_score(y_test, rf_model.predict(X_test))
        # fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
        # plt.plot(fpr, tpr, label='Random Forest (AUC = %0.2f)' % rf_roc_auc)
        #
        # # ROC for SVM
        # svm_roc_auc = roc_auc_score(y_test, svm.predict(X_test))
        # fpr, tpr, thresholds = roc_curve(y_test, svm.predict_proba(X_test)[:,1])
        # plt.plot(fpr, tpr, label='SVM (AUC = %0.2f)' % svm_roc_auc)
        #
        # # ROC for Logistic Regression
        # logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
        # fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
        # plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.2f)' % logit_roc_auc)
        #
        # # ROC for DT
        # dt_roc_auc = roc_auc_score(y_test, dt.predict(X_test))
        # fpr, tpr, thresholds = roc_curve(y_test, dt.predict_proba(X_test)[:,1])
        # plt.plot(fpr, tpr, label='Baseline Decision Tree (AUC = %0.2f)' % dt_roc_auc)
        #
        # plt.plot([0, 1], [0, 1],'r--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        # plt.legend(loc="lower right")
        # plt.show()
        #
        # ################## END ROC-AUC Curve ###################################




        # end time
        endTIme = time.time()
        totalTime = endTIme - startTime
        print("Time taken:", totalTime)

if __name__ == '__main__':
  main()
