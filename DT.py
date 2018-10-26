#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 00:54:03 2018

@author: vishalrana
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from pmlb import fetch_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

#Dropping column
    data3 = pd.read_csv('cleanData.csv', na_values = '?',encoding='latin-1')
    data3= data3.loc[:, ~data3.columns.str.contains('^Unnamed')]
    data3 = data3.drop(['tweet_location','year','sidebar_hue','link_sat'
                        ,'totalLettersName','sidebar_vue','sidebar_sat','sidebar_hue'], axis=1)
    data3 = data3.drop(['link_vue','retweet_count'],axis=1)
    data3 = data3[['tweet_location','retweet_count','link_hue','sidebar_vue','gender_catg']]
    X1 = data3.drop('gender_catg', axis=1)
    y1 = data3['gender_catg']
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.10)

#Decission Tree
    dt = DecisionTreeRegressor(criterion="mse")
    dt_fit = dt.fit(X_train1, y_train1)
    dt_scores = cross_val_score(dt_fit, X_train1, y_train1, cv = 5)
    print("mean cross validation score: {}".format(np.mean(dt_scores)))
    print("score without cv: {}".format(dt_fit.score(X_train1, y_train1)))
    print(r2_score(y_test, dt_fit.predict(X_test1)))
    print(dt_fit.score(X_test1, y_test1))
    scoring = make_scorer(r2_score)
    g_cv = GridSearchCV(DecisionTreeRegressor(),
              param_grid={'min_samples_split': range(2, 10)},
              scoring=scoring, cv=5, refit=True)

    g_cv.fit(X_train1, y_train1)
    y_pred1 = g_cv.predict(X_test1)
    leng = len(y_pred1)
    for x in range(leng):
        if y_pred1[x] > 0.5: 
            y_pred1[x] = 1
        else:
            y_pred1[x] = 0  
#To print the report  
    print(classification_report(y_test1,y_pred1))
   
#Visualizing decission tree    
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import pydotplus
    dot_data = StringIO()
    export_graphviz(dt_fit, out_file=dot_data, filled = True, rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    (graph,) = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())

#To print the report  
    print(classification_report(y_test1,y_pred1))
    
    
    
#Choosing best features
    %matplotlib inline

    sns.set_style("darkgrid")


    class PipelineRFE(Pipeline):

        def fit(self, X, y=None, **fit_params):
            super(PipelineRFE, self).fit(X, y, **fit_params)
            self.feature_importances_ = self.steps[-1][-1].feature_importances_
            return self

    pipe = PipelineRFE(
    [
        ('std_scaler', preprocessing.StandardScaler()),
        ("ET", ExtraTreesRegressor(random_state=42, n_estimators=250))
    ]
    )

    # Sets RNG seed to reproduce results 
    _ = StratifiedKFold(random_state=42)

    feature_selector_cv = feature_selection.RFECV(pipe, cv=10, step=1, scoring="neg_mean_squared_error")
    feature_selector_cv.fit(X_train1, y_train1)
    print("Optimal number of features : %d" % feature_selector_cv.n_features_)
    print("Feature ranking: ", feature_selector_cv.ranking_)
    selected_features = feature_names[feature_selector_cv.support_].tolist()
    selected_features
    
    
    
#Feature Correlation    
fs = FeatureSelector(data = data3)
##fs = FeatureSelector(data = X_train1, labels = X_train1)
fs.identify_missing(missing_threshold = 0.01)
fs.missing_stats.head()
fs.head()
fs.missing_stats.head()
fs.identify_collinear(correlation_threshold = 0.10)
collinear_features = fs.ops['collinear']
fs.record_collinear.head()
fs.plot_collinear(plot_all = True)