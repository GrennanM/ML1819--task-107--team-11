# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 02:23:23 2018

@author: Geet
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:20:40 2018

@author: Geet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from IPython.display import display
#from collections import Counter
#from nltk.corpus import stopwords
#from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

def main():
  # load the data
  data = pd.read_csv('C:/Users/Geet/Downloads/gender-classifier-DFE-791531.csv', encoding='latin-1')
  
  # data cleaning
  #data.info()
  
  #print(data['gender'].head(3))
  #print(data['gender'].value_counts())
  drop_items_idx = data[data['gender'] == 'unknown'].index
  data.drop (index = drop_items_idx, inplace = True)
  drop_items_idx = data[data['gender'] == 'brand'].index
  data.drop (index = drop_items_idx, inplace = True)
  #print(data['gender'].value_counts())
  
  #print(data['_unit_id'].head(3))
  data.drop (columns = ['_unit_id'], inplace = True)
  
  #print(data['_golden'].head(3))
  #print(data['_golden'].value_counts())
  data.drop (columns = ['_golden'], inplace = True)
  
  #print(data['_unit_state'].head(3))
  #print(data['_unit_state'].value_counts())
  data.drop (columns = ['_unit_state'], inplace = True)
  
  #print(data['_trusted_judgments'].head(3))
  #print(data['_trusted_judgments'].value_counts())
  data.drop (columns = ['_trusted_judgments'], inplace = True)
  
  #print(data['profile_yn'].head(3))
  #print(data['profile_yn'].value_counts())
  drop_items_idx = data[data['profile_yn'] == 'no'].index
  data.drop (index = drop_items_idx, inplace = True)
  data.drop (columns = ['profile_yn'], inplace = True)
  
  #print(data['gender:confidence'].head(3))
  #print(data['gender:confidence'].value_counts())
  drop_items_idx = data[data['gender:confidence'] < 1].index
  data.drop (index = drop_items_idx, inplace = True)
  data.drop (columns = ['gender:confidence'], inplace = True)
  
  #print(data['profile_yn:confidence'].value_counts())
  #drop_items_idx = data[data['profile_yn:confidence'] < 1].index
  #data.drop (index = drop_items_idx, inplace = True)
  data.drop (columns = ['profile_yn:confidence'], inplace = True)
  
  data.drop (columns = ['gender_gold','profile_yn_gold','profileimage','tweet_coord','tweet_id'],inplace = True)
  #data.info()
  
  #reformating data
  #data['fav_number']=pd.factorize(data['fav_number'])[0]
  #data['retweet_count']=pd.factorize(data['retweet_count'])[0]
  #data['tweet_count']=pd.factorize(data['tweet_count'])[0]
  
  data._last_judgment_at.replace(np.NaN, 0, inplace=True)
  data['_last_judgment_at'] = pd.to_datetime(data['_last_judgment_at'])
  data['date'] = pd.DatetimeIndex(data['_last_judgment_at']).day
  data.drop (columns = ['_last_judgment_at'], inplace = True)
  
  data['created'] = pd.to_datetime(data['created'])
  data['year'] = pd.DatetimeIndex(data['created']).year
  data['month'] = pd.DatetimeIndex(data['created']).month
  data.drop (columns = ['created'], inplace = True)
  #print(data['year'].value_counts())
  
  #data['tweet_created'] = pd.to_datetime(data['tweet_created'])
  #data['tweet_year'] = pd.DatetimeIndex(data['tweet_created']).year
  #data['tweet_month'] = pd.DatetimeIndex(data['tweet_created']).month
  #data['tweet_date'] = pd.DatetimeIndex(data['tweet_created']).day
  #print(data['tweet_date'])
  #data['tweet_date'] = pd.factorize(data['tweet_date'])[0]
  data.drop (columns = ['tweet_created'], inplace = True)
  #print(data['tweet_year'].value_counts())
  
  data['totalWords']=data['text'].str.split(' ').str.len()
  data['totalLetters']=data['text'].str.len()
  #print(data['totalWords'].head(5))
  data.drop (columns = ['text'], inplace = True)
  
  data['totalWordsDesc']=data['description'].str.split(' ').str.len()
  data.totalWordsDesc.replace(np.NaN, 0, inplace=True)
  #pd.factorize(data['totalWordsDesc'])[0]
  data['totalLetterssDesc']=data['description'].str.len()
  data.totalLetterssDesc.replace(np.NaN, 0, inplace=True)
  #pd.factorize(data['totalLetterssDesc'])[0]
  #print(data['totalWordsDesc'].head(5))
  data.drop (columns = ['description'], inplace = True)
  
  data['totalLettersName']=data['name'].str.len()
  data.drop (columns = ['name'], inplace = True)
  
  #data['link_color_string']=data['link_color_short'].apply(str)
  #print(data['link_color_string'].head(10))
  #data['link_color_decimal']=data['link_color_string'].apply(int,base=16)
  #print(data['link_color_decimal'].head(5))
  
  #data['link_color_catg1']=data['link_color'].astype('category')
  #print(data['link_color_catg1'].head(10))
  
  #print(data['link_color'].value_counts())
  data['link_color_catg']=pd.factorize(data['link_color'])[0]
  #print(data['link_color_catg'].head(10))
  data.drop (columns = ['link_color'], inplace = True)
  
  data['sidebar_color_catg']=pd.factorize(data['sidebar_color'])[0]
  data.drop (columns = ['sidebar_color'], inplace = True)
  
  #data.tweet_location.replace(np.NaN, 0, inplace=True)
  #data['tweet_location']=pd.factorize(data['tweet_location'])[0]
  data.tweet_location.where(data.tweet_location.isnull(), 1, inplace=True)
  data.tweet_location.replace(np.NaN, 0, inplace=True)
  #print(data['tweet_location'].head(10))
  
  #data.user_timezone.replace(np.NaN, 0, inplace=True)
  #data['user_timezone']=pd.factorize(data['user_timezone'])[0]
  data.user_timezone.where(data.user_timezone.isnull(), 1, inplace=True)
  data.user_timezone.replace(np.NaN, 0, inplace=True)
  
  data['gender_catg']=pd.factorize(data['gender'])[0]
  data.drop (columns = ['gender'], inplace = True)
  data.info()
  
  #number of columns
  colmn = (len(data.columns)) - 1
  array = data.values
  X = array[:,0:colmn]
  Y = array[:,colmn]
  
  # feature extraction
  model = LogisticRegression()
  rfe = RFE(model, 3)
  fit = rfe.fit(X, Y)
  print('Num Features:',fit.n_features_to_select)
  print("Selected Features:",fit.support_)
  #print("Feature Ranking:",fit.ranking_)
  
  logit_model=sm.Logit(Y,X)
  result=logit_model.fit()
  #print(result.summary2())
  
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
  logreg = LogisticRegression()
  logreg.fit(X_train, y_train)
  
  y_pred = logreg.predict(X_test)
  print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
  
  logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
  fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
  plt.figure()
  plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.savefig('Log_ROC')
  plt.show()
  
if __name__ == '__main__':
  main()