#!/usr/bin/env python3

# set up code for twitter dataset
# will return a file cleanData.csv

import numpy as np
import pandas as pd
from sklearn import preprocessing
from textAnalysis_ex import *

def main():
        # load the dataset
        dataset = '/home/markg/Documents/TCD/ML/ML1819--task-107--team-11/dataset/gender-classifier-DFE-791531.csv'
        data = pd.read_csv(dataset, encoding='latin-1')

        # reformat date column
        data['created'] = pd.to_datetime(data['created'])

        # create new columns for year and month % remove original column
        data['year'] = pd.DatetimeIndex(data['created']).year
        data['month'] = pd.DatetimeIndex(data['created']).month
        data = data.drop(['created'], axis=1)

        # drop entries where gender=brand and gender=unknown
        drop_items_idx = data[data['gender'] == 'unknown'].index
        data.drop (index = drop_items_idx, inplace = True)
        drop_items_idx = data[data['gender'] == 'brand'].index
        data.drop (index = drop_items_idx, inplace = True)

        # drop unnecessary columns
        notNeededCols = ['_unit_id', '_golden', '_unit_state', '_trusted_judgments',
         'profileimage','tweet_coord','tweet_id', '_last_judgment_at', 'tweet_created',
         'gender_gold','profile_yn_gold', 'description']
        data.drop(columns=notNeededCols, inplace=True)

        # drop entries where profile doesn't exist and column profile_yn
        drop_items_idx = data[data['profile_yn'] == 'no'].index
        data.drop (index = drop_items_idx, inplace = True)
        data.drop (columns = ['profile_yn'], inplace = True)

        # drop entries where gender conf < 1  % col gender_conf
        drop_items_idx = data[data['gender:confidence'] < 1].index
        data.drop (index = drop_items_idx, inplace = True)
        data.drop (columns = ['gender:confidence'], inplace = True)
        data.drop (columns = ['profile_yn:confidence'], inplace = True)

        # count the length of letters in name and create column
        data['totalLettersName']=data['name'].str.len()
        data.drop (columns = ['name'], inplace = True)

        # change tweet location to 0 if present, 1 if empty
        data.tweet_location.where(data.tweet_location.isnull(), 1, inplace=True)
        data.tweet_location.replace(np.NaN, 0, inplace=True)

        # change timezone to 0 if present, 1 if empty
        data.user_timezone.where(data.user_timezone.isnull(), 1, inplace=True)
        data.user_timezone.replace(np.NaN, 0, inplace=True)

        # change gender to 0=Male, 1=Female
        data['gender_catg']=pd.factorize(data['gender'])[0]
        data.drop (columns = ['gender'], inplace = True)

        # categorize colours
        data['link_color_catg']=pd.factorize(data['link_color'])[0]
        data['sidebar_color_catg']=pd.factorize(data['sidebar_color'])[0]
        data.drop (columns = ['sidebar_color'], inplace = True)
        data.drop (columns = ['link_color'], inplace = True)

        # analyize text for sentiment & drop text column
        text_sent=[]
        for tweet in data['text']:
          text_sent.append(textAnalysis(tweet))
        data['text_sent']=text_sent
        data.drop(columns = ['text'], inplace=True)

        # standardize numeric variables (could also consider using robust scaler here)
        numericVariables = ['fav_number', 'tweet_count','retweet_count', 'totalLettersName',
         'year', 'month', 'link_color_catg', 'sidebar_color_catg']
        scaler = preprocessing.StandardScaler()
        data[numericVariables] = scaler.fit_transform(data[numericVariables])

        data.to_csv('cleanData.csv')

if __name__ == '__main__':
  main()
