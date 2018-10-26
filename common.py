# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:12:15 2018

@author: Geet
"""

import pandas as pd
import numpy as np
from numpy import mean
from numpy import std

def stndrd_Devtn(column,num):
  data_mean, data_std = mean(column), std(column)
  cut_off = data_std * num
  lower, upper = data_mean - cut_off, data_mean + cut_off
  return lower,upper

def colur_hex_to_huv(column):
  H, S, V = [], [], []
  linkColorList = column
  for color in linkColorList:
    try:
      color = list(int(color[i:i+2], 16) for i in (0, 2 ,4))
    except ValueError:
      color = [0, 0, 0]
    # must divide by 255 as co-ordinates are in range 0 to 1
    hsv = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
    # rescale to hsv
    x = round(hsv[0]*360, 1)
    y = round(hsv[1]*100, 1)
    z = round(hsv[2]*100, 1)
    H.append(x)
    S.append(y)
    V.append(z)
  return H,S,V

def gender_to_numeric(a):
  if a == 'male':
    return 0
  if a == 'female':
    return 1

def main():
  # load the data
  data = pd.read_csv('C:/Users/Geet/Downloads/gender-classifier-DFE-791531.csv', encoding='latin-1')
  data.info()
  
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
   'gender_gold','profile_yn_gold', 'description', 'user_timezone','text']
  data.drop(columns=notNeededCols, inplace=True)
  data= data.loc[:, ~data.columns.str.contains('^Unnamed')]

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
  
  # convert linkColor to hsv
  data['link_hue'],data['link_sat'],data['link_vue'] = colur_hex_to_huv(data['link_color'])
  data.drop(columns = ['link_color'], inplace = True)
  
  # convert sidebar to hsv
  data['sidebar_hue'],data['sidebar_sat'],data['sidebar_vue'] = colur_hex_to_huv(data['sidebar_color'])
  data.drop (columns = ['sidebar_color'], inplace = True)

  ####### OUTLIER CODE #######################
  l,u = stndrd_Devtn(data['fav_number'],3)
  drop_items_idx = data[(data['fav_number'] > u) | (data['fav_number'] < l)].index
  data.drop (index = drop_items_idx, inplace = True)
  
  l,u = stndrd_Devtn(data['retweet_count'],3)
  drop_items_idx = data[(data['retweet_count'] > u) | (data['retweet_count'] < l)].index
  data.drop (index = drop_items_idx, inplace = True)
  
  l,u = stndrd_Devtn(data['tweet_count'],3)
  drop_items_idx = data[(data['tweet_count'] > u) | (data['tweet_count'] < l)].index
  data.drop (index = drop_items_idx, inplace = True)
  
  # change gender to 0=Male, 1=Female
  data['gender_catg'] = data['gender'].apply(gender_to_numeric)
  data.drop (columns = ['gender'], inplace = True)
  data.info()
  
  # standardize numeric variables (could also consider using robust scaler here)
  numericVariables = ['fav_number', 'tweet_count','retweet_count', 'totalLettersName',
   'year', 'month', 'link_hue', 'link_vue', 'link_sat',
   'sidebar_hue', 'sidebar_sat', 'sidebar_vue']

  scaler = preprocessing.StandardScaler()
  data[numericVariables] = scaler.fit_transform(data[numericVariables])
  
  data.to_csv('cleanData.csv')
  data.info()
  print (data.head(5))


if __name__ == '__main__':
  main()
