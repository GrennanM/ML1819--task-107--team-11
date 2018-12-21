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
import re
from textblob import TextBlob
import numpy as np

def clean_tweet(tweet):
  return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def textAnalysis(tweet):
  analysis = TextBlob(clean_tweet(tweet))
  if analysis.sentiment.polarity > 0:
    return 1
  elif analysis.sentiment.polarity == 0:
    return 0
  else:
    return -1

def main():
  # load the data
  dataset = '/home/markg/Documents/TCD/ML/ML1819--task-107--team-11/dataset/gender-classifier-DFE-791531.csv'
  data = pd.read_csv(dataset, encoding='latin-1')
  #data.description.replace(np.NaN, 'NULL', inplace=True)
  text_sent=[]
  for tweet in data['text']:
    text_sent.append(textAnalysis(tweet))
  data['text_sent']=text_sent

  print(data['text_sent'].head(20))


if __name__ == '__main__':
  main()
