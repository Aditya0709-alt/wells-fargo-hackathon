#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def consecutive_training_samples(ts):
    """
    Hypothesis: assignment is a dataframe with DATE as ordered index
    Goal is to find >= 3-week consecutive training samples, especially in the prediction zone
    """
    
    # Init
    couples = []
    start = ts.index.date[0]
    end = ts.index.date[0]
    current = [start]
    
    for i, day in enumerate(ts.index.date):
        if ((day not in current) and
            (day - timedelta(days=1)) in current):
            current.append(day)
            end = day
        elif ((day not in current) and
              (day - timedelta(days=1)) not in current):
            delta = end - start
            if delta.days >= 21:
                couples.append([np.datetime64(start), np.datetime64(end)])
            start = day
            end = day
            current = [start]
    #couples = couples[1:]
    return pd.DataFrame(couples)
   
if '__name__' == '__main__':
    submission = pd.read_csv('./submission.txt',
                             sep='\t',
                             parse_dates=['DATE'])
    assignment_names = submission['ASS_ASSIGNMENT'].unique()
    for assignment_name in assignment_names:
        ts = pd.read_csv('./data/CSPL_RECEIVED_CALLS_series/' + assignment_name + '.csv',
                         sep=";",
                         parse_dates=['DATE'],
                         index_col=['DATE'])
        couples = consecutive_training_samples(ts)
        couples.to_csv('./data/consecutive_training_days/' + assignment_name + '_list_com.csv')