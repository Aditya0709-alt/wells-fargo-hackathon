import numpy as np
import pandas as pd

n_burn_in = 672

holidays = np.array([
    '2011-04-25',
    '2011-05-01', #WE
    '2011-05-08', #WZ
    '2011-06-02',
    '2011-06-13',
    '2011-07-14',
    '2011-08-15',
    '2011-11-01',
    '2011-11-11',
    '2011-12-25', #WE
    '2011-12-31', #WE
    '2012-01-01', #WE
    '2012-04-09',
    '2012-05-01',
    '2012-05-08',
    '2012-05-17',
    '2012-05-28',
    '2012-07-14', # WE
    '2012-08-15',
    '2012-11-01',
    '2012-11-11', #WE
    '2012-12-25',
    '2012-12-31',
    '2013-01-01',
    '2013-04-01',
    '2013-05-01',
    '2013-05-08',
    '2013-05-09',
    '2013-05-20', #WE
    '2013-07-14', #WE
    '2013-08-15',
    '2013-11-01',
    '2013-11-11',
    '2013-12-25',
    '2013-12-31'
])


def smooth_holidays(series):
    res = pd.DataFrame()
    res["Value"] = series.values
    res.index = series.index
    res['rolled'] = np.roll(res['Value'], 336)
    res['max'] = np.maximum(res['Value'], res['rolled'])
    for holiday in holidays:
        if holiday in res.index:
            res.loc[holiday, 'Value'] = res[holiday]['max']
    return pd.DataFrame(res['Value'])


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, X_ds):
        temp = smooth_holidays(X_ds).values.reshape((-1,1)).copy()
        
        timeseries = temp.copy()
        for i in range(1, 48):
            timeseries = np.hstack((np.roll(temp, i), timeseries))
        for i in range(1, 48):
            timeseries = np.hstack((np.roll(temp, 336 + i), timeseries))
        for i in range(1, 48):
            timeseries = np.hstack((np.maximum(np.roll(temp, 336 + i), np.roll(temp, i)).reshape((-1, 1)), timeseries))
        
        timeseries = timeseries[n_burn_in::, :]
        return timeseries