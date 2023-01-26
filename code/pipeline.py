# coding=utf-8
"""
Goal of this script is to handle the data science pipeline for a given center.
Meaning it takes care of all cross-validation details so
that users can keep focused on models
"""
import numpy as np
import sys
import csv
import pandas as pd
from importlib import import_module

# n_burn_in: number of observations used to predict
n_burn_in = 672
# n_cv: number of splits for the cross validation
n_cv = 4

#dates for the spltit between train and test data
date_train = np.datetime64('2012-08-01 00:00:00.000')
date_train_y = np.datetime64('2012-08-08 00:00:00.000')
end_train = np.datetime64('2012-12-21 00:00:00.000')
end_train_y = np.datetime64('2012-12-28 00:00:00.000')


def get_cv(y_train_array):
    """
    Return indices for the cross validation
    """
    n = len(y_train_array)
    n_common_block = int(n / 2)
    n_validation = n - n_common_block
    block_size = int(n_validation / n_cv)
    print('length of common block: %s half_hours = %s weeks' %
          (n_common_block, n_common_block / 336))
    print('length of validation block: %s half_hours = %s weeks' %
          (n_validation, n_validation / 336))
    print('length of each cv block: %s half_hours = %s weeks' %
          (block_size, block_size / 336))
    for i in range(n_cv):
        train_is = np.arange(n_common_block + i * block_size)
        test_is = np.arange(n_common_block + i * block_size, n_common_block + n_cv * block_size)
        if i==n_cv-1:
            yield (train_is, test_is)


def score(y_true, y_pred):
    """
    compute the score between true labels and predicted labels
    """
    return np.mean(np.exp(0.1 * (y_true - y_pred)) - 0.1 * (y_true  - y_pred) - 1)


def read_data(assignment):
    """
    Reaf data and create dataframe X and the array y
    """
    X_ds = []
    y_array = []
    
    file_name = "./data/treated_data/" + assignment + ".csv"
    
    X_ds = pd.read_csv(file_name, sep=";", header=None, names=["DATE", "Value"])
    X_ds.index = X_ds["DATE"].values.astype(np.datetime64)
    X_ds = X_ds["Value"]
    y_array = X_ds.copy()
        
    return X_ds, y_array

def get_train_test_data(assignment):
    """
    Split data between train set and test set
    """
    X_ds, y_array = read_data(assignment)

    X_train_ds = X_ds[X_ds.index<date_train]
    y_train_array = y_array[X_ds.index<date_train_y].iloc[1008::]
    print('length of training array: %s half hours = %s weeks' %
          (len(y_train_array), len(y_train_array) / 336))
    
    X_test_ds = X_ds[X_ds.index>date_train]
    X_test_ds = X_test_ds[X_test_ds.index<end_train]
    y_test_array = y_array[X_ds.index>date_train]
    y_test_array = y_test_array[y_test_array.index<end_train_y].iloc[1008::]
    print('length of test array: %s half hours = %s weeks' %
          (len(y_test_array), len(y_test_array) / 336))
    return X_train_ds, y_train_array, X_test_ds, y_test_array


def get_compl_data(assignment, list_ranges, module_path, ts_feature_extractor_name="ts_feature_extractor"):
    """
    Get complementary data, i.e. data between weeks we have to predict
    """
    X_ds, y_array = read_data(assignment) 

    timedelta = np.datetime64('2013-01-28 00:00:00.000') - np.datetime64('2013-01-21 00:00:00.000')

    ts_feature_extractor = import_module(ts_feature_extractor_name, module_path)
    ts_fe = ts_feature_extractor.FeatureExtractor()

    X_comp = X_ds[X_ds.index>list_ranges[0][0]]
    X_comp = X_comp[X_comp.index<(list_ranges[0][1]-timedelta)]
    X_comp = ts_fe.transform(X_comp)

    y_comp = X_ds[X_ds.index>(list_ranges[0][0] + timedelta)]
    y_comp = y_comp[y_comp.index<list_ranges[0][1]]
    y_comp = y_comp.values[n_burn_in::]

    for a,b in list_ranges[1::]:
        X_temp = X_ds[X_ds.index>a] 
        X_temp = X_temp[X_temp.index<(b-timedelta)]

        X_comp = np.vstack((X_comp, ts_fe.transform(X_temp)))

        y_temp = X_ds[X_ds.index>(a + timedelta)]
        y_temp = y_temp[y_temp.index<b]

        y_comp = np.concatenate((y_comp, y_temp.values[n_burn_in::]))

    return X_comp, y_comp


def train_submission(module_path, X_ds, y_array, train_is, X_comp, y_comp, 
                        ts_feature_extractor_name="ts_feature_extractor",
                        regressor_name="regressor"):
    """
    train a model on the train data (complementary or not)
    """
    X_train_ds = X_ds[train_is]
    y_train_array = y_array[train_is].values[n_burn_in::]

    # Feature extraction
    ts_feature_extractor = import_module(ts_feature_extractor_name, module_path)
    ts_fe = ts_feature_extractor.FeatureExtractor()
    X_train_array = ts_fe.transform(X_train_ds)
    
    #Ajout du complément
    X_train = np.vstack((X_train_array, X_comp))
    y_train = np.concatenate((y_train_array, y_comp))

    # Regression
    regressor = import_module(regressor_name, module_path)
    reg = regressor.Regressor()
    reg.fit(X_train, y_train)
    return ts_fe, reg


def test_submission(trained_model, X_ds, test_is, X_comp=None):
    """
    compute the prediction for X_ds
    """

    X_test_ds = X_ds[test_is]
    ts_fe, reg = trained_model
    # Feature extraction
    X_test_array = ts_fe.transform(X_test_ds)
    if not(X_comp is None):
        X_test_array = np.vstack((X_test_array, X_comp))
    # Regression
    y_pred_array = reg.predict(X_test_array)
    return y_pred_array


def final_predict(assignment, list_comp, ts_feature_extractor_name="ts_feature_extractor",
                  regressor_name="regressor",
                  adjustment=1.):
    """
    Train model on all data and predict everything we can 
    """
    X_ds, y_array = read_data(assignment)

    X_train_ds = X_ds[X_ds.index<end_train]
    y_train_array = y_array[X_ds.index<end_train_y].iloc[1008::]

    X_comp, y_comp = get_compl_data(assignment, list_comp, './', ts_feature_extractor_name)
    trained_model = train_submission('./', X_train_ds, y_train_array, range(len(y_train_array)), X_comp, y_comp,
                                     ts_feature_extractor_name, regressor_name)
    
    
    
    y_train_pred_array = test_submission(trained_model, X_train_ds, range(len(y_train_array)), X_comp=X_comp)
    
    train_score = score(
                np.concatenate((y_train_array[range(len(y_train_array))].values[n_burn_in::], y_comp)), y_train_pred_array)
    print('train RMSE = %s' % (round(train_score, 3)))
    
    
    
    y_pred_array = test_submission(trained_model, X_ds, range(len(y_array)))
    y_pred_completed = np.concatenate((np.ones(1008), y_pred_array))[:-336]
    if assignment == "Téléphonie":
        index = X_ds.index.values.astype(np.datetime64)
        f_adjustment_bool = (index < np.datetime64('2013-06-22 00:00:00.000'))
        n = y_pred_completed.shape
        f_adjustment = np.ones(n[0]) + 0.15 * f_adjustment_bool.astype(int)[-n[0]:]
    else:
        f_adjustment = adjustment
    result = pd.DataFrame(X_ds.copy())
    result["prediction"] = (y_pred_completed * f_adjustment + .5).astype(int)
    result["DATE"] = result.index
    result.reset_index(drop=True)
    result.to_csv('./data/preds_full_range/' + assignment + "_predictions_full_range.csv", sep=";", index=False)
    print("Done")
    

def assignment_setting(assignment):
    """
    Create the right names for files, and adjustement factor
    """
    if assignment == "Téléphonie":
        return "ts_feature_extractor_Telephonie", "regressor_Telephonie", 1.15
    elif assignment == "Tech. Axa":
        return "ts_feature_extractor", "regressor", 1.05
    elif assignment == "CAT":
        return "ts_feature_extractor_CAT", "regressor_CAT", 1.11
    else:
        return "ts_feature_extractor", "regressor", 1.11


if __name__=="__main__":
    """
    Call the file with an assignment and a number between 0 and 2
    If this number is 0, compute only the cross validation on train and test data.
    If this number is 1, compute the cross validaiton and predict the all weeks.
    If this number is 2, compute only the prediction for all weeks
    """
    assignment = sys.argv[1]

    ts_feature_extractor_name, regressor_name, adjustment = assignment_setting(assignment)

    if len(sys.argv)>2:
        training = (int(sys.argv[2]) < 2)
        final = (int(sys.argv[2]) > 0)
    else:
        final = False
        training = True
    X_train_ds, y_train_array, X_test_ds, y_test_array = get_train_test_data(assignment)
    
    train_scores = []
    valid_scores = []
    test_scores = []

    data_list = pd.read_csv("./data/consecutive_training_days/" + assignment +"_list_com.csv", sep=",")
    startings = data_list["0"].values.astype(np.datetime64)
    endings = data_list["1"].values.astype(np.datetime64)

    list_comp = zip(startings, endings)

    X_comp, y_comp = get_compl_data(assignment, list_comp, './', ts_feature_extractor_name)
    len_comp = len(y_comp)
    if final:
        final_predict(assignment, list_comp, ts_feature_extractor_name, regressor_name, adjustment)
    if training:
        for number, (train_is, valid_is) in enumerate(get_cv(y_train_array)):
            
            limit_comp = len_comp/5*(number+1)
            trained_model = train_submission('./code/', X_train_ds, y_train_array, train_is, X_comp[0:limit_comp, :],
                                             y_comp[:limit_comp], ts_feature_extractor_name, regressor_name)
            
            y_train_pred_array = test_submission(trained_model, X_train_ds, train_is, X_comp=X_comp[0:limit_comp, :])

            train_score = score(
                np.concatenate((y_train_array[train_is].values[n_burn_in::], y_comp[0:limit_comp])), y_train_pred_array)

            y_valid_pred_array = test_submission(trained_model, X_train_ds, valid_is)
            valid_score = score(y_train_array[valid_is].values[n_burn_in::], y_valid_pred_array)


            y_test_pred_array = test_submission(trained_model, X_test_ds, range(len(y_test_array)), X_comp[limit_comp::, :])
            test_score = score(np.concatenate((y_test_array[n_burn_in::], y_comp[limit_comp:])), y_test_pred_array)

            print('train RMSE = %s; valid RMSE = %s; test RMSE = %s' %
                  (round(train_score, 3), round(valid_score, 3), round(test_score, 3)))

            train_scores.append(train_score)
            valid_scores.append(valid_score)
            test_scores.append(test_score)

        print(u'mean train RMSE = %s ± %s' %
              (round(np.mean(train_scores), 3), round(np.std(train_scores), 4)))
        print('mean valid RMSE = %s ± %s' %
              (round(np.mean(valid_scores), 3), round(np.std(valid_scores), 4)))
        print('mean test RMSE = %s ± %s' %
              (round(np.mean(test_scores), 3), round(np.std(test_scores), 4)))
