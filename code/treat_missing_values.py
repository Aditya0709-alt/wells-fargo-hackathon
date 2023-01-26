import os
import pandas as pd
import numpy as np

def check_dataframe(X_ds):
    # Some constants
    timedelta = np.datetime64('2013-01-28 00:30:00.000') - np.datetime64('2013-01-28 00:00:00.000')
    startdate = np.datetime64('2011-01-01 00:00:00.000')
    enddate = np.datetime64('2013-12-31 23:30:00.000')

    #Dates that are present in the dataframe
    gotten_values = X_ds.index.values
    gotten_values.sort()

    #Dates that are not in the dataframe
    added_values = []

    #counter will parcour all the dates between startdate and enddate
    counter = startdate

    #indice is where we are in the sortted array of gotten values
    indice = 0

    #Loop over all dates, if they are present, increase counter and indice, 
    #else, put them in added_values and increase only counter
    while counter <= enddate:
        if indice == len(gotten_values):
            added_values.append(counter)
            counter += timedelta
        else:
            if (counter == gotten_values[indice]):
                counter += timedelta
                indice += 1
            else:
                if counter > gotten_values[indice]:
                    print("ERROR")
                    break
                else:
                    added_values.append(counter)
                    counter += timedelta

    #  creation of a dataframe with all new values
    df_temp = pd.DataFrame.from_dict({"DATE" : added_values, "CSPL_RECEIVED_CALLS": np.zeros(len(added_values))})
    df_temp.index=df_temp["DATE"]
    df_temp = df_temp["CSPL_RECEIVED_CALLS"]

    #Concatenation with the old one

    result = pd.concat([X_ds, df_temp])
    result = result.sort_index()

    # Replace added_values by the mean of the previous and the following value
    # or by the previous value if the following one is in added_values
    for i, x in enumerate(added_values):
        if not i%500:
            print( "value number " + str(i) )
        if x == startdate:
            result[x] = result[x+timedelta]
        elif (x+timedelta in added_values) | (x == enddate):
            result[x] = result[x-timedelta]
        else:
            result[x] = (result[x-timedelta] + result[x + timedelta])/2
    
    return result

if '__name__' == '__main__':
    files = os.listdir("./data/CSPL_RECEIVED_CALLS_series/")
    files = [elt for elt in files if '.csv' in elt]
    for file_name in files:
        data = pd.read_csv("./data/CSPL_RECEIVED_CALLS_series/" + file_name, sep=";")
        X_ds = data["CSPL_RECEIVED_CALLS"]
        X_ds.index = data["DATE"].values.astype(np.datetime64)
        result = check_dataframe(X_ds)
        result.to_csv("./data/treated_data/" + file_name, sep=";")