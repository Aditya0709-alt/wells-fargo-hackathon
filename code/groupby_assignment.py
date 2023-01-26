import pandas as pd

if __name__ == "__main__":
    raw = pd.read_csv('./data/raw/train_2011_2012_2013.csv',
                      sep=';',
                      parse_dates=['DATE'])

    raw_grouped = raw.groupby(by=['ASS_ASSIGNMENT'])

    for assignment_name, assignment in raw_grouped:
        print assignment_name
        assignment[['DATE', 'CSPL_RECEIVED_CALLS']] \
        .groupby('DATE') \
        .sum() \
        .to_csv('./data/CSPL_RECEIVED_CALLS_series/' + assignment_name + '.csv', sep=';')