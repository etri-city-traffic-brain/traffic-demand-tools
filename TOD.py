'''
Using dataset from smart intersection, time table with TOD labels is estimated by K-Means method
* Unit: 30 minute
* Single intersection
* Go-direction traffic includes right-turn traffic

* Input dataset:
- ORT_CCTV_5MIN_LOG
- ORT_CCTV_MST

* Output:
- TOD table
- Traffic analysis according to each TOD period (Traffic: veh/30mins)

* Example code: python TOD.py --crsrd-id 1860001900 --input-dir ./data --output-dir ./result --max-tod 4

* Range of SC Interpretation
0.71-1.0 A strong structure has been found
0.51-0.70 A reasonable structure has been found
0.26-0.50 The structure is weak and could be artificial
< 0.25 No substantial structure has been found

'''
import argparse
import os
import statistics

import pandas as pd
# from yellowbrick.cluster import KElbowVisualizer
from dplython import (DplyFrame, X, select, sift, group_by, summarize)
from sklearn import preprocessing
from sklearn.cluster import KMeans

# import numpy as np
# from pandas import DataFrame
# import csv

parser = argparse.ArgumentParser()
parser.add_argument('--crsrd-id', required = True, help = 'ID for the crossroad of interest', type=str)
parser.add_argument('--input-dir', required = True, help = 'directory including inputs', type=str)
parser.add_argument('--output-dir', required = True, help = 'directory to save outputs', type=str)
parser.add_argument('--max-tod', required = False, help = 'maximum number of TOD groups', type=int, default = 4)
# parser.add_argument('--vis', required = False, help = 'visualize result(1) or not(0, default)', type=int, default = 0)
args = parser.parse_args()

pd.set_option('mode.use_inf_as_na', True)

def load_data(input_dir, crsrd_id):
    cctv_log = pd.read_csv(input_dir + "/ORT_CCTV_5MIN_LOG.csv")
    cctv_mst = pd.read_csv(input_dir + "/ORT_CCTV_MST.csv")

    cctv_log['DATE'] = pd.DataFrame(pd.DatetimeIndex(cctv_log['REG_DT']).date)
    cctv_log['HOUR'] = pd.DataFrame(pd.DatetimeIndex(cctv_log['REG_DT']).hour)
    cctv_log['MINUTE'] = (pd.DataFrame(pd.DatetimeIndex(cctv_log['REG_DT']).minute) // 30) * 30
    cctv_log['temp_DAY'] = pd.to_datetime(cctv_log['DATE']).dt.dayofweek
    cctv_log.loc[cctv_log['temp_DAY'] < 5, 'DAY'] = int(0) #mon - fri
    cctv_log.loc[cctv_log['temp_DAY'] == 5, 'DAY'] = int(1) #sat
    cctv_log.loc[cctv_log['temp_DAY'] == 6, 'DAY'] = int(2) #sun
    df0 = DplyFrame(cctv_log) >> group_by(X.DATE, X.DAY, X.HOUR, X.MINUTE, X.CCTV_ID)>> summarize(GO_TRF=X.GO_BIKE.sum() + X.GO_CAR.sum() + X.GO_SUV.sum() + X.GO_VAN.sum() + X.GO_TRUCK.sum() + X.GO_BUS.sum() + X.RIGHT_BIKE.sum() + X.RIGHT_CAR.sum() + X.RIGHT_SUV.sum() + X.RIGHT_VAN.sum() + X.RIGHT_TRUCK.sum() + X.RIGHT_BUS.sum(),
                                                                                                  LEFT_TRF=X.LEFT_BIKE.sum() + X.LEFT_CAR.sum() + X.LEFT_SUV.sum() + X.LEFT_VAN.sum() + X.LEFT_TRUCK.sum() + X.LEFT_BUS.sum())
    # Extract records of selected crossroad
    cctv_mst = DplyFrame(cctv_mst) >> sift(X.CRSRD_ID  == crsrd_id) >> select(X.CRSRD_ID, X.CCTV_ID)
    df0 = pd.merge(df0, cctv_mst, how = "inner", on = "CCTV_ID")
    df0 = df0.sort_values(['DATE','HOUR','MINUTE','CCTV_ID'])

    # Time frame from existing dataset
    tf = DplyFrame(df0.drop_duplicates(['DATE','DAY','HOUR','MINUTE'], keep='last')) >> select(X.DATE, X.DAY, X.HOUR, X.MINUTE)

    # Process the datastructure into pivot
    cctv_list = sorted(cctv_mst['CCTV_ID'].unique())
    df1 = tf

    for cctv in cctv_list:
        a = df0 >> sift(X.CCTV_ID == cctv) >> select(X.DATE, X.DAY, X.HOUR, X.MINUTE, X.GO_TRF, X.LEFT_TRF)
        df1 = pd.merge(df1, a, how='left', on=['DATE', 'DAY', 'HOUR', 'MINUTE'], suffixes=('', '_' + str(cctv)))

    df1 = df1.set_index(['DATE', 'DAY', 'HOUR', 'MINUTE'])
    df1 = df1.fillna(df1.rolling(window = 24, min_periods = 1, center = True).mean())
    df1 = df1.fillna(0)
    df1 = df1.reset_index()

    df1['TOTAL_TRF'] = DplyFrame(df1.iloc[:, 4:3 + len(cctv_list) * 2].sum(axis=1, skipna=True))
    df1 = df1 >> sift(X.TOTAL_TRF > 0)
    print(df1)
    # Name the cctv id and direction - for tod_traffic_analyzer

    cols = [cctv + '_GO_RATE' for cctv in cctv_list]
    cols.extend([cctv + '_LEFT_RATE' for cctv in cctv_list])
    cols = sorted(cols)
    cols = ['TOD'] + cols + ['TOTAL_TRF']

    return df1, cols

def estimate_tod(dat, max_k):
    max_score = 0
    k = 1
    input_dat = data_normalize(dat)

    # for i in range(2, max_k):
    #     model = KMeans(n_clusters = i, random_state = 10)
    #     TOD = model.fit_predict(input_dat)
    #     score = silhouette_score(input_dat, TOD)
    #     print("Silhouette score of {} clusters is {}".format(i, score))
    #     if(score > max_score):
    #         k = i
    #         max_score = score
    #
    # print("Best number of clusters is {} with score {}".format(k, max_score))
    # model = KMeans(n_clusters=k, random_state=10)
    # TOD = model.fit_predict(input_dat)
    # dat['TOD'] = TOD

    model = KMeans(n_clusters=max_k, random_state=10)
    TOD = model.fit_predict(input_dat)
    dat['TOD'] = TOD

    dat_TOD = pd.DataFrame(dat.groupby(['DAY', 'HOUR', 'MINUTE'])['TOD'].apply(statistics.mode)).reset_index()

    return dat_TOD

def data_normalize(dat):
    temp = dat.drop(['DATE','DAY','HOUR','MINUTE'], axis = 1)
    a = temp.drop(['TOTAL_TRF'], axis = 1).to_numpy()
    b = temp['TOTAL_TRF'].to_numpy().reshape(-1,1)
    temp = pd.DataFrame(a/b)
    temp = temp.fillna(0)
    # temp = temp.dropna(axis = 0)

    #vertical normalize: total traffic across time-series
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(dat[['TOTAL_TRF']])
    temp2 = pd.DataFrame(minmax_scale.transform(dat[['TOTAL_TRF']]))
    temp = pd.concat([temp, temp2], axis = 1)

    return temp

def visualize_tod(dat,max_k): #Sihouette score and distribution of TOD groups into each time slide
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1, max_k), metric='silhouette', timings=True)
    visualizer.fit(dat)
    visualizer.show()

def tod_traffic_analyzer(day_index, dat, cols):
    a = dat.drop(['DATE','DAY','HOUR','MINUTE'], axis = 1).groupby("TOD").mean()
    a1 = a.drop(['TOTAL_TRF'], axis = 1).to_numpy()
    a2 = a['TOTAL_TRF'].to_numpy().reshape(-1,1)

    #calculate turning rate
    a1 = pd.DataFrame(a1/a2)
    turning_rate = pd.concat([round(a1,2), a['TOTAL_TRF']], axis = 1)
    turning_rate = turning_rate.reset_index()

    turning_rate.columns = cols
    turning_rate['DAY'] = day_index

    return turning_rate

def run_process(dat, cols, max_k, vis=0):
    tod_result = pd.DataFrame()
    info_result = pd.DataFrame()

    #iterate from weekday, saturday and sunday (0,3)
    for i in range(3):
        temp = dat >> sift(X.DAY == i)
        tod_result = tod_result.append(estimate_tod(temp, max_k))
        info_result = info_result.append(tod_traffic_analyzer(i, temp, cols))

        # if(vis == 1):
        #     print("Visualize the Silhouette score of day group (0:weekday, 1: sat, 2:sun)", i)
        #     visualize_tod(k)

    return tod_result, info_result


if __name__ == '__main__':
    dirname = os.path.dirname(args.output_dir)
    k = int(args.max_tod)

    if os.path.dirname!='' and not os.path.exists(dirname):
        os.makedirs(dirname)

    if(k < 2):
        print("maximum number of TOD should be equal or larger than 2")
    else:
        df, col_name = load_data(args.input_dir, int(args.crsrd_id))
        r1, r2 = run_process(df, col_name, k) #r1: tod, r2: traffic attributes for each tod

        r1.to_csv(args.output_dir + '/tod_result_' + args.crsrd_id + ".csv")
        r2.to_csv(args.output_dir + '/traffic_characteristics_' + args.crsrd_id + ".csv")
