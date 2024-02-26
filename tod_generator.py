'''
CODE 예시
pip install -r requirements.txt
python tod_generator.py --input-dir ./data --output-dir ./result --max-tod 10
'''

import os
import warnings
import sys

import numpy as np
import pandas as pd
import argparse

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

if not sys.warnoptions:
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', required = True, help = 'directory including inputs', type=str)
parser.add_argument('--output-dir', required = True, help = 'directory to save outputs', type=str)
parser.add_argument('--max-tod', required = False, help = 'maximum number of TOD groups', type=int, default = 10)

args = parser.parse_args()
pd.set_option('mode.use_inf_as_na', True)

def load_data(input_dir):
    """
    function: input으로부터 입력 자료로 읽은 후 1차 가공 (시간 변수, 15분 단위)된 데이터를 반환
    input: input_dir (입력 자료를 포함한 디렉토리)
    output: df (1차 가공된 15분 단위 교통량 데이터)
    """
    df = pd.read_csv(input_dir + "/traffic_input.csv")


    #시간 변수 가공
    df['REG_DT'] = pd.to_datetime(df['REG_DT'], format = '%Y-%m-%d %H:%M:%S')
    df['DOW'] = df['REG_DT'].dt.dayofweek
    df['DOW'] = np.where(df['DOW'] < 5, 0, df['DOW'] - 4)

    df['DATE'] = df['REG_DT'].dt.date.astype(str)
    df['HOUR'] = df['REG_DT'].dt.hour.astype(int)

    sa = pd.read_csv(input_dir + "/crsrd_sa.csv")
    return df, sa

def aggregate_data(data, sa):
    """
    function: 1차 가공된 데이터로부터 진출입 방향별, 요일 (평일, 토, 일), 시간 (1시간 단위, 24시간)별 평균 교통량 계산
    input: data (1차 가공된 1시간 단위 교통량 데이터)
    output: df_agg (2차 가공된 방향별 (columns), 시간대별 (index) 교통량)
    """

    num_days = 3
    num_hours = 24

    df_agg = {'DOW': np.repeat(range(num_days), num_hours),
              'HOUR': np.tile(range(num_hours), num_days)}

    df_agg = pd.DataFrame(df_agg)
    df_agg.drop_duplicates(inplace = True)
    df_agg_dict = {}

    for sa_id in set(sa.SA):
        crsrd_list = sa.loc[sa['SA']==sa_id].CRSRD_ID.values
        temp = data[data['CRSRD_ID'].isin(crsrd_list)]

        temp_agg = temp.groupby(['DATE','DOW','HOUR','CRSRD_ID','DIR']).TRF.sum().reset_index()
        temp_agg = temp_agg.pivot_table(index=['DOW', 'HOUR'],
                                        columns=['CRSRD_ID', 'DIR'],
                                        values='TRF', aggfunc='mean').fillna(0)
        temp_agg.columns = temp_agg.columns.map(lambda x: '-'.join([str(i) for i in x]))

        temp_agg['TOTAL_VOLUME'] = temp_agg.sum(axis=1)
        temp_agg.iloc[:, :-1] = temp_agg.iloc[:, :-1].div(temp_agg['TOTAL_VOLUME'], axis=0)
        temp_agg['TOTAL_VOLUME'] = temp_agg['TOTAL_VOLUME'] / (temp_agg['TOTAL_VOLUME'].sum())
        temp_agg.reset_index(inplace=True)

        temp_agg = pd.merge(df_agg, temp_agg, how='left', on=['DOW', 'HOUR']).fillna(0)

        df_agg_dict[sa_id] = temp_agg

    return df_agg_dict

def reduce_dim_pca(data):
    """
    function: PCA (주성분분석)을 통한 차원 축소 (비슷한 교통류 방향끼리 묶음)
    input: data (2차 가공된 방향별 (columns), 시간대별 (index) 교통량)
    output: pca_array (차원축소된 행렬 - 시간 특성 없음)
     """
    data.sort_values(['DOW','HOUR'], inplace =True)

    pca = PCA(0.9)
    pca_array = pca.fit_transform(data.drop(['DOW', 'HOUR'], axis=1).values * 100)

    return pca_array


def tod_cluster(pca_array, timeframe, max_tod):
    """
    function: 클러스터링 기반의 TOD 생성,
              K-Means 와 Hierarchical Clustering 기법 중 더 높은 Silhouette score를 갖는 기법 및 클러스터 수에 대해 시간 군집 수행
    input: pca_array, timeframe (TOD 분할 후 label용), max_tod (최대 고려 가능 TOD 수)
    output: timeframe (각 시간대별 TOD Plan)
     """
    max_silhouette = 0
    opt_k = 0
    sil_score = []

    for i in range(3, max_tod):
        km_model = KMeans(n_clusters=i, random_state=23)
        km_TOD = km_model.fit_predict(pca_array)
        km_sil = silhouette_score(pca_array, km_TOD)

        h_model = AgglomerativeClustering(n_clusters=i, linkage='ward', affinity='euclidean')
        h_TOD = h_model.fit_predict(pca_array)
        h_sil = silhouette_score(pca_array, h_TOD)

        sil_score.append(['K-Means', i, km_sil])
        sil_score.append(['Hierarchical', i, h_sil])

    sil_score = pd.DataFrame(sil_score)
    sil_score.columns = ['CLUSTER_METHOD', 'NUM_CLUSTER', 'SCORE']
    opt_params = sil_score.iloc[sil_score['SCORE'].idxmax()]

    if opt_params['CLUSTER_METHOD'] == 'K-Means':
        model = KMeans(n_clusters=opt_params['NUM_CLUSTER'], random_state=23)
        tod = model.fit_predict(pca_array)
        print("K-Means Clustering 기반의 " + str(opt_params['NUM_CLUSTER']) + "개의 TOD가 생성되었습니다.")
    else:
        model = AgglomerativeClustering(n_clusters=opt_params['NUM_CLUSTER'], linkage='ward', affinity='euclidean')
        tod = model.fit_predict(pca_array)
        print("Hierarchical Clustering 기반의 " + str(opt_params['NUM_CLUSTER']) + "개의 TOD가 생성되었습니다.")

    timeframe['TOD_PLAN'] = tod

    return timeframe


if __name__ == '__main__':
    dirname = os.path.join(os.getcwd(), args.output_dir)

    if os.path.dirname != '' and not os.path.exists(dirname):
        os.makedirs(dirname)

    df, sa = load_data(args.input_dir)
    df_agg_dict = aggregate_data(df, sa)

    tod_time_table_sa = pd.DataFrame()

    for sa_id in df_agg_dict:
        print("SA: ", sa_id)
        df_agg = df_agg_dict[sa_id]
        df_norm_dim = reduce_dim_pca(df_agg)

        tod_time_table = tod_cluster(df_norm_dim, df_agg[['DOW','HOUR']], int(args.max_tod))
        tod_time_table['SA'] = sa_id

        tod_time_table_sa = pd.concat([tod_time_table_sa,tod_time_table], ignore_index=True)


    tod_time_table_sa.to_csv(args.output_dir + '/' + "TOD_TABLE_SA.csv", index = False, encoding = 'utf-8-sig')

    print("완료하였습니다.")
