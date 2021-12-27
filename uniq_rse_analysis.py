import pandas as pd
import sys
import os
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects



# Filtering according to target date and target time
# Cureent Target Time: 09:30 to 10:30 (1hour)
# If you need to change the time, you can modify following function.
def checktargetdate(x, targetdates):
    data = {}
    data['date'] = x[:8]
    if x[:8] in targetdates:
        if int(x[8:]) >= 93000 and int(x[8:])<=103000:
            data['check'] = 1
        else:
            data['check'] = 0
    else:
        data['check'] = 0
    return pd.Series(data=data, index=['date', 'check'])

def checktargetdate2(x, tdate):
    if x[:8] == tdate:
        return 1
    else:
        return 0


def checkarea(x, targetspot):
    if (x['BEFORE_RSE_ID'] in targetspot) or (x['RSE_ID'] in targetspot):
        return 1
    else:
        return 0


def checksection(x, br, r):
    if (x['BEFORE_RSE_ID'] == br) and (x['RSE_ID'] == r):
        return 1
    else:
        return 0

def checksection2(x, rsesct):
    if (x['BEFORE_RSE_ID'] + '-' + x['RSE_ID']) in rsesct:
        return x['BEFORE_RSE_ID'] + '-' + x['RSE_ID']
    else:
        return 'no target'

def add_median_lables(ax):
    lines = ax.get_lines()
    boxes = [child for child in ax.get_children() if type(child).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        text = ax.text(x, y, f'{y:.1f}', ha='center', va='center', fontweight='bold', color= 'white')
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])



def main(targetrsefile, targetdate):
    targetdates = []
    targetdates.append(targetdate)
    targetspot = ['RSE1507', 'RSE1504', 'RSE8102', 'RSE8507']    # Target RSE Spots in Target Area --> It could be changed and modified to others according to Target Area
    print('target date: {}'.format(targetdates))

    df = pd.read_excel(targetrsefile, sheet_name='RSE', engine='openpyxl')
    df = df.reset_index()
    df = df[['등록_시각', 'OBE_ID', '전 RSE ID', '현재 RSE ID', '전 수집 일자', '현재 수집 일자', '전-현재 통과시간(초)', '전-현재 처리에러코드']]
    df.columns = ['REG_YMDHMS', 'OBE_ID', 'BEFORE_RSE_ID', 'RSE_ID', 'BEFORE_COLCT_YMDHMS', 'NOW_COLCT_YMDHMS', 'BEFORE_NOW_TRVL_TM', 'BEFORE_NOW_PROCESS_ERROR_CD']

    df[['date', 'check']] = df['NOW_COLCT_YMDHMS'].apply(lambda x: checktargetdate(str(x),targetdates))
    is_target = df['check'] == 1
    df = df[is_target]
    df['check2'] = df[['BEFORE_RSE_ID', 'RSE_ID']].apply(lambda x: checkarea(x,targetspot), axis=1)
    is_target2 = df['check2'] == 1
    df = df[is_target2]

    is_check = df['BEFORE_NOW_PROCESS_ERROR_CD'] == 1305 # the code '1305' means the 'Have an Error'
    df = df[is_check]

    targetsection = ['RSE8507-RSE8102', 'RSE8102-RSE1504', 'RSE1504-RSE1507', 'RSE1507-RSE1504', 'RSE8102-RSE8507']

    print('Target RSE Section: {}'.format(targetsection))

    for tdate in targetdates:
        is_target3 = df['date'] == tdate
        temp = df[is_target3]
        print(temp)
        print('DATE: {}, Total No.: {} (Unique No.: {}) '.format(tdate, temp['OBE_ID'].count(), len(temp['OBE_ID'].unique())))

        temp['rse_section']= temp[['BEFORE_RSE_ID', 'RSE_ID']].apply(lambda x: checksection2(x, targetsection), axis=1)

        is_target_sct = temp['rse_section'] != 'no target'
        temp2 = temp[is_target_sct]
        temp2 = temp2.drop_duplicates(['OBE_ID', 'BEFORE_RSE_ID', 'RSE_ID'])
        temp2 = temp2.sort_values(by='rse_section')

        ## boxplot
        figure = plt.figure(figsize=(15, 12))
        sns.set(style='whitegrid')
        bplot = sns.boxplot(x='rse_section', y='BEFORE_NOW_TRVL_TM', data=temp2, showmeans=True,
                            meanprops={'marker': 'o', 'markerfacecolor': 'blue', 'markeredgecolor': 'black',
                                       'markersize': '10'},
                            width=0.5, linewidth=0.75)
        plt.ylim(0, 700)
        add_median_lables(bplot.axes)

        xlabels = [x.get_text() for x in bplot.get_xticklabels()]
        nobs = temp2.groupby('rse_section')['BEFORE_NOW_TRVL_TM'].count()

        for i, l in enumerate(xlabels):
            n = nobs[l]
            bplot.annotate('n={}'.format(n), xy=(i, 0.01), xycoords=('data', 'axes fraction'), ha='center')

        bplot.axes.set_title('Travel Time per RSE Section', fontsize=16)
        bplot.set_xlabel('RSE Sections', fontsize=18)
        bplot.set_ylabel('Travel Time [s]', fontsize=18)
        bplot.tick_params(labelsize=12)

        resultdir = './output'
        try:
            if not os.path.exists(resultdir):
                os.makedirs(resultdir)
        except OSError:
            print('[Error] Cannot create the Result Output Directory {}'.format(resultdir))

        resultfile = os.path.join(resultdir, 'RSE_Analysis_Result_Fig'+ tdate + '.jpg')
        bplot.figure.savefig(resultfile, format='jpeg', dpi=100)

        print('-----------------------------------------------------------------------------------------')
        for sc in targetsection:
            is_temp = temp2['rse_section'] == sc
            temp5 = temp2[is_temp]
            print('{} : {} / {}'.format(sc, len(temp5['OBE_ID'].unique()), temp5['OBE_ID'].count()))
            print(
                'sum: {}, mean: {}, median: {}'.format(temp5['BEFORE_NOW_TRVL_TM'].sum(), temp5['BEFORE_NOW_TRVL_TM'].mean(),
                                                       temp5['BEFORE_NOW_TRVL_TM'].median()))

        print('-----------------------------------------------------------------------------------------')

if __name__ == "__main__":
    if not main(sys.argv[1], sys.argv[2]):
        sys.exit(1)