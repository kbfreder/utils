import pickle
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta as rd
from calendar import monthrange as mr


def pkl_this(filename, df):
    '''Saves df as filename. Must include .pkl extension'''
    with open(filename, 'wb') as picklefile:
        pickle.dump(df, picklefile)

def open_pkl(filename):
    '''Must include .pkl extension. Returns same object as original.
    '''
    with open(filename,'rb') as picklefile:
        return pickle.load(picklefile)



def num_not_zero(data):
    return len(data[data != 0])

def num_pos(data):
    return len(data[data > 0])

def trimmed_mean(data, ptile=95):
    cutoff = np.percentile(data, ptile)
    trimmed_data = data[data <= cutoff].dropna()
    return np.mean(trimmed_data)

def pos_lpc(df, lpc_col = 'LATE_CHARGE'):
    return df[df[lpc_col] >= 0]

def non_neg_lpc(df, lpc_col = 'LATE_CHARGE'):
    return df[df[lpc_col] >= 0]

def fix_nan_lpc(row):
    next_lpc = row['NEXT_INV_LPC']
    if np.isnan(next_lpc) & (row['BILL_DATE'] <= dt.datetime.today() - dt.timedelta(days=30)):
        return 0
    else:
        return next_lpc


def check_dup_cols(df, col_name, suffixes):
    col_sfx_1 = col_name + suffixes[0]
    col_sfx_2 = col_name + suffixes[1]

    check = np.sum(df[col_sfx_1] == df[col_sfx_2]) == len(df)
    print(check)
    if not check:
        return df[df[col_sfx_1] != df[col_sfx_2]][[col_sfx_1, col_sfx_2]]


def df_rel_lens(df1, df2):
    return len(df1), len(df2), len(df1)/len(df2)


def date_months_prior(n):
        '''Return datetime object x months prior from today
        rounded to first day of month'''
        t = dt.datetime.today()
        exact_dt = t - rd(months=n)
        return dt.datetime(exact_dt.year, exact_dt.month, 1)


def last_months_last_day():
        '''Return datetime object for last day of previous month
        
                Ex: today = 2019-06-15
                
                last_months_last_day()
                >>> Timestamp (2019, 5, 31)
        '''
        t = dt.datetime.today()
        m = t.month
        y = t.year
        if m == 1:
                y -= 1
                m = 12
        else:
                m -= 1
        
        d = mr(y, m)[1]

        return dt.datetime(y, m, d)


def find_val_in_df(val, df):
        '''Return name(s) of column(s) in 'df' in which 'val' is found'''

        for col in list(df.columns):
                if val in list(df[col].unique()):
                        print(col)


def compare_dfs(df1, df2):
    '''Returns DataFrame listing differences (by index & column) between df1 & df2
    Note: df1 & df2 must share the same index.
    '''
    # fillna's to aid in comparison (otherwise nulls return error?)
    df1 = df1.fillna('none specified')
    df2 = df2.fillna('none specified')

    # which entries have changed
    ne_stack = (df1 != df2).stack()
    changed = ne_stack[ne_stack]
    changed.index.names = ['IDX', 'COL']

    diff_idx = np.where(df1 != df2)

    chg_from = df1.values[diff_idx]
    chg_to = df2.values[diff_idx]
    res = pd.DataFrame({'from': chg_from, 'to': chg_to}, index=changed.index)

    return res


def check_for_nulls(df):
    '''Return a DataFrame listing columns in df that contain nulls, and the # of nulls'''
    n = df.shape[0]
    null_dict = {}
    for col in list(df.columns):
        if df[col].isnull().any():
            num_null = np.sum(df[col].isnull())
            null_dict[col] = [num_null, np.round(((num_null/n) * 100), 1)]
    
    if len(null_dict) > 0:
        null_df = pd.DataFrame.from_dict(null_dict, orient='index')
        null_df.reset_index(inplace=True)
        null_df.columns = ['Column', 'Num Nulls', '% Null']
        null_df.sort_values(by='Column', inplace=True)
    else:
        null_df = pd.DataFrame()

    return null_df


def null_report(df):
    '''Return a DataFrame listing the # of nulls in each column in df'''
    n = df.shape[0]
    null_dict = {}
    idx = 0
    for col in list(df.columns):
        num_null = np.sum(df[col].isnull())
        null_dict[idx] = [col, num_null, np.round((num_null/n), 4)]
        idx += 1
    
    if len(null_dict) > 0:
        null_df = pd.DataFrame.from_dict(null_dict, orient='index')
        null_df.reset_index(inplace=True)
        null_df.columns = ['Index', 'Column', 'Num Nulls', '% Null']
        null_df.sort_values(by='Index', inplace=True)
    else:
        null_df = pd.DataFrame()

    return null_df


def distro_val_cts(df, col):
    ''' Return horizontal bar chart of value counts of 'col' in 'df' '''
    print('Number Unique values: ', df[col].nunique())
    vals = df[col].value_counts()
    plt.barh(range(len(vals)), vals)
    plt.xscale('log');
    plt.title(col)
    plt.ylabel(col + ' values (arbitrary)')
    plt.xlabel('Value Count')
    return vals


def write_to_excel(filename, df1, sheetname1, index_bool1, *args):
    '''Write df(s) to Excel

    filename: Name of Excel file to save as.
              do not include path (../../reports) or file extension (.xlsx)

    df1 will be saved as Sheet sheetname 1. index_bool indicates whether to include Index
    
    optional: additional df, sheetnames, index_bool to write to Excel file
              (provide as list)

    Ex: 
        write_to_excel('report', summary_df, 'Summary', True, comparison_report, 'Report', False)
    '''
    assert len(args) // 3 == 1
        
    with pd.ExcelWriter('../../reports/' + filename + '.xlsx') as writer:
        df1.to_excel(writer, sheet_name=sheetname1, index=index_bool1)  
        if len(args) > 0:
            triplets = list(zip(*[args[i::3] for i in range(3)]))
            
            for t in triplets:
                dfx = t[0]
                sheetnamex = t[1]
                index_boolx = t[2]
                dfx.to_excel(writer, sheet_name=sheetnamex, index=index_boolx)


def get_project_path():
    '''Return path of project folder
        assuming cookiecutter structure, this is two folders "up"
    
    Note, calling notebook must have run `import os`
    '''
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(os.path.curdir)))

def get_save_path():
    import os

    # set save path for queries
    cwd = os.getcwd()
    if '/' in cwd:
        char = '/'
    else:
        char = '\\'

    path_keyword = os.getcwd().split(char)[-2]

    try:
        if path_keyword == 'src' or path_keyword == 'notebooks':
        # cookiecutter folder structure
            save_path = '../../data/raw/'
        else:
        # KF's old folder structure
            save_path = 'Data/'
    except IndexError:
        save_path = 'Data/'
    
    return save_path