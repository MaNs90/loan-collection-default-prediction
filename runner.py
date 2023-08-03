import boto3
import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import skopt
import pickle
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from dateutil import relativedelta

def trainMonth(df, LABEL_START_DATE, LABEL_END_DATE, max_month, loan_products):
   
    LAST_1M_TRAIN_START_DATE = get_previous_month_unix(LABEL_START_DATE)
    LAST_2M_TRAIN_START_DATE = get_previous_month_unix(get_previous_month_unix(LAST_1M_TRAIN_START_DATE))
    LAST_3M_TRAIN_START_DATE = get_previous_month_unix(LAST_2M_TRAIN_START_DATE)
    
    print(LAST_1M_TRAIN_START_DATE)
    print(LAST_2M_TRAIN_START_DATE)
    print(LAST_3M_TRAIN_START_DATE)
    
    month_start_str = convert_unix_to_date(LABEL_START_DATE)
    excluded_range = pd.date_range(month_start_str,max_month, freq='MS').strftime('%Y-%m-%d')#.astype(str)

    df_without_september = filterDF(df, LAST_1M_TRAIN_START_DATE, excluded_range)

    display(df_without_september.shape)

    df_without_september = get_initial_stats(df_without_september,LABEL_START_DATE)

    display(df_without_september['months_since_last_max_days'].value_counts())
    
    df_without_september = df_without_september[df_without_september['loanTenure'] > 2].copy()

    df_without_september2 = df_without_september.copy()
    
    df_without_september2['months_since_last_overdue_2'] = df_without_september2['months_since_last_overdue'] + 1
    df_without_september2['never_overdue'] = np.where(df_without_september2['months_since_last_overdue_2'] == 51, 1, 0)
    df_without_september2['months_since_last_overdue_2'] = df_without_september2['months_since_last_overdue_2'].replace({51:999})
    df_without_september2 = df_without_september2.drop(columns = 'months_since_last_overdue')
    df_without_september2 = df_without_september2.rename(columns = {'months_since_last_overdue_2':'months_since_last_overdue'})

    display(df_without_september2.shape)
    
    df_without_september2 = df_without_september2.join(loan_products.set_index('lsid'),how = 'inner').drop(columns = 'loan_status')
    
    display(df_without_september2.shape)
    
    df_without_september2 = get_mature_loans(df_without_september2,LABEL_START_DATE)

    df_without_september2 = get_labels(df_without_september2,LABEL_START_DATE,LABEL_END_DATE)

    df_without_september2 = remove_finished_loans(df_without_september2, LABEL_START_DATE)

    df_without_september2 = get_rescheduled_loans(df_without_september2, LABEL_START_DATE)

    # df_without_september2.drop(columns = ['ppinsts','earlyinst','avg_early_days','avg_early_days_3m','last_early_days','avg_early_days_isNA','avg_early_days_3m_isNA','last_early_days_isNA','perc_last_early_avg_early3m'],inplace = True)

    df_without_september2 = get_early_days(df_without_september2, LAST_1M_TRAIN_START_DATE, LABEL_START_DATE, LAST_3M_TRAIN_START_DATE)

    # df_without_september2 = get_early_days_provided(df_without_september2, LAST_1M_TRAIN_START_DATE, LABEL_START_DATE, LAST_3M_TRAIN_START_DATE)

    df_without_september2 = get_next_inst_status(df_without_september2,LABEL_START_DATE,LABEL_END_DATE)

    ts_converted = convert_unix_to_date(LAST_1M_TRAIN_START_DATE)
    print(ts_converted)
    df_without_september2 = df_without_september2.join(df[df['month'] == ts_converted].set_index('loanid')['overdue_tiers'])

    display(df_without_september2['overdue_tiers'].value_counts())

    df_without_september2 = get_days_till_next_inst(df_without_september2,LABEL_START_DATE,LABEL_END_DATE)

    display(pd.crosstab(df_without_september2['overdue_tiers'],df_without_september2['pay_status']))

    df_without_september2 = get_loan_history(df_without_september2, LAST_1M_TRAIN_START_DATE, LABEL_START_DATE, LAST_2M_TRAIN_START_DATE)

    display(df_without_september2.isna().sum().sum())


    ### Run till here for M-2

    df_without_september_shift2 = df_without_september2.copy() # M-2
    
    return df_without_september_shift2



def trainInstallments(df, base,num_months):
    print(df.shape)
    display(df['month'].value_counts())
    max_month = pd.to_datetime(df['month']).max().strftime('%Y-%m-%d')
    month_start_str = base
    month_start_dt = pd.to_datetime(base)
    month_start_unix = (month_start_dt - pd.Timedelta(hours = 2) - pd.to_datetime('1970-01-01')) // pd.Timedelta('1ms')
    month_end_dt = month_start_dt + relativedelta.relativedelta(months=1)
    month_end_unix = (month_end_dt - pd.Timedelta(hours = 2) - pd.to_datetime('1970-01-01')) // pd.Timedelta('1ms')
    month_end_str = month_end_dt.strftime('%Y-%m-%d')
    timeshifted_dfs = []
    loan_products = getLoanProducts()
    for i in range(num_months):
        print('Label Start: ', month_start_str)
        print('Label End: ', month_end_str)
        print('Label Start Unix: ', month_start_unix)
        print('Label End Unix: ', month_end_unix)
        print()
        
        train_timeshifted_df = trainMonth(df, month_start_unix , month_end_unix, max_month, loan_products)
        timeshifted_dfs.append(train_timeshifted_df)
        month_start_str = month_end_str
        month_start_dt = datetime.strptime(month_start_str,'%Y-%m-%d')
        month_start_unix = (month_start_dt - pd.Timedelta(hours = 2) - pd.to_datetime('1970-01-01')) // pd.Timedelta('1ms')
        month_end_dt = month_start_dt + relativedelta.relativedelta(months=1)
        month_end_unix = (month_end_dt - pd.Timedelta(hours = 2) - pd.to_datetime('1970-01-01')) // pd.Timedelta('1ms')
        month_end_str = month_end_dt.strftime('%Y-%m-%d')
        
    train_df = pd.concat(timeshifted_dfs)  
    display(train_df['pay_status'].value_counts())
    
    return train_df


def generate_next_month(LABEL_START_DATE, LABEL_END_DATE, data, df):

    LAST_1M_TRAIN_START_DATE = get_previous_month_unix(LABEL_START_DATE)
    LAST_2M_TRAIN_START_DATE = get_previous_month_unix(get_previous_month_unix(LAST_1M_TRAIN_START_DATE))
    LAST_3M_TRAIN_START_DATE = get_previous_month_unix(LAST_2M_TRAIN_START_DATE)

    data.head()
    
    df_without_september2 = get_mature_loans(data,LABEL_START_DATE)
    
    df_without_september2['months_since_last_overdue_2'] = df_without_september2['months_since_last_overdue'] + 1
    df_without_september2['never_overdue'] = np.where(df_without_september2['months_since_last_overdue_2'] == 51, 1, 0)
    df_without_september2['months_since_last_overdue_2'] = df_without_september2['months_since_last_overdue_2'].replace({51:999})
    df_without_september2 = df_without_september2.drop(columns = 'months_since_last_overdue')
    df_without_september2 = df_without_september2.rename(columns = {'months_since_last_overdue_2':'months_since_last_overdue'})
    
    loan_products = getLoanProducts()
    
    df_without_september2 = df_without_september2.join(loan_products.set_index('lsid'),how = 'inner').drop(columns = 'loan_status')

    df_without_september2 = get_included_loans(df_without_september2,LABEL_START_DATE,LABEL_END_DATE)

    df_without_september2 = remove_finished_loans(df_without_september2, LABEL_START_DATE)

    df_without_september2 = get_rescheduled_loans(df_without_september2, LABEL_START_DATE)
    
    df_without_september2 = get_early_days_test(df_without_september2, LAST_1M_TRAIN_START_DATE, LABEL_START_DATE, LAST_3M_TRAIN_START_DATE)
    
    df_without_september2 = get_next_inst_status(df_without_september2,LABEL_START_DATE,LABEL_END_DATE)

    ts_converted = convert_unix_to_date(LAST_1M_TRAIN_START_DATE)
    print('Date of getting overdue tiers: ',ts_converted)
    df_without_september2 = df_without_september2.join(df[df['month'] == ts_converted].set_index('loanid')['overdue_tiers'])

    df_without_september2['overdue_tiers'].value_counts()

    df_without_september2.info()

    df_without_september2 = get_days_till_next_inst(df_without_september2,LABEL_START_DATE,LABEL_END_DATE)

    df_without_september2 = get_loan_history(df_without_september2, LAST_1M_TRAIN_START_DATE, LABEL_START_DATE, LAST_2M_TRAIN_START_DATE)
    
    print('Date of getting days/installments overdue: ',ts_converted)

    df_without_september2.isna().sum()
    
    display(df_without_september2.isna().sum())

    df_without_september2.drop(columns = 'product_id',inplace = True)
    df_test = transform_features_test(df_without_september2,drop_columns=False)

    return df_test

if __name__ == '__main__':

  train_df = trainInstallments()
  generate_next_month()
  
