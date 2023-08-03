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
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from IPython.display import display
from django.conf import settings

def getLoanProducts():
    """Gets the date of the previous month

    Parameters
    ----------
    None

    Returns
    -------
    Pandas DF
        All loan products that are non-seasonal
    """

    engine = create_engine(settings.*)
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''
        loan_products = pd.read_sql(SQL,conn)

    return loan_products    

def filterDF(data, ts, date_range):
    ts_converted = convert_unix_to_date(ts)
    result = data[data['loanid'].isin(data[data['month'] == ts_converted]['loanid'].unique())]
    display(result['month'].value_counts())
    
    result = result[~result['month'].isin(date_range)][[***]].copy()
    
    excluded_months = ['2019-09-01','2019-10-01','2019-11-01','2019-12-01']
    result = result[~result['month'].isin(excluded_months)].copy()
    display(result['month'].value_counts())
    
    return result

def get_initial_stats(data, ts):
    """Performs first level aggregations of the data, its considered the base dataframe

    Parameters
    ----------
    data : pandas DF
        Initial DF containing loan level data per month
    timestamp : int
        Unix timestamp of first day of the month after the last training month

    Returns
    -------
    pandas DF
        a dataframe containing first level aggregations of the data
    """
    # Added 2 hours because pd.Timestamp converts this to UTC
    ts_converted = pd.Timestamp(ts, unit = 'ms') + pd.Timedelta(hours = 2)

    result = data.sort_values('month').groupby('loanid').agg({
        'principal': 'first',
        'issueDate':[lambda x: int((ts - x.iloc[0])
                                   /(60*60*24*30*1000)),
                     lambda x: int((ts - x.iloc[0])
                                   /(60*60*24*1000)),
                     lambda x: int((x.iloc[0] - data.loc[x.idxmax(),'birthdate'])/(60*60*24*30*12*1000))],
        'principal_overdue': 'last',
        'interest_overdue': 'last',
        'installments_overdue': lambda x: 50 if sum(x == 0) == len(x) else round((ts_converted - pd.Timestamp(data.loc[x[x != 0].index[-1],'month']))/np.timedelta64(1, 'M')) - 1,
        'days_overdue':['max',
#                         lambda x: relativedelta.relativedelta(ts_converted,
#                                                               pd.Timestamp(data.loc[x.idxmax(),'month'])).months 
#                         + relativedelta.relativedelta(ts_converted,
#                                                       pd.Timestamp(data.loc[x.idxmax(),'month'])).years * 12 -1
                        lambda x: 50 if sum(x == 0) == len(x) else round((ts_converted - pd.Timestamp(data.loc[x.idxmax(),'month']))/np.timedelta64(1, 'M')) - 1
                       ],
        'beneficiarytype_s': 'first',
        'wallet': ['last',
                   lambda x: x.iloc[-1]/ data.loc[x.index[-1],'principal']],
        'businesssector':'first',
        'provision':['last', 
                     'max'],
        'gender': 'first',
        'governorate': 'first'
    })

    result.columns = [***]
    result['is_new'] = np.where(result['loanTenure'] <= 2, 1, 0)
    
    return result

def get_mature_loans(data,ts):
    """Gets all loans who are still active after the last training date
        and merges them to the aggregated df

    Parameters
    ----------
    data : pandas DF
        DF containing loan level aggregations
    timestamp : int
        Unix timestamp of first day of the month after the last training month

    Returns
    -------
    pandas DF
        a dataframe containing loan level aggregations plus an indicator variable for mature loans
    """
    engine = create_engine(settings.*)
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''
        mature_loans = pd.read_sql(SQL,conn, params = {'end_date':ts})
        
        
    mature_loans['is_mature'] = 0
    result = data.join(mature_loans.set_index('loanid'))
    result['is_mature'] = result['is_mature'].fillna(1)
    display(result['is_mature'].value_counts())
    
    return result

def get_labels(data, ts_start, ts_end):
    """Gets all loans who have a due installment between ts_start and ts_end
        and is not rescheduled and get all pay_installment transactions in the same period

    Parameters
    ----------
    data : pandas DF
        DF containing loan level aggregations
    ts_start : int
        Unix timestamp of first day of the label month 
    ts_end : int
        Unix timestamp of first day of the month after the label month    

    Returns
    -------
    pandas DF
        a dataframe containing loan level aggregations plus a label column indicating if a loan has
            been paid in the above period or not
    """
    
    # Any loan that has been rescheduled during that period is considered paid
    engine = create_engine(settings.*)
    with engine.connect() as conn, conn.begin():
        SQL = '''
         ***
        '''
        loan_labels = pd.read_sql(SQL,conn, params = {'start_date':ts_start,'end_date':ts_end})
    
    display(loan_labels.shape)
    loan_labels['pay_status'] = loan_labels['pay_status'].map({0:1,1:0})
    display(loan_labels['pay_status'].value_counts())

    with engine.connect() as conn, conn.begin():
        SQL = '''
         ***
        '''
        loan_labels2 = pd.read_sql(SQL,conn, params = {'start_date':ts_start,'end_date':ts_end})   
        
    display(loan_labels2.shape)   
    display(loan_labels2['transactionamount'].isna().sum())
    
    # Use Outer if you want all active loans to be with you, use Inner if you want only the loans who have dues next month
    final_labels = loan_labels.set_index('loanid').join(loan_labels2.set_index('loan_id'), how = 'outer').dropna(how='all')
    final_labels['pay_status2'] = np.where(final_labels['transactionamount'].isna(), 1, 0)
    final_labels['pay_status'] = final_labels['pay_status'].fillna(final_labels['pay_status2'])
    final_labels['pay_status'] = np.where(final_labels['pay_status2'] == 0, 0, final_labels['pay_status'])
    display(pd.crosstab(final_labels['pay_status'],final_labels['pay_status2']))
    display(final_labels['pay_status'].isna().sum())
    
    result = data.join(final_labels['pay_status'])
    display(result.shape)
    display(result['pay_status'].value_counts())

    # display(result[(result['pay_status'].isna()) 
    #                       & (result['last_principal_overdue'] == 0)
    #                       & (result['is_mature'] == 1)
    #                      ])
    # I will fillna only for the loans who have an overdue principal but do not have a due date next month 
    # because they should come pay their dues. However, the loans with nans in their labels but do not have 
    # any overdue principal I wll drop them because they do not have anything to pay next month
    result[(result['pay_status'].isna()) & (result['last_principal_overdue'] != 0)] = result[(result['pay_status'].isna()) 
                                                                                             & (result['last_principal_overdue'] != 0)].fillna(1)
    # display(result[(result['pay_status'].isna())])
    # These NAs are due to them not having any due installments next month and they have no overdue amounts so they have no reason to pay anything
    result[(result['pay_status'].isna())]['is_mature'].value_counts()
    result = result.dropna(subset = ['pay_status'])
    display(result.shape)
    display(result.info())
    
    return result

def remove_finished_loans(data, ts):
    """Removes early payment loans and loans that paid their last installment before ts period
        ####OR 10/2020 so they did not appear in end of 10/2020 df (WRONG!)

    Parameters
    ----------
    data : pandas DF
        DF containing loan level aggregations
    ts : int
        Unix timestamp of first day of the label month 
    
    Returns
    -------
    pandas DF
        a dataframe containing loan level aggregations excluding removed loans
    """
    engine = create_engine(settings.*)
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''

        loans_included = pd.read_sql(SQL,conn, params = {'end_date':ts})
    
    result = data.join(loans_included.set_index('loanid'), how = 'inner')
    result = result.drop(columns = 'lastpaymentdate')
    display(result.shape)
    
    return result

def get_rescheduled_loans(data, ts):
    """Gets all rescheduled and postponed installments along with their stats

    Parameters
    ----------
    data : pandas DF
        DF containing loan level aggregations
    ts : int
        Unix timestamp of first day of the label month 
    
    Returns
    -------
    pandas DF
        a dataframe containing loan level aggregations plus rescheduling aggregations
    """
    engine = create_engine(settings.*)
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''
        loan_reschedules = pd.read_sql(SQL,conn, params = {'end_date':ts})
        
    result = data.join(loan_reschedules.set_index('loanid'))
    result[['rescheduledinsts','postponedinsts','rescheduledtimes']] = result[['rescheduledinsts','postponedinsts','rescheduledtimes']].fillna(0)

    result['last_rescheduled_date'] = result['last_rescheduled_date'].fillna(1514757600000)
    result['last_postponed_date'] = result['last_postponed_date'].fillna(1514757600000) 

    result['last_rescheduled_date'] =  ((ts - result['last_rescheduled_date'])/(60*60*24*1000)).astype(int)
    result['last_postponed_date'] =  ((ts - result['last_postponed_date'])/(60*60*24*1000)).astype(int)

    display(result.isna().sum())
    
    return result
    

def get_early_days(data, ts_start, ts_end, ts_3m):
    """Gets all early and late days statistics which serve as the payment patterns

    Parameters
    ----------
    data : pandas DF
        DF containing loan level aggregations
    ts_start : int
        Unix timestamp of first day of the month before label month 
    ts_end : int
        Unix timestamp of first day of the label month 
    ts_3m : int
        Unix timestamp of first day of the 4th month before the label month      
    
    Returns
    -------
    pandas DF
        a dataframe containing loan level aggregations plus payment patterns aggregations
    """
    engine = create_engine(settings.*)
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''
        loan_info = pd.read_sql(SQL,conn, params = {'start_date':ts_start,'end_date':ts_end,'prev3m_date':ts_3m})
        
    display(loan_info[loan_info['loanid'] == '5ff09f2e77cc9b55f40f277b'])  
    loan_info.columns = ['loanid','ppinsts_month','previous_defaults','previous_paid_intime','earlyinst_month','avg_early_days_month','avg_early_days_3m_month','last_early_days_month','std_early_days_month','std_early_days_3m_month']
    result = data.join(loan_info.set_index('loanid'), rsuffix = '_month')
    
    result = result[result['loanTenure'] > 1].copy()

    display(result.shape)

    display(result['pay_status'].value_counts())

    display(result.isna().sum())

    result = result.dropna(subset = ['ppinsts_month','earlyinst_month'])

    display(result.isna().sum())

    result['avg_early_days_isNA_month'] = np.where(result['avg_early_days_month'].isna(),1,0)
    result['avg_early_days_3m_isNA_month'] = np.where(result['avg_early_days_3m_month'].isna(),1,0)
    result['last_early_days_isNA_month'] = np.where(result['last_early_days_month'].isna(),1,0)
    result['std_early_days_3m_isNA_month'] = np.where(result['std_early_days_3m_month'].isna(),1,0)
    result['std_early_days_isNA_month'] = np.where(result['std_early_days_month'].isna(),1,0)
    
    result.loc[result[(result['is_mature'] == 0)].index,'avg_early_days_3m_month'] = result.loc[result[(result['is_mature'] == 0)].index,'avg_early_days_3m_month'].fillna(result.loc[result[(result['is_mature'] == 0)].index,'avg_early_days_3m_month'].dropna().min()*2)

    result.loc[result[(result['is_mature'] == 1)].index,'avg_early_days_3m_month'] = result.loc[result[(result['is_mature'] == 1)].index,'avg_early_days_3m_month'].fillna(result.loc[result[(result['is_mature'] == 1)].index,'avg_early_days_3m_month'].dropna().min()*2)

    result.loc[result[(result['is_mature'] == 0)].index,'last_early_days_month'] = result.loc[result[(result['is_mature'] == 0)].index,'last_early_days_month'].fillna(result.loc[result[(result['is_mature'] == 0)].index,'avg_early_days_3m_month']-30)
    result.loc[result[(result['is_mature'] == 1)].index,'last_early_days_month'] = result.loc[result[(result['is_mature'] == 1)].index,'last_early_days_month'].fillna(result.loc[result[(result['is_mature'] == 1)].index,'avg_early_days_3m_month']-30)
    
    result['std_early_days_month'] = result['std_early_days_month'].fillna(-1)
    result['std_early_days_3m_month'] = result['std_early_days_3m_month'].fillna(-1)
    
    display(result.isna().sum())
    result = result.dropna()
    display(result.shape)
    
    return result

def get_next_inst_status(data, ts_start, ts_end):
    """Gets the status of the installment between ts_start and ts_end

    Parameters
    ----------
    data : pandas DF
        DF containing loan level aggregations
    ts_start : int
        Unix timestamp of first day of the label month  
    ts_end : int
        Unix timestamp of first day of the month after the label month     
    
    Returns
    -------
    pandas DF
        a dataframe containing loan level aggregations plus next installment status
    """
    engine = create_engine(settings.*)
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''
        inst_reschedule = pd.read_sql(SQL,conn, params = {'start_date':ts_start,'end_date':ts_end})
        
    display(inst_reschedule[(inst_reschedule['is_rescheduled_inst'] + inst_reschedule['is_postponed_inst']) > 1])
    
    result = data.join(inst_reschedule.set_index('loanid'))
    
    display(result['is_rescheduled_inst'].isna().sum())
    display(result[result['is_rescheduled_inst'].isna()])
    
    result['is_rescheduled_inst'] = result['is_rescheduled_inst'].fillna(0)
    result['is_postponed_inst'] = result['is_postponed_inst'].fillna(0)
    
    display(result.shape)
    display(result['is_rescheduled_inst'].value_counts())
    
    return result

def get_days_till_next_inst(data, ts_start, ts_end):
    """Gets the days left till next installment is due

    Parameters
    ----------
    data : pandas DF
        DF containing loan level aggregations
    ts_start : int
        Unix timestamp of first day of the label month  
    ts_end : int
        Unix timestamp of first day of the month after the label month     
    
    Returns
    -------
    pandas DF
        a dataframe containing loan level aggregations plus days left till next inst
    """
    engine = create_engine(settings.*)
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''
        loans_endofmonth = pd.read_sql(SQL,conn, params = {'start_date':ts_start,'end_date':ts_end})
        
    result = data.join(loans_endofmonth.set_index('loanid'))
    display(result.isna().sum())
#     result['days_till_endofmonth_NA'] = np.where(result['days_till_endofmonth'].isna(),1,0)
    result['days_till_next_inst_NA'] = np.where(result['days_till_next_inst'].isna(),1,0)
#     result['days_till_endofmonth'] = result['days_till_endofmonth'].fillna(-1)
    result['days_till_next_inst'] = result['days_till_next_inst'].fillna(-1)
    
    display(result.isna().sum())
    return result

def get_last_activity(data, ts):
    """Gets last activity on a loan recorded in transactions table

    Parameters
    ----------
    data : pandas DF
        DF containing loan level aggregations
    ts : int
        Unix timestamp of first day of the label month 
    
    Returns
    -------
    pandas DF
        a dataframe containing loan level aggregations plus days since last activity
    """
    engine = create_engine(settings.*)
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''
        loans_lastactivity = pd.read_sql(SQL,conn, params = {'end_date':ts})
    
    result = data.join(loans_lastactivity.set_index('loan_id'), how = 'inner')
    result['last_activity_date'] = result['last_activity_date'].fillna(1514757600000)
    result['last_activity_date'] = ((ts - result['last_activity_date'])/(60*60*24*1000)).astype(int)
        
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''
        loans_lastactivity2 = pd.read_sql(SQL,conn, params = {'end_date':ts})
        
    result = result.join(loans_lastactivity2.set_index('loan_id'), how = 'inner')  
    
    display(result.isna().sum())
    
    return result

def get_last_change_mom(orig_df, data, ts):
    """Calculates the last two months rate of change between multiple columns

    Parameters
    ----------
    orig_df : pandas DF
        DF containing all active loans each month for the last year  
    data : pandas DF
        DF containing loan level aggregations
    ts : int
        Unix timestamp of first day of the month before the label month    
    
    Returns
    -------
    pandas DF
        a dataframe containing loan level aggregations plus change MoM
    """
    ts_converted = convert_unix_to_date(ts)
    ts_converted_neg1 = get_previous_month_date(ts)
    ts_converted_neg2 = get_previous_month_date(get_previous_month_unix(ts))
    
    df_percdiff = orig_df[orig_df['month'].isin([
                                       ts_converted_neg1,
                                       ts_converted])].sort_values('month').groupby('loanid')[['principal_overdue','interest_overdue',
                                                                                               'days_overdue','installments_overdue',
                                                                                               'principal_paid','wallet','provision']].pct_change()
    df_percdiff.columns = [col + '_change' for col in df_percdiff.columns]
    df_percdiff = df_percdiff.dropna(how='all')
    df_percdiff = df_percdiff.replace([np.inf, -np.inf], 1).fillna(0).join(orig_df['loanid'])
    display(df_percdiff['loanid'].nunique())
    
    result = data.join(df_percdiff.set_index('loanid'))
    display(result.isna().sum())
    
    result = result.dropna(subset = df_percdiff.set_index('loanid').columns)
    
    df_std = orig_df[orig_df['month'].isin([ts_converted_neg2,
                                       ts_converted_neg1,
                                       ts_converted])].sort_values('month').groupby('loanid')[['principal_overdue','interest_overdue',
                                                                                           'days_overdue','installments_overdue',
                                                                                           'principal_paid','wallet','provision']].std()
    df_std.columns = [col + '_std' for col in df_std.columns]
    df_std = df_std.dropna(how='all')
    result = result.join(df_std)
    
    return result

def get_loan_history(data, ts_start, ts_end, ts_3m):
    """Gets the last three months loan history between ts_start and ts_end

    Parameters
    ----------
    data : pandas DF
        DF containing loan level aggregations
    ts_start : int
        Unix timestamp of first day of the month before label month  
    ts_end : int
        Unix timestamp of first day of the label month     
    
    Returns
    -------
    pandas DF
        a dataframe containing loan level aggregations plus loan history
    """
    engine = create_engine(settings.*)
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''
        loan_history = pd.read_sql(SQL,conn, params = {'start_date':ts_3m,'end_date':ts_end})
        
    loan_history = loan_history.groupby(['loan_id','to_char']).sum().unstack()
    loan_history.columns = ['paid_month_3','paid_month_2','paid_month_1','paid_insts_3','paid_insts_2','paid_insts_1']
#     loan_history.columns = ['paid_insts_3','paid_insts_2','paid_insts_1']
    
    loan_history[['is_paid_month_3','is_paid_month_2','is_paid_month_1']] = loan_history[['paid_month_3','paid_month_2','paid_month_1']].fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    display(loan_history)
    
    result = data.join(loan_history).fillna(0)
    result['paid_month_last3'] = result['paid_month_3'] + result['paid_month_2'] + result['paid_month_1']
    result['is_paid_month_last3'] = result['is_paid_month_3'] + result['is_paid_month_2'] + result['is_paid_month_1']
    
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''
        loan_installments_payment = pd.read_sql(SQL,conn, params = {'start_date':ts_start,'end_date':ts_end,'prev3m_date':ts_3m})
        
    result = result.join(loan_installments_payment.set_index('lsid'),how = 'inner')     
    
    with engine.connect() as conn, conn.begin():
        SQL = '''
        ***
        ;
        '''
        loan_installments_info = pd.read_sql(SQL,conn, params = {'start_date':ts_start,'end_date':ts_end,'prev3m_date':ts_3m})
    
    loan_installments_info['months_since_last_payment'] = np.ceil((ts_end - loan_installments_info['last_payment_date'])/(60*60*24*30*1000)).fillna(999)#|.astype(int)
    result = result.join(loan_installments_info.set_index('lsid'),how = 'inner')
    
    result.drop(columns = ['last_payment_date'],inplace = True)
    result['avg_time_betw_payments_isna'] = np.where(result['avg_time_betw_payments'].isna(),1,0)
    result['avg_time_betw_payments_3m_isna'] = np.where(result['avg_time_betw_payments_3m'].isna(),1,0)
    result.fillna(result['avg_time_betw_payments'].mean(),inplace = True)
    result.fillna(result['avg_time_betw_payments_3m'].mean(), inplace = True)
    
    return result

def getTiers(data):
    if data['overdue_tiers_(-1, 0]'] == 1:
        return '(-1, 0]'
    elif data['overdue_tiers_(0, 7]'] == 1:
        return '(0, 7]'
    elif data['overdue_tiers_(7, 30]'] == 1:
        return '(7, 30]'
    elif data['overdue_tiers_(30, 60]'] == 1:
        return '(30, 60]'
    elif data['overdue_tiers_(60, 90]'] == 1:
        return '(60, 90]'
    elif data['overdue_tiers_(90, 120]'] == 1:
        return '(90, 120]'
    elif data['overdue_tiers_(120, 10000]'] == 1:
        return '(120, 10000]'
