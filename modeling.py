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

def transform_features(data, sample_size = 200000, drop_columns = False):
    """Transforma features into ML-suitable structure

    Parameters
    ----------
    data : pandas DF
        DF containing loan level aggregations
    sample_size : int, optional
        The number of examples to be taken from the original data
    sample_size : boolean, optional
        Whether to drop highly correlated columns or not 
    
    Returns
    -------
    pandas DF
        a dataframe containing the transformed loan level aggregations
    """
    
    result = data.sample(sample_size)
    if drop_columns:
        result = result.drop(columns = [*])
    
    result = result.drop(columns = ['governorate'])
    display(result['pay_status'].value_counts())
    display(result.info())
    
    numeric_cols = result.select_dtypes(include=['int64','float64']).columns
    categorical_cols = result.select_dtypes(include=['object']).columns

    enc = OneHotEncoder(handle_unknown='error', sparse=False, drop='if_binary')
    transformed_x = enc.fit_transform(result[categorical_cols])
    transformed_x = pd.DataFrame(transformed_x, columns=enc.get_feature_names(categorical_cols))
    result = pd.concat([result,transformed_x.set_index(result.index)], axis = 1)
    result = result.drop(columns = categorical_cols)

    # pickle.dump(enc,open("classification_models/one_hot_encoder_sep.pkl", "wb"))

    display(result.head())
    
    display(result.shape)
    
    return result, enc



def split_data(data, split_overdue = True):
    """Splits the data into train and test and checks if a further split by overdue tiers
        should be made

    Parameters
    ----------
    data : pandas DF
        DF containing loan level aggregations
    split_overdue : boolean, optional
        Whether to split the data further into low overdue and high overdue data
    
    Returns
    -------
    numpy Arrays
        Multiple arrays containing the train and test data of each split
    """

    X = data[[col for col in data.columns if col != 'pay_status']].values
    y = data['pay_status'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
        
    return X_train, X_test, y_train, y_test        


def model_training(X_train, X_test, y_train, y_test):
    """Trains the model on the given train and test splits using the best hyperparameter
        tunings which uses Bayes Search

    Parameters
    ----------
    X_train : Numpy Array
        Array containing the training data
    X_test : Numpy Array
        Array containing the training labels
    y_train : Numpy Array
        Array containing the test data  
    y_test : Numpy Array
        Array containing the test labels    

    Returns
    -------
    xgb_clf: sklearn Classifier Object 
        Classifier object containing the final trained model
    y_pred: Numpy Array 
        Array containing the prediction labels        
    """
    
    print('********* Training Model *********\n')
    print('Number of training samples after sampling: \n')
    display(len(y_train))
    print('Distribution of training samples after sampling: \n')
    display(pd.Series(y_train).value_counts())

    data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
    
    params = {
            'n_estimators': [100, 200, 500, 750],
            'min_child_weight': [1, 2, 4, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5, 6],
    #         'learning_rate': [0.001, 0.01, 0.03,0.1, 0.3]
            }

    xg_reg = xgb.XGBClassifier(objective='reg:logistic'
#                                , scale_pos_weight = (y_train == 0).sum()/(y_train==1).sum()
                              )
    grid_search_xgb = skopt.BayesSearchCV(xg_reg, params, n_iter=40,cv=5, scoring='f1', n_points = 5
                                            ,return_train_score=True, verbose = 2, n_jobs = -1) 
    print('Grid Search Iterations: ',grid_search_xgb.total_iterations)
    grid_search_xgb.fit(X_train, y_train)

    display(grid_search_xgb.best_params_)

    xgb_clf = xgb.XGBClassifier(objective='reg:logistic'
#                             , scale_pos_weight = (y_train == 0).sum()/(y_train==1).sum()
                            ,**grid_search_xgb.best_params_
#                             ,n_estimators = 500, subsample = 1, min_child_weight = 1, max_depth = 5, gamma = 1, colsample_bytree = 0.8
                            ,random_state=40).fit(X_train, y_train)

#     xgb_clf = xgb.XGBClassifier(objective='reg:logistic'
#                             , scale_pos_weight = 1
#                             ,colsample_bytree = 0.8,
#                                 gamma = 1.0,
#                                 max_depth = 6,
#                                 min_child_weight = 1,
#                                 n_estimators = 100,
#                                 subsample = 0.6
# #                             ,n_estimators = 500, subsample = 1, min_child_weight = 1, max_depth = 5, gamma = 1, colsample_bytree = 0.8
#                             ,random_state=40).fit(X_train, y_train)

    y_pred = xgb_clf.predict(X_test)
    print('Distribution of predictive samples: \n')
    display(pd.Series(y_pred).value_counts())
    print('Model Report: \n')
    display(print(classification_report(y_test, y_pred)))
    
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize = (12,6))
    sns.heatmap(confusion_matrix, annot=True, fmt = 'g')
    plt.show()
    report = classification_report(y_test, y_pred, output_dict=True)
    return xgb_clf, y_pred, report

def train_xgboost(X_train, X_test, y_train, y_test, model, 
                  use_train_in_split = True, use_split = True):
    """Trains the model on the given train and test splits using the best hyperparameter
        tunings which uses Bayes Search

    Parameters
    ----------
    X_train : Numpy Array
        Array containing the training data
    X_test : Numpy Array
        Array containing the training labels
    y_train : Numpy Array
        Array containing the test data  
    y_test : Numpy Array
        Array containing the test labels 
    X_train_nop : Numpy Array
        Array containing the no provision training data 
    y_train_nop : Numpy Array
        Array containing the no provision training labels
    X_train_p : Numpy Array
        Array containing the provision training data
    y_train_p : Numpy Array
        Array containing the provision training labels 
    X_test_nop : Numpy Array
        Array containing the no provision testing data 
    y_test_nop : Numpy Array
        Array containing the no provision testing labels 
    X_test_p : Numpy Array
        Array containing the provision testing data    
    y_test_p : Numpy Array
        Array containing the provision testing labels
    use_train_in_split : boolean, optional
        Whether to use full training data to train the no provision or not
    use_split : boolean, optional
        Whether to split into two models or one model fits all   

    Returns
    -------
    xgb_clf: sklearn Classifier Object 
        Classifier object containing the final trained no provision model
    y_pred: Numpy Array 
        Array containing the no provision prediction labels  
    xgb_clf2: sklearn Classifier Object 
        Classifier object containing the final trained provision model
    y_pred2: Numpy Array 
        Array containing the provision prediction labels      
    """
    if model == 'provisions':
        print('Training the Provisions Model: \n')
        undersample = RandomUnderSampler(sampling_strategy=1)
        X_SM, y_SM = undersample.fit_resample(X_train, y_train)
        X_SM, y_SM = X_train, y_train
        print('Original training labels distribution for Provision: \n')
        display(pd.Series(y_train).value_counts())
        xgb_clf, y_pred, report = model_training(X_train, X_test, y_train, y_test) 
        return xgb_clf, y_pred, report
    
    elif model == 'variable_noprovisions':
        print('Training the Variable No Provisions Model: \n')
        print('********* Using only no provision train for no provision *********\n')
#             sm = SMOTE(random_state=42, sampling_strategy = )
#             X_SM, y_SM = sm.fit_sample(X_train_nop, y_train_nop)
        undersample = RandomUnderSampler(sampling_strategy=1)
        X_SM, y_SM = undersample.fit_resample(X_train, y_train)
#         X_SM, y_SM = X_train, y_train
        print('Original training labels distribution for Variable No Provision: \n')
        display(pd.Series(y_train).value_counts())
        xgb_clf, y_pred, report = model_training(X_SM, X_test, y_SM, y_test) 
        return xgb_clf, y_pred, report
    
    elif model == 'static_noprovisions':
        print('Training the Static No Provisions Model: \n')
        print('********* Using only no provision train for no provision *********\n')
#             sm = SMOTE(random_state=42, sampling_strategy = )
#             X_SM, y_SM = sm.fit_sample(X_train_nop, y_train_nop)
        undersample = RandomUnderSampler(sampling_strategy=0.1)
        X_SM, y_SM = undersample.fit_resample(X_train, y_train)
#         X_SM, y_SM = X_train, y_train
        print('Original training labels distribution for Static No Provision: \n')
        display(pd.Series(y_train).value_counts())
        xgb_clf, y_pred, report = model_training(X_SM, X_test, y_SM, y_test)
        return xgb_clf, y_pred, report
    
    else:    
        raise ValueError('Model type invalid') 

def dropCorrCols(df):
    cor_matrix = df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95) and column not in ['overdue_tiers_(-1, 0]','overdue_tiers_(0, 7]','overdue_tiers_(7, 30]','overdue_tiers_(30, 60]','overdue_tiers_(60, 90]','overdue_tiers_(90, 120]','overdue_tiers_(120, 10000]']]

    return to_drop
        

def processDF(df_sampled):
    df_sampled_noprovision = df_sampled[(df_sampled['overdue_tiers_(-1, 0]'] == 1) | (df_sampled['overdue_tiers_(0, 7]'] == 1)].copy()
    df_sampled_provision = df_sampled[(df_sampled['overdue_tiers_(-1, 0]'] == 0) & (df_sampled['overdue_tiers_(0, 7]'] == 0)].copy()   

    df_sampled_noprovision_variable = df_sampled_noprovision[(df_sampled_noprovision['never_overdue'] == 0)].copy()
    df_sampled_noprovision_static = df_sampled_noprovision[(df_sampled_noprovision['never_overdue'] == 1)].copy()

    df_sampled_noprovision_static['last_early_days_month'] = np.where(df_sampled_noprovision_static['last_early_days_isNA_month'] == 1, df_sampled_noprovision_static['avg_early_days_3m_month'], df_sampled_noprovision_static['last_early_days_month'])
    df_sampled_noprovision_static['avg_time_betw_payments_3m'] = np.where(df_sampled_noprovision_static['avg_time_betw_payments_3m_isna'] == 1, df_sampled_noprovision_static['avg_time_betw_payments'], df_sampled_noprovision_static['avg_time_betw_payments_3m'])
    df_sampled_noprovision_static['std_early_days_month'] = np.where(df_sampled_noprovision_static['std_early_days_isNA_month'] == 1, df_sampled_noprovision_static['std_early_days_3m_month'], df_sampled_noprovision_static['std_early_days_month'])

    df_sampled_noprovision_static['is_paid_month_1'] = np.where(df_sampled_noprovision_static['is_paid_month_1'] == 0, 1, df_sampled_noprovision_static['is_paid_month_1'])
    df_sampled_noprovision_static['is_paid_month_2'] = np.where(df_sampled_noprovision_static['is_paid_month_2'] == 0, 1, df_sampled_noprovision_static['is_paid_month_2'])
    df_sampled_noprovision_static['is_paid_month_3'] = np.where(df_sampled_noprovision_static['is_paid_month_3'] == 0, 1, df_sampled_noprovision_static['is_paid_month_3'])

    df_sampled_noprovision_variable['last_early_days_month'] = np.where(df_sampled_noprovision_variable['last_early_days_isNA_month'] == 1
                                                                        , -1, df_sampled_noprovision_variable['last_early_days_month'])

    df_sampled_provision_dropped_cols = dropCorrCols(df_sampled_provision)

    df_sampled_noprovision_variable_dropped_cols = dropCorrCols(df_sampled_noprovision_variable)

    df_sampled_noprovision_static_dropped_cols = dropCorrCols(df_sampled_noprovision_static)

    df_sampled_provision = df_sampled_provision.drop(columns = df_sampled_provision_dropped_cols)
    df_sampled_noprovision_variable = df_sampled_noprovision_variable.drop(columns = df_sampled_noprovision_variable_dropped_cols)
    df_sampled_noprovision_static = df_sampled_noprovision_static.drop(columns = df_sampled_noprovision_static_dropped_cols)

    return df_sampled_provision, df_sampled_noprovision_variable, df_sampled_noprovision_static, df_sampled_provision_dropped_cols, df_sampled_noprovision_variable_dropped_cols, df_sampled_noprovision_static_dropped_cols



def writeModels(xgb_clf_static, xgb_clf_variable, xgb_clf_provision, ohenc):
    
    s3_resource = boto3.resource('s3'
                                 , aws_access_key_id=settings.*
                                 , aws_secret_access_key=settings.*
                                )

    pickled_static = pickle.dumps(xgb_clf_static)
    s3_resource.Object(settings.*,
                       '*/cash_collection_optimization/cash_collection_opt_noprov_static_model.pkl').put(Body=pickled_static)

    pickled_variable = pickle.dumps(xgb_clf_variable)
    s3_resource.Object(settings.*,
                       '*/cash_collection_optimization/cash_collection_opt_noprov_variable_model.pkl').put(Body=pickled_variable)

    pickled_prov = pickle.dumps(xgb_clf_provision)
    s3_resource.Object(settings.*,
                       '*/cash_collection_optimization/cash_collection_opt_provision_model.pkl').put(Body=pickled_prov)                   
    
    pickled_onehot = pickle.dumps(ohenc)
    s3_resource.Object(settings.*,
                       '*/cash_collection_optimization/cash_collection_opt_one_hot_encoder.pkl').put(Body=pickled_onehot)
