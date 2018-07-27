import os
import pandas as pd
import numpy as np
"""
##########################################################
########### Helper functions #############################
##########################################################
"""
def basic_info(df):
    """prints basic info about input data frame"""
    print('Num of rows and columns: ',df.shape)
    print('Missing value status: ',df.isnull().values.any())
    print('Columns names:\n ')
    for col in df.columns.values:
        print(col)
    #print(df.head())
    return 

def check_missing_data(df):
    """ return columns with missing data along with percentage """
    total = df.isnull().sum().sort_values(ascending = False)
    percent = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

def get_categorical_features(df):
    """returns list of names of categorical columns"""
    cat_features=df.columns[df.dtypes=='object']
    return list(cat_features)

def convert_Object_number(df, cols=None):
    ''' converts col data type from  object to number. Before transformation checks few values of each column and ingnore if not a number
    Paramters
    ---------
    df : data frame
    
    cols: list of column names then transformation will be done only for the list. cols default is None. if no columns passed to the function transformation willbe applied on all Object type cols
    
    Returns
    --------
    df : transformed dataframe
    '''
    import random
    if cols == None:
        cols = df[df.dtypes.loc[df.dtypes=='object'].index]
    for col in cols:
        try:
            # check few non null value is a number or not
            for x in range(5):
                rand_ix= random.randint(1,len(df))
                val = df.loc[rand_ix,col].values
                if is_number(val)== False:
                    print("All values are not numbers")
                    raise ValueError("All values are not numbers")
            df[col] = df[col].astype(np.float64)
            print('{} transformed from Object to Float'.format(col))
        except:
            print('Failed to transform column {}'.format(col))
            
    return df
    

def onehot_encoding(df,cat_features_name):
    """ returns dummy columns for features and avoids dummy column trap """
    df=pd.get_dummies(df,columns=cat_features_name, drop_first = True)
    return df

def dataFrame_size_MB(df):
    return df.memory_usage().sum() / 1024**2
    
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    print('Memory usage of dataframe is {:.2f} MB'.format(dataFrame_size_MB(df)))
    
    for col in df.columns:
        col_type = df[col].dtype
        #print('{} is type {}'.format(col,col_type))
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            if str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    df[col] = df[col].astype(np.float64)

    print('Memory usage of dataframe after memory optimization is {:.2f} MB'.format(dataFrame_size_MB(df)))
    return df



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def fill_numeric(df,strategy):
    ''' Fills missing numeric columns of dataframe
    Parameters
    ----------
    df - dataframe
    param strategy - mean, most_frequent, median or Number 
    '''
    from sklearn.preprocessing import Imputer
    imp=None
    if strategy == 'mean':
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    elif strategy == 'median':
        imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    elif strategy == 'most_frequent':
        imp = Imputer(missing_values='most_frequent', strategy='most_frequent', axis=0)
    elif is_number(strategy):
        df[df.dtypes.loc[df.dtypes!='object'].index]=df[df.dtypes.loc[df.dtypes!='object'].index].fillna(strategy)
        return df
    else:
        raise ValueError('invalid strategy')
        
    imp=imp.fit (df[df.dtypes.loc[df.dtypes!='object'].index])
        
    return imp.transform(df[df.dtypes.loc[df.dtypes!='object'].index])

def fill_categorical(df, strategy, values=None):
    '''
    fills missing categorical columns of dataframe
    Parameters
    ----------
    df : dataframe
    strategy :  most_frequent, values
    values : when strategy = 'values' then expects a dict with keys as column names and static reolacement values 
    
    Returns
    ----------
    df: Transformed dataframe
    '''
    
    if strategy == 'most_frequent':
        for col in df[df.dtypes.loc[df.dtypes=='object'].index]:
            value_counts=df[col].value_counts()
            max_val=value_counts.max()
            freq_val=value_counts[value_counts==max_val].index[0]
            df[col].fillna(freq_val, inplace=True)
        return df
    if strategy == "values" :
        #check values type
        if type(values)!=dict :
            raise ValueError('values should be type dict')
        # validate keys passed
        keys = list(values.keys())
        valid_keys = df.columns & keys
        if len(keys) != len(valid_keys):
            raise ValueError('values dict has invalid keys. All keys should be ')
        return df[df.dtypes.loc[df.dtypes=='object'].index].fillna(value=values)
    else:
        raise ValueError('invalid strategy')
    return df
"""
##########################################################
##########################################################
##########################################################
"""