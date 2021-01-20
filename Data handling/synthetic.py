# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:35:52 2020

@author: EU
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import os

writing_path = (r'\\cs.aau.dk\Fileshares\IT703e20\(NEW)CleanDatasets\\')
reading_path = (r'\\cs.aau.dk\Fileshares\IT703e20\(NEW)CleanDatasets\\')
original_csv_path = 'data4project_global.csv'
ds1_csv_path = 'unique-noBlanks-noNegatives.csv'
ds2_csv_path = 'no0s-unique-noBlanks-noNegatives.csv'
number_of_splitting_chunks = 5


def main():
    """
    print("READING ds2")
    dataset = pd.read_csv(reading_path + ds2_csv_path)
    dataset['CALDAY'] = pd.to_datetime(dataset['CALDAY'].astype(str), format='%Y%m%d')
    dataset = dataset.sort_values('CALDAY')
    
    print('___________________________________')
    makeDataset(dataset, 100000, "100k", "ds2_100k_timeDistributed")
    
    print('___________________________________')
    makeDataset(dataset, 500000, "500k", "ds2_500k_timeDistributed")
    print('___________________________________')
    makeDataset(dataset, 1000000, "1m", "ds2_1m_timeDistributed")
    
    print('___________________________________')
    makeDataset(dataset, 2000000, "OG(2m)", "ds2_OG(2m)_timeDistributed")
    
    print("___________________________________")
    makeDataset(dataset, 5000000, "5m", "ds2_5m_timeDistributed")
    
    print('___________________________________')
    makeDataset(dataset, 10000000, "10m", "ds2_10m_timeDistributed")
    
    print('___________________________________')
    """
    mini_set = "100k"
    readingSplitDatasetSubRoutine(mini_set, "ds2_" + mini_set + "_timeDistributed")
    
def readingSplitDatasetSubRoutine(mini_set, file_name):
    mini_set = "/SVD/" + mini_set
    df = pd.DataFrame()
    for id in range(number_of_splitting_chunks):
        df1 = pd.read_csv(reading_path + mini_set + "/" + file_name + '_{id}.csv'.format(id=id+1))
        df = df.append(df1)
    print(str(len(df)) + " | i= " + str(id+1))
    #makeNCFDatasets(df, mini_set, file_name)
    df['NORMALIZED_CUSTOMER_ID'] = pd.factorize(df['CUSTOMER_ID'])[0]
    
    bin_labels_5 = [1, 2, 3, 4]
    df['NORMALIZED_TRANSACTION_COUNT'] = pd.cut(df['TRANSACTION_COUNT'],
                              bins=4,
                              labels=bin_labels_5)
    df['NORMALIZED_QUANTITY_SUM'] = pd.cut(df['QUANTITY_SUM'],
                              bins=4,
                              labels=bin_labels_5)
    df['NORMALIZED_TIME_DIFF_DAYS'] = pd.cut(df['TIME_DIFF_DAYS'],
                              bins=4,
                              labels=bin_labels_5)
        
    print(pd.qcut(df['TRANSACTION_COUNT'],
                              q=4,duplicates='drop').value_counts())
    print(pd.qcut(df['QUANTITY_SUM'],
                              q=4).value_counts())
    
    print(pd.qcut(df['TIME_DIFF_DAYS'],
                              q=5,duplicates='drop').value_counts())
    df['NORMALIZED_TRANSACTION_COUNT'] = pd.to_numeric(df['NORMALIZED_TRANSACTION_COUNT'])
    df['NORMALIZED_QUANTITY_SUM'] = pd.to_numeric(df['NORMALIZED_QUANTITY_SUM'])
    df['NORMALIZED_TIME_DIFF_DAYS'] = pd.to_numeric(df['NORMALIZED_TIME_DIFF_DAYS'])
    
    df['RATING'] =  df['NORMALIZED_TIME_DIFF_DAYS'] * df['NORMALIZED_QUANTITY_SUM'] * df['NORMALIZED_TRANSACTION_COUNT'] / 3

    splitDataSets(df[['NORMALIZED_CUSTOMER_ID','PRODUCT_ID',
             'NORMALIZED_TRANSACTION_COUNT',
             'NORMALIZED_QUANTITY_SUM', 'NORMALIZED_TIME_DIFF_DAYS', 'RATING'             
             ]], number_of_splitting_chunks, mini_set, file_name+"_normalized_in_quartiles")

    df = df[['CUSTOMER_ID', 'NORMALIZED_CUSTOMER_ID','MATERIAL','PRODUCT_ID',
             'TRANSACTION_COUNT', 'NORMALIZED_TRANSACTION_COUNT','QUANTITY_SUM',
             'NORMALIZED_QUANTITY_SUM','FIRST_PURCHASE','LAST_PURCHASE', 
             'TIME_DIFF_DAYS', 'NORMALIZED_TIME_DIFF_DAYS', 'RATING'             
             ]]
    
    
    #splitDataSets(df, number_of_splitting_chunks, mini_set, file_name+"_normalized_in_quartiles")

def makeDataset(data, size, mini_set, file_name):
    print("PREPROCESSING")
    
    print("- DROPPING columns")
    original_data = data.drop(['DOC_NUMBER_TEXT', 'is_accessory', 'MATERIAL_TEXT', 'MATL_GROUP'], axis=1)
    
    clean_data = addRatings(original_data)
    
    material_product_dict = dict(zip(clean_data['PRODUCT_ID'], clean_data['MATERIAL']))
    customer_normalized_dict = dict(zip(clean_data['CUSTOMER_ID'], clean_data['NORMALIZED_CUSTOMER_ID']))
    while size > len(clean_data):
        clean_data = clean_data.sort_values(['FIRST_PURCHASE', 'LAST_PURCHASE'])
        temp_data = generateSyntethic(clean_data, (size - len(clean_data)))
        temp_data = addingOffsetDates(temp_data)
        print('- - MERGING')
        clean_data = clean_data.append(temp_data)
        clean_data = clean_data.groupby(['CUSTOMER_ID', 'PRODUCT_ID'], as_index=False).first()
        print('- - DUPPED SIZE: ' + str(len(clean_data) + len(temp_data))+ ' | CLEAN SIZE: ' + str(len(clean_data)))
    print("- FIXING material - product mapping")
    clean_data['MATERIAL'] = clean_data['PRODUCT_ID']
    clean_data['MATERIAL'] = clean_data['MATERIAL'].map(material_product_dict)    
    del material_product_dict
    
    print("- FIXING customer_id - normalized_customer_id mapping")
    clean_data['NORMALIZED_CUSTOMER_ID'] = clean_data['CUSTOMER_ID']
    clean_data['NORMALIZED_CUSTOMER_ID'] = clean_data['NORMALIZED_CUSTOMER_ID'].map(customer_normalized_dict)   
    
    print(len(clean_data))
    clean_data['TIME_DIFF_DAYS'] = (clean_data['LAST_PURCHASE'] - clean_data['FIRST_PURCHASE']).astype('timedelta64[D]')
    
    clean_data = clean_data.sort_values('FIRST_PURCHASE')
    clean_data = clean_data.head(size)
    print("|||||| PRINTING INFO |||||")
    print("time info:")
    print("- starting: ") 
    print(clean_data['FIRST_PURCHASE'].head(1))
    print("- ending: ")
    print(clean_data['FIRST_PURCHASE'].tail(1))
    
    a = clean_data['FIRST_PURCHASE'].iloc[1]
    b = clean_data['FIRST_PURCHASE'].iloc[-1]

    delta = b - a
    print('- TIME PERIOD:')
    print(delta)
    print('FINAL SIZE: ' + str(len(clean_data)))
    
    splitDataSets(clean_data, number_of_splitting_chunks, "SVD/" + mini_set, file_name)
    print("MAKING TT")
    clean_data = clean_data.drop(['LAST_PURCHASE', 'FIRST_PURCHASE', 'QUANTITY_SUM', 'TRANSACTION_COUNT', 'TIME_DIFF_DAYS'], axis=1)
    splitDataSets(clean_data, number_of_splitting_chunks, "TT/" + mini_set, file_name)
    
    makeNCFDatasets(clean_data, mini_set, file_name)
    
def makeNCFDatasets(clean_data, mini_set, file_name):    
    print("MAKING NCF")
    neg_df = generateNegativeFeedback(clean_data, len (clean_data) * 2)
    neg_df_list = list()
    clean_data['RATING_TYPE'] = 1
    for data in  enumerate(np.array_split(neg_df, number_of_splitting_chunks)):
        neg_df_list.append(data[1])
    print("APPENDING NEGATIVE FEEDBACK")
    for id,data in  enumerate(np.array_split(clean_data, number_of_splitting_chunks)):
        print(type(neg_df_list[id]))
        df = pd.DataFrame(neg_df_list[id])
        data = data.append(df)
        writeToCsv(data, "NCF/" + mini_set, file_name + '_{id}.csv'.format(id=id+1))
    
def addRatings(data):
    print("ADDING ratings")

    print("- ADDING TRANSACTION COUNT COLUMN")
    ds = data.sort_values(by=['CUSTOMER_ID', 'MATERIAL'])
    ds = data.groupby(['CUSTOMER_ID','MATERIAL'])['QUANTITY'].count().reset_index()
    ds.set_index(['CUSTOMER_ID','MATERIAL'])
    ds = ds.rename({'QUANTITY': 'TRANSACTION_COUNT'}, axis=1)
    
    print("- ADDING QUANTITY SUM COLUMN")
    ds_sum = data.sort_values(by=['CUSTOMER_ID', 'MATERIAL'])
    ds_sum = data.groupby(['CUSTOMER_ID', 'MATERIAL'])['QUANTITY'].sum().reset_index()
    ds_sum.set_index(['CUSTOMER_ID','MATERIAL'])
    ds['QUANTITY_SUM'] = ds_sum['QUANTITY']
    del ds_sum   
    
    print("- ADDING timestamp columns")
    time_ds = data.sort_values(by=['CUSTOMER_ID', 'MATERIAL'])
    time_ds['FIRST_PURCHASE'] = data['CALDAY']
    time_ds['LAST_PURCHASE'] = data['CALDAY']
    time_ds = time_ds.groupby(['CUSTOMER_ID', 'MATERIAL']).agg(
        {"FIRST_PURCHASE": "min", "LAST_PURCHASE": "max"}).reset_index()
    time_ds.set_index(['CUSTOMER_ID','MATERIAL'])
    ds['FIRST_PURCHASE'] = time_ds['FIRST_PURCHASE']
    ds['LAST_PURCHASE'] = time_ds['LAST_PURCHASE']
    del time_ds
    ds['TIME_DIFF_DAYS'] = (ds['LAST_PURCHASE'] - ds['FIRST_PURCHASE']).astype('timedelta64[D]')
    
    print("- - - positive differences:" + str(len(ds[ds['TIME_DIFF_DAYS'] > 0])))
    print("- - - negative differences:" + str(len(ds[ds['TIME_DIFF_DAYS'] < 0])))
    print("- - - neutral differences:" + str(len(ds[ds['TIME_DIFF_DAYS'] == 0])))
    
    print("- ADDING PRODUCT_ID column")
    ds['PRODUCT_ID'] = pd.factorize(ds['MATERIAL'])[0]
    ds['NORMALIZED_CUSTOMER_ID'] = pd.factorize(ds['CUSTOMER_ID'])[0]
    
    print("- REORDERING columns")
    ds = ds[['CUSTOMER_ID', 'NORMALIZED_CUSTOMER_ID', 'MATERIAL', 'PRODUCT_ID', 'TRANSACTION_COUNT', 'QUANTITY_SUM',
             'FIRST_PURCHASE', 'LAST_PURCHASE', 'TIME_DIFF_DAYS']]
    return ds


def generateSyntethic(data, requiredSize):
    print("GENERATING " + str(requiredSize) + " synthetic structured transactions")
    print("- SHUFFILING data")
    shuffle = data[['CUSTOMER_ID', 'PRODUCT_ID']]
    shuffle.reset_index(inplace=True, drop=True)
    for _ in range(1):
        shuffle.apply(np.random.shuffle, axis=0)
        
    shuffle.reset_index(inplace=True, drop=True)
    syntethic = data.copy()
    syntethic.reset_index(inplace=True, drop=True)
    syntethic['CUSTOMER_ID'] = shuffle['CUSTOMER_ID']
    syntethic['PRODUCT_ID'] = shuffle['PRODUCT_ID']
    syntethic.reset_index(inplace=True, drop=True)
    
    return syntethic

def addingOffsetDates(data):
    print("- ADDING dates")
    syntethic = data
    syntethic['FIRST_PURCHASE']= pd.to_datetime(syntethic['FIRST_PURCHASE'])
    diff = syntethic['FIRST_PURCHASE'].iloc[-1].year - syntethic['FIRST_PURCHASE'].iloc[1].year
    syntethic['FIRST_PURCHASE'] = syntethic['FIRST_PURCHASE'] + pd.DateOffset(years=diff)
    
    syntethic['LAST_PURCHASE']= pd.to_datetime(syntethic['LAST_PURCHASE'])
    diff = syntethic['LAST_PURCHASE'].iloc[-1].year - syntethic['LAST_PURCHASE'].iloc[1].year
    syntethic['LAST_PURCHASE'] = syntethic['LAST_PURCHASE'] + pd.DateOffset(years=diff)
    return syntethic

def generateNegativeFeedback(data, size):
    print("GENERATING negative feedback")
    material_product_dict = dict(zip(data['PRODUCT_ID'], data['MATERIAL']))
    data["RATING_TYPE"] = np.nan
    neg_data = data.copy()
    while size > len(neg_data):
        temp_data = generateSyntethic(neg_data, (size - len(neg_data)))
        temp_data["RATING_TYPE"] = 0
        print('- - MERGING')
        neg_data = neg_data.append(temp_data)
        neg_data = neg_data.drop_duplicates(subset=['CUSTOMER_ID', 'PRODUCT_ID'], keep=False)
        neg_data = pd.concat([neg_data, data, data]).drop_duplicates(subset=['CUSTOMER_ID', 'PRODUCT_ID'], keep=False)
        print('- - DUPPED SIZE: ' + str(len(neg_data) + len(temp_data))+ ' | CLEAN SIZE: ' + str(len(neg_data)))

    print("- FIXING material - product mapping")
    neg_data['MATERIAL'] = neg_data['PRODUCT_ID']
    neg_data['MATERIAL'] = neg_data['MATERIAL'].map(material_product_dict)    
    del material_product_dict
    neg_data.reset_index(inplace=True, drop=True)
    return neg_data.head(size)

def splitDataSets(data, number_of_chunks, mini_set, file_name):
    print("SPLITTING: ")
    for id, data in  enumerate(np.array_split(data, number_of_chunks)):
        writeToCsv(data, mini_set, file_name + '_{id}.csv'.format(id=id+1))

def writeToCsv(data, mini_set, file_name):
    print("UPLOADING " + mini_set + '/' + file_name)
    if not os.path.exists(writing_path + mini_set):
        os.makedirs(writing_path + mini_set)
    
    ixs = np.array_split(data.index, 100)
    for ix, subset in tqdm(enumerate(ixs), total=len(ixs)):
        if ix == 0:
            data.loc[subset].to_csv(writing_path + mini_set + "/" + file_name, mode='w', index=False)
        else:
            data.loc[subset].to_csv(writing_path + mini_set + "/" + file_name, header=None, mode='a', index=False)


if __name__ == '__main__':
    main()