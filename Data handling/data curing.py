# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:35:49 2020

@author: RaidenRabit
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

writing_path = (r'\\cs.aau.dk\Fileshares\IT703e20\\')
original_csv_path = 'data4project_global.csv'
ds1_csv_path = 'unique-noBlanks-noNegatives.csv'
ds2_csv_path = 'no0s-unique-noBlanks-noNegatives.csv'


removeNegatives = lambda n : n >= 0
remove0s = lambda n : n > 0

def main():
    createTheTwoCleanDataSets()
    
def createMCQs():
    print("READING ds1")
    ds1 = pd.read_csv(reading_path + ds1_csv_path)
    print("DROPPING columns")
    ds1 = ds1.drop(['DOC_NUMBER_TEXT', 'CALDAY', 'is_accessory', 'MATERIAL_TEXT', 'MATL_GROUP'], axis=1)
    print("GROUPING BY")
    ds1 = ds1.sort_values(by=['CUSTOMER_ID', 'MATERIAL'])
    ds1 = ds1.groupby(['CUSTOMER_ID','MATERIAL'])['QUANTITY'].sum().reset_index()
    ds1['PRODUCT_ID'] = pd.factorize(ds1['MATERIAL'])[0]
    ds1 = ds1[['CUSTOMER_ID', 'MATERIAL', 'PRODUCT_ID', 'QUANTITY']]
    print(ds1)
    writeToCsv(ds1, "./MCQ with 0s and INT PRODUCT_ID.csv")
    del ds1
    
    print("READING ds2")
    ds2 = pd.read_csv(reading_path + ds2_csv_path)
    print("DROPPING columns")
    ds2 = ds2.drop(['DOC_NUMBER_TEXT', 'CALDAY', 'is_accessory', 'MATERIAL_TEXT', 'MATL_GROUP'], axis=1)
    print("GROUPING BY")
    ds2 = ds2.sort_values(by=['CUSTOMER_ID', 'MATERIAL'])
    ds2 = ds2.groupby(['CUSTOMER_ID','MATERIAL'])['QUANTITY'].sum().reset_index()
    ds2['PRODUCT_ID'] = pd.factorize(ds2['MATERIAL'])[0]
    ds2 = ds2[['CUSTOMER_ID', 'MATERIAL', 'PRODUCT_ID', 'QUANTITY']]
    writeToCsv(ds2, "./MCQ NO 0s AND INT PRODUCT_ID.csv")
    
    
def createTheTwoCleanDataSets():
    print("READING all records")
    types = {'DOC_NUMBER_TEXT':'object', 'CALDAY':'object', 'MATERIAL':'object',
             'MATERIAL_TEXT':'object', 'MATL_GROUP':'object', 'QUANTITY':'float64', 
             'is_accessory':'bool', 'CUSTOMER_ID':'object'}
    
    all_records = pd.read_csv(reading_path + original_csv_path, dtype=types, decimal=",", error_bad_lines=False)
    
    print("REMOVING duplicate rows")
    all_records = removeDuplicateRows(all_records)
    print("REMOVING blank quantities")
    all_records = removeNaNs(all_records, "QUANTITY")
    
    print("REMOVING negative quantities")
    all_records = removeElements(all_records, removeNegatives, "QUANTITY")
    
    print("WRITING ds1")
    writeToCsv(all_records, ds1_csv_path)
    
    print("CREATING DS2")
    print("REMOVING '0' quantities")
    ds2 = removeElements(all_records, remove0s, "QUANTITY")
    del all_records
    print("WRITING ds2")
    writeToCsv(ds2, ds2_csv_path)
    
def removeDuplicateRows(data):
    return data.drop_duplicates(keep = 'first')

def removeElements(data, condition, columns):
    return data[condition(data[columns])]  
    
def removeNaNs(data, columns):
    data[columns].replace('', np.nan, inplace=True)
    data.dropna(subset=[columns], inplace=True)
    return data

def writeToCsv(data, file_path):
    print("UPLOADING")
    ixs = np.array_split(data.index, 100)
    for ix, subset in tqdm(enumerate(ixs), total=len(ixs)):
        if ix == 0:
            data.loc[subset].to_csv(writing_path + file_path, mode='w', index=False)
        else:
            data.loc[subset].to_csv(writing_path + file_path, header=None, mode='a', index=False)


if __name__ == '__main__':
    main()