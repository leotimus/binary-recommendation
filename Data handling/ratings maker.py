# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 09:07:05 2020

@author: RaidenRabit

"""

import pandas as pd
import numpy as np
from scipy import sparse
from scipy import io

path = (r'\\cs.aau.dk\Fileshares\IT703e20\CleanDatasets\no_0s\\')
ratings_csv_name = 'ratings_global.csv'
mcq_no_0s = 'MCQ_NO_0s.csv'
original_csv_path = 'data4project_global.csv'

ds1_csv_path = 'unique-noBlanks-noNegatives.csv'
ds2_csv_path = 'no0s-unique-noBlanks-noNegatives.csv'

def main():
    
    print("READING reading all records")
    all_records = pd.read_csv(path + ds2_csv_path)
    
    #all_records = pd.read_csv(path + original_csv_path)
    print("EXTRACTING unique customers")
    unique_customers = extractUniqueCustomerIds(all_records)
    print("unique customers count: " + str(len(unique_customers)))
    
    print("EXTRACTING unique products")
    unique_products = extractUniqueProducts(all_records)
    print("unique products count: " + str(len(unique_products)))
    
    del all_records #deleting the original data from the RAM for better performance
    
    print("READING MCQ no 0s")
    #ignoring bad lines because some of them have 3 values or . within the customer_id
    binary_global_df = pd.read_csv(path + mcq_no_0s).drop(['QUANTITY'], axis=1)
    print("customer - product pairs count: " + str(len(binary_global_df)))
    
    print("CREATING matrix")
    matrix = createBinaryMatrix(unique_customers, unique_products, binary_global_df)
    print(matrix.head(5))
    writeToCsv(matrix, path + './matrix.csv')
    
def extractUniqueProducts(data):
    #dropping unnecesary columns
    product_df = data.drop(['CUSTOMER_ID', 'DOC_NUMBER_TEXT', 'CALDAY', 'QUANTITY',
                            'is_accessory', 'MATERIAL_TEXT', 'MATL_GROUP'], axis=1)
    #sorting values for better processing
    product_df.sort_values("MATERIAL", inplace = True) 
    #removing duplicates while keeping only the 1st value
    product_df.drop_duplicates(subset ="MATERIAL", keep = 'first', inplace = True) 
    #returnung a list for better processing later on
    #T is for transposing the values. The list would be [[1,2,3,etc.]]
    return product_df.T.values.tolist()[0]

def extractUniqueCustomerIds(data):
    #dropping unnecesary columns
    customer_df = data.drop(['MATERIAL', 'DOC_NUMBER_TEXT', 'CALDAY', 'QUANTITY',
                            'is_accessory', 'MATERIAL_TEXT', 'MATL_GROUP'], axis=1)
    #sorting values for better processing
    customer_df.sort_values("CUSTOMER_ID", inplace = True) 
    #removing duplicates while keeping only the 1st value
    customer_df.drop_duplicates(subset ="CUSTOMER_ID", keep = 'first', inplace = True) 
    #returnung a list for better processing later on
    #T is for transposing the values. The list would be [[1,2,3,etc.]]
    return customer_df.T.values.tolist()[0]

def createBinaryMatrix(rows, columnss, values):
    #initializing dataframe with the list of products as columns
    indptr = np.array(rows, dtype=np.uint64)    # a is a python array('L') contain row index information
    indices = np.array(columnss, dtype=np.uint64)   # b is  a python array('L') contain column index information
    data = np.ones((len(indices),), dtype=np.uint64)
    test = sparse.coo_matrix((data,indices,indptr), shape=(len(indptr)-1, len(indices)-1), dtype=np.nan)
    writeToCsv(test, path + "./indexedMatrix.csv")
    print("ADDING transaction values to the matrix")
    print("Progress: ")
    
    # iterate over the transaction list to insert 1s where the product was purchased by the customer
    for index, row in values.iterrows():
         print(str(index) + "/" + str(len(values))) #printing the progress
         test.at[row['CUSTOMER_ID'], row['MATERIAL']] = 1
         
    return test


def writeToCsv(data, file_name):
    data.to_csv(path + file_name, index=False)

if __name__ == '__main__':
    main()