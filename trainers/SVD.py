import numpy as np
import pandas as pd
import math

CHUNK_SIZE = 100
NUMBER_OF_CHUNKS_TO_EAT = 100
EPOCHS = 40
USER_ID_COLUMN = "CUSTOMER_ID"
ITEM_ID_COLUMN = " MATERIAL"
RATING_COLUMN = " is_real"
LEARNING_RATE = 0.01
REGULARIZATION = 0.01

NUMBER_OF_FACTORS = 5

FILE_PATH = r'/run/user/1000/gvfs/smb-share:server=cs.aau.dk,share=fileshares/IT703e20/CleanDatasets/with_0s/binary_MC_with_0s_populated1000.csv'
VERBOSE = False
PRINT_EVERY = 500

def print_verbose(message):
    if VERBOSE:
        print(message)

def convert_ids(data_chunks):
    user_ids = {}
    item_ids = {}
    next_user_id = 0
    next_item_id = 0

    number_of_chunks_to_eat = NUMBER_OF_CHUNKS_TO_EAT

    for chunk in data_chunks:
        for index, row in chunk.iterrows():
            if not (row[USER_ID_COLUMN] in user_ids):
                user_ids[row[USER_ID_COLUMN]] = next_user_id
                next_user_id += 1
            
            if not (row[ITEM_ID_COLUMN] in item_ids):
                item_ids[row[ITEM_ID_COLUMN]] = next_item_id
                next_item_id += 1


            if index%PRINT_EVERY == 0:
                print_verbose("digesting... at index: {} next_item_id is {}".format(index, next_item_id))
        
        if number_of_chunks_to_eat <= 0:
            break
        else:
            number_of_chunks_to_eat -= 1

    return user_ids, item_ids, next_user_id-1, next_item_id-1

def  fit_model(data_chunks, user_matrix, item_matrix, user_ids, item_ids):
    number_of_chunks_to_eat = NUMBER_OF_CHUNKS_TO_EAT

    for chunk in data_chunks:
        for index, row in chunk.iterrows():
            
            user = user_ids[row[USER_ID_COLUMN]]
            item = item_ids[row[ITEM_ID_COLUMN]]

            item_vector = item_matrix[item, :]
            user_vector = user_matrix[user, :]

            error = row[RATING_COLUMN] - np.dot(item_vector, user_vector)
            item_vector = item_vector + LEARNING_RATE * (error * user_vector - REGULARIZATION * item_vector)
            user_vector = user_vector + LEARNING_RATE * (error * item_vector - REGULARIZATION * user_vector)

            for n in range(0, NUMBER_OF_FACTORS):
                item_matrix[item, n] = item_vector[n]
                user_matrix[user, n] = user_vector[n]


            if index%PRINT_EVERY == 0:
                print_verbose("training... at index: {} error is {}".format(index, error))
        
        if number_of_chunks_to_eat <= 0:
            break
        else:
            number_of_chunks_to_eat -= 1

def  mean_absolute_error(data_chunks, user_matrix, item_matrix, user_ids, item_ids):
    number_of_chunks_to_eat = NUMBER_OF_CHUNKS_TO_EAT
    accumulator = 0
    count = 0

    for chunk in data_chunks:
        for index, row in chunk.iterrows():
            
            user = user_ids[row[USER_ID_COLUMN]]
            item = item_ids[row[ITEM_ID_COLUMN]]

            item_vector = item_matrix[item, :]
            user_vector = user_matrix[user, :]

            error = row[RATING_COLUMN] - np.dot(item_vector, user_vector)
            
            accumulator += np.absolute(error)
            count += 1

            if index%PRINT_EVERY == 0:
                print_verbose("calculating mean error... at index: {}".format(index))

        if number_of_chunks_to_eat <= 0:
            break
        else:
            number_of_chunks_to_eat -= 1

    return accumulator / count

class rating_prediction:
    index = 0
    prediction = 0
    
    def __init__(self, index, prediction):
        self.index = index
        self.prediction = prediction

    def __str__(self):
        return f"({self.prediction} @ {self.index})"

    def __repr__(self):
        return f"({self.prediction} @ {self.index})"

"""
    k: number of recommendations to make
"""
def recommend(user_vector, item_matrix, k):
    result = [rating_prediction(None, -math.inf) for _ in range(0, k)]

    minimum_value = result[0]

    for i, item_vector in enumerate(item_matrix):
        predicted = np.dot(user_vector, item_vector)
        if (predicted > minimum_value.prediction):
            result.remove(minimum_value)
            result.append(rating_prediction(i, predicted))
            minimum_value = min(result, key=lambda x: x.prediction)
    
    return result
    

if __name__ == "__main__":
    data_chunks = pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE)

    print("Digesting....\n----------------")
    user_ids, item_ids, uid_max, iid_max = convert_ids(data_chunks)
    
    user_matrix = np.random.random((uid_max + 1, NUMBER_OF_FACTORS))
    item_matrix = np.random.random((iid_max + 1, NUMBER_OF_FACTORS))

    print("Training:")

    for i in range(0,  EPOCHS):
        data_chunks = pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE)
        fit_model(data_chunks, user_matrix, item_matrix, user_ids, item_ids)

        data_chunks = pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE)

        err = mean_absolute_error(data_chunks, user_matrix, item_matrix, user_ids, item_ids)
        print (f"::::EPOCH {i}::::      Error: {err}")
    
    recommendations = recommend(user_matrix[4], item_matrix, 10)
    print (recommendations)
    