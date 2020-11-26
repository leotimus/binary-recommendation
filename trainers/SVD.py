import numpy as np
import pandas as pd
import math
import trainers.topKMetrics as topk
import smbclient as smbc
from src.AAUfilename import getAAUfilename
from getpass import getpass

EPOCHS = 20
LEARNING_RATE = 0.01
REGULARIZATION = 0.01
NUMBER_OF_FACTORS = 40

EPOCH_ERROR_CALCULATION_FREQUENCY = 5
VERBOSE = True
PRINT_EVERY = 25000

GRUNDFOS = True

# The data is expected in chunks, either in separate files or in a single
# file that pandas will then split to the size specified. 
# NB: The NUMBER_OF_CHUNKS_TO_EAT is for training. Another chunk after that
# should be reserved for testing.
if not GRUNDFOS:
    FILE_PATH = r"data/ml-100k/all.csv"
    CHUNK_MODE = "single-file" # Either "single-file" or "multi-file"
    CHUNK_SIZE = 2E4
    NUMBER_OF_CHUNKS_TO_EAT = 4
    USER_ID_COLUMN = "user_id"
    ITEM_ID_COLUMN = "item_id"
    RATING_COLUMN = "rating"

    TRANSACTION_COUNT_COLUMN = TRANSACTION_COUNT_SCALE = QUANTITY_SUM_COLUMN = QUANTITY_SUM_SCALE = None
else:
    # Grundfos Data columns: CUSTOMER_ID,PRODUCT_ID,MATERIAL,TRANSACTION_COUNT,QUANTITY_SUM,FIRST_PURCHASE,LAST_PURCHASE,TIME_DIFF_DAYS
    FILE_PATH = r"(NEW)CleanDatasets/SVD/2m(OG)/ds2_OG(2m)_timeDistributed_{0}.csv"
    CHUNK_MODE = "multi-file" # Either "single-file" or "multi-file"}.csv
    CHUNK_SIZE = None
    NUMBER_OF_CHUNKS_TO_EAT = 4
    USER_ID_COLUMN = "CUSTOMER_ID"
    ITEM_ID_COLUMN = "PRODUCT_ID"
    RATING_COLUMN = None

    TRANSACTION_COUNT_COLUMN = "TRANSACTION_COUNT"
    TRANSACTION_COUNT_SCALE = 0.6
    TRANSACTION_COUNT_QUINTILES = (1, 2, 4)

    QUANTITY_SUM_COLUMN = "QUANTITY_SUM"
    QUANTITY_SUM_SCALE = 0.4
    QUANTITY_SUM_QUINTILES = (1, 1, 2)

def print_verbose(message):
    if VERBOSE:
        print(message)

def digest(data_chunks):
    user_ids = {}
    item_ids = {}
    next_user_id = 0
    next_item_id = 0

    total_so_far = 0
    average_so_far = 0

    number_of_chunks_to_eat = NUMBER_OF_CHUNKS_TO_EAT

    for chunk in data_chunks:
        next_user_id, next_item_id = convert_ids(chunk, user_ids, item_ids, next_user_id, next_item_id)

        total_so_far, average_so_far = calculate_average(chunk, total_so_far, average_so_far)
        
        number_of_chunks_to_eat -= 1
        if number_of_chunks_to_eat <= 0:
            break

    return user_ids, item_ids, next_user_id-1, next_item_id-1, average_so_far

def convert_ids(chunk, user_ids, item_ids, next_user_id, next_item_id):
    for index, row in chunk.iterrows():
        if not (row[USER_ID_COLUMN] in user_ids):
            user_ids[row[USER_ID_COLUMN]] = next_user_id
            next_user_id += 1
        
        if not (row[ITEM_ID_COLUMN] in item_ids):
            item_ids[row[ITEM_ID_COLUMN]] = next_item_id
            next_item_id += 1

        if index%PRINT_EVERY == 0:
            print_verbose("digesting... at index: {} next_item_id is {}".format(index, next_item_id))
        
    return next_user_id, next_item_id

def calculate_average(chunk, total_so_far, average_so_far):
    chunk_total = chunk.shape[0]
    chunk_accumulator = 0

    for index, row in chunk.iterrows():
        
        rating = get_rating(row)

        chunk_accumulator += rating

        if index%PRINT_EVERY == 0:
            print_verbose(f"calculating global average... at index: {index}")
    
    chunk_average = chunk_accumulator / chunk_total

    total_so_far += chunk_total
    average_so_far = ((chunk_total/total_so_far) * chunk_average) + (((total_so_far-chunk_total)/total_so_far) *average_so_far)

    return total_so_far, average_so_far

class svd_prediction_doer:
    def __init__(self, user_ids, item_ids, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias):
        self.user_matrix = user_matrix
        self.item_matrix = item_matrix
        self.user_bias_vector = user_bias_vector
        self.item_bias_vector = item_bias_vector
        self.global_bias = global_bias
        self.user_ids = user_ids
        self.item_ids = item_ids

    def predict(self, weird_array):
        user = self.user_ids[weird_array[0][0]]
        item = self.item_ids[weird_array[1][0]]
        return predict(user, item, self.user_matrix, self.item_matrix, self.user_bias_vector, self.item_bias_vector, self.global_bias)
         
def predict(user, item, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias):
    item_vector = item_matrix[item, :]
    user_vector = user_matrix[user, :]
    item_bias = item_bias_vector[item]
    user_bias = user_bias_vector[user]

    return user_bias + item_bias + global_bias + np.dot(item_vector, user_vector)

def  fit_model(data_chunks, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids):
    number_of_chunks_to_eat = NUMBER_OF_CHUNKS_TO_EAT

    for chunk in data_chunks:
        for index, row in chunk.iterrows():
            
            user = user_ids[row[USER_ID_COLUMN]]
            item = item_ids[row[ITEM_ID_COLUMN]]

            item_vector = item_matrix[item, :]
            user_vector = user_matrix[user, :]

            error = get_rating(row) - predict(user, item, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias)

            item_vector = item_vector + LEARNING_RATE * (error * user_vector - REGULARIZATION * item_vector)
            user_vector = user_vector + LEARNING_RATE * (error * item_vector - REGULARIZATION * user_vector)
            user_bias_vector[user] = user_bias_vector[user] + LEARNING_RATE * (error * user_bias_vector[user] - REGULARIZATION * user_bias_vector[user])
            item_bias_vector[item] = item_bias_vector[item] + LEARNING_RATE * (error * item_bias_vector[item] - REGULARIZATION * item_bias_vector[item])

            for n in range(0, NUMBER_OF_FACTORS):
                item_matrix[item, n] = item_vector[n]
                user_matrix[user, n] = user_vector[n]

            if index%PRINT_EVERY == 0:
                print_verbose(f"training... at index: {index} error is {error}")

        number_of_chunks_to_eat -= 1
        if number_of_chunks_to_eat <= 0:
            break

def  mean_generic_error(generic, data_chunks, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids):
    number_of_chunks_to_eat = NUMBER_OF_CHUNKS_TO_EAT
    accumulator = 0
    count = 0

    for chunk in data_chunks:
        for index, row in chunk.iterrows():
            
            user = user_ids[row[USER_ID_COLUMN]]
            item = item_ids[row[ITEM_ID_COLUMN]]

            error = get_rating(row) - predict(user, item, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias)
            
            accumulator += generic(error)
            count += 1

            if index%PRINT_EVERY == 0:
                print_verbose("calculating mean error... at index: {}".format(index))

        number_of_chunks_to_eat -= 1
        if number_of_chunks_to_eat <= 0:
            break

    return accumulator / count

def mean_absolute_error(data_chunks, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids):
    return mean_generic_error(abs, data_chunks, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids)

def mean_square_error(data_chunks, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids):
    return mean_generic_error(lambda x: x**2, data_chunks, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids)

def get_rating(row):
    if RATING_COLUMN!=None:
        return row[RATING_COLUMN]
    else:
        return \
         TRANSACTION_COUNT_SCALE * place_in_quintile(row[TRANSACTION_COUNT_COLUMN], TRANSACTION_COUNT_QUINTILES) +\
         QUANTITY_SUM_SCALE * place_in_quintile(row[QUANTITY_SUM_COLUMN], QUANTITY_SUM_QUINTILES)

def place_in_quintile(value, quintiles):
    q1, median, q3 = quintiles
    if value > q3: return 4
    elif value > median: return 3
    elif value > q1: return 2
    else: return 1

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

class grundfos_network_drive_files:
    def __init__(self, file_path, credentials, columns):
        self.file_path = file_path
        self.columns = columns
        self.username, self.password = credentials
    
    def __iter__(self):
        self.next_chunk = 1
        return self

    def __next__(self):
        file_path = self.file_path.format(self.next_chunk)
        network_path = getAAUfilename(file_path)
        
        try:
            with smbc.open_file(network_path, mode="r", username=self.username, password=self.password) as f:
                chunk = pd.read_csv(f, usecols=self.columns)
        except FileNotFoundError:
            raise StopIteration

        self.next_chunk += 1
        return chunk

def read_csv(credentials):
    if GRUNDFOS:
        return grundfos_network_drive_files(FILE_PATH, credentials, [USER_ID_COLUMN, ITEM_ID_COLUMN, TRANSACTION_COUNT_COLUMN, QUANTITY_SUM_COLUMN])
    else:
        return pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE, usecols=[USER_ID_COLUMN, ITEM_ID_COLUMN, RATING_COLUMN])

def get_test_set(test_dataframe):
	result = set()
	for _, row in test_dataframe.iterrows():
		result.add((row[USER_ID_COLUMN], row[ITEM_ID_COLUMN]))
	return result
    
if __name__ == "__main__":
    credentials = (None, None)
    if GRUNDFOS:
        username = input("Username:")
        password = getpass()
        credentials = (username, password)

    data_chunks = read_csv(credentials)

    print("-"*16)
    print("Digesting....")
    user_ids, item_ids, uid_max, iid_max, global_bias = digest(data_chunks)
    number_of_users = uid_max + 1
    number_of_items = iid_max + 1

    data_chunks = read_csv(credentials)

    print(f"number of users: {number_of_users}")
    print(f"number of items: {number_of_items}")
    print(f"global bias: {global_bias}")
    
    user_matrix = np.random.random((number_of_users, NUMBER_OF_FACTORS))
    item_matrix = np.random.random((number_of_items, NUMBER_OF_FACTORS))
    user_bias_vector = np.zeros(number_of_users)
    item_bias_vector = np.zeros(number_of_items)

    print("-"*16)
    print("Training:")

    for i in range(0,  EPOCHS):
        data_chunks = read_csv(credentials)
        fit_model(data_chunks, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids)

        data_chunks = read_csv(credentials)

        if i%EPOCH_ERROR_CALCULATION_FREQUENCY==0:
            err = mean_square_error(data_chunks, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids)
            print (f"::::EPOCH {i:=3}::::      MSE: {err}")
        else:
            print (f"::::EPOCH {i:=3}::::")
    
    print("-"*16)
    print("Evaluating...")
    actual_user_ids = user_ids.keys() # The reader is asked to recall that user_ids is a dict that maps the actual ids to our own made-up sequential integer ids
    actual_item_ids = item_ids.keys()

    print("Reading test data.")
    data_chunks = read_csv(credentials)
    number_of_chunks_to_eat = NUMBER_OF_CHUNKS_TO_EAT
    # Be sure we are at the very last chunk
    for _ in data_chunks:
        number_of_chunks_to_eat -= 1
        if number_of_chunks_to_eat <= 0:
            break

    test_dataframe = next(data_chunks) # If you get a StopIteration exception on this line, you forgot to reserve a last chunk for testing.

    test_set = get_test_set(test_dataframe)

    print("Calculating top-k results")
    prediction_doer = svd_prediction_doer(user_ids, item_ids, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias)
    top_10_ratings = topk.topKRatings(10, prediction_doer, actual_user_ids, actual_item_ids, mtype="svd")
    results = topk.topKMetrics(top_10_ratings, test_set, actual_user_ids, actual_item_ids)
    print(results)