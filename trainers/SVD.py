import numpy as np
import pandas as pd
import math
import trainers.topKMetrics as topk
import smbclient as smbc
import tensorflow as tf
import tensorflow_recommenders as tfrs
import git
from datetime import datetime
from src.AAUfilename import getAAUfilename
from getpass import getpass
from src.benchmarkLogger import benchThread

EPOCHS = 10
LEARNING_RATE = 0.01
EMBEDDING_REGULARIZATION = 0.1
BIAS_REGULARIZATION = 0.01
NUMBER_OF_EMBEDDINGS = 50

TOPK_BATCH_SIZE = 5000
EPOCH_ERROR_CALCULATION_FREQUENCY = 100
VERBOSE = True
PRINT_EVERY = 1351 # Get more random-looking numbers

GRUNDFOS = True
EVALUATE = False

BENCHMARK_FILE_NAME = "svd-benchmark.csv"

# The data is expected in chunks, either in separate files or in a single
# file that pandas will then split to the size specified. 
# NB: The NUMBER_OF_CHUNKS_TO_EAT is for training. Another chunk after that
# should be reserved for testing.
if not GRUNDFOS:
    FILE_PATH = r"data/ml-100k/all.csv"
    CHUNK_SIZE = 20000
    NUMBER_OF_CHUNKS_TO_EAT = 5
    USER_ID_COLUMN = "user_id"
    ITEM_ID_COLUMN = "item_id"
    RATING_COLUMN = "rating"

    TRANSACTION_COUNT_COLUMN = TRANSACTION_COUNT_SCALE = QUANTITY_SUM_COLUMN = QUANTITY_SUM_SCALE = None
else:
    # Grundfos Data columns: CUSTOMER_ID,PRODUCT_ID,MATERIAL,TRANSACTION_COUNT,QUANTITY_SUM,FIRST_PURCHASE,LAST_PURCHASE,TIME_DIFF_DAYS
    FILE_PATH = r"(NEW)CleanDatasets/NCF/100k/ds2_100k_timeDistributed_{0}.csv"
    #FILE_PATH = r"(NEW)CleanDatasets/NCF/500k/ds2_500k_timeDistributed_{0}.csv"
    #FILE_PATH = r"(NEW)CleanDatasets/NCF/1m/ds2_1m_timeDistributed_{0}.csv"
    #FILE_PATH = r"(NEW)CleanDatasets/NCF/2m(OG)/ds2_OG(2m)_timeDistributed_{0}.csv"
    NUMBER_OF_FILES = 5
    NUMBER_OF_CHUNKS_TO_EAT = 5
    USER_ID_COLUMN = "CUSTOMER_ID"
    ITEM_ID_COLUMN = "PRODUCT_ID"
    RATING_COLUMN = "RATING_TYPE"
    #RATING_COLUMN = None

    TRANSACTION_COUNT_COLUMN = "TRANSACTION_COUNT"
    TRANSACTION_COUNT_SCALE = 0.5
    QUANTITY_SUM_QUINTILES = (1, 1, 2)

    QUANTITY_SUM_COLUMN = "QUANTITY_SUM"
    QUANTITY_SUM_SCALE = 0.5
    TRANSACTION_COUNT_QUINTILES = (1, 2, 4)

verbose_print_count = 0
verbose_print_indicators = r"-\|/-\|/"
def print_verbose(message):
    global verbose_print_count
    if VERBOSE:
        indicator = verbose_print_indicators[verbose_print_count%len(verbose_print_indicators)]
        indicator = f"{indicator}  "
        verbose_print_count += 1
        print(indicator + message, end="\r")

def clear_verbose_print():
    global verbose_print_count
    if VERBOSE:
        verbose_print_count = 0
        print("\r" + " "*80, end="\r")

def get_config():
    repo = git.Repo(search_parent_directories=True)
    git_commit_sha = repo.head.object.hexsha
    result = {
        "epochs":EPOCHS,
        "learning_rate":LEARNING_RATE,
        "regularization":EMBEDDING_REGULARIZATION,
        "number_of_embeddings":NUMBER_OF_EMBEDDINGS,
        "file_path":FILE_PATH,
        "git_commit_sha": git_commit_sha
    }
    if GRUNDFOS:
        grundfos_specific = {
            "rating_column":RATING_COLUMN,
            "transaction_count_column":TRANSACTION_COUNT_COLUMN,
            "transaction_count_scale":TRANSACTION_COUNT_SCALE,
            "transaction_count_quintiles":TRANSACTION_COUNT_QUINTILES,
            "quantity_sum_column":QUANTITY_SUM_COLUMN,
            "quantity_sum_scale":QUANTITY_SUM_SCALE,
            "quantity_sum_quintiles":QUANTITY_SUM_QUINTILES
        }
        return dict(result, **grundfos_specific)
    else:
        return result

def digest(dataset):
    user_ids = {}
    item_ids = {}
    next_user_id = 0
    next_item_id = 0

    total_so_far = 0
    average_so_far = 0

    number_of_chunks_to_eat = NUMBER_OF_CHUNKS_TO_EAT

    for i, chunk in enumerate(dataset):
        next_user_id, next_item_id = convert_ids(chunk, i, user_ids, item_ids, next_user_id, next_item_id)

        total_so_far, average_so_far = calculate_average(chunk, i, total_so_far, average_so_far)
        
        number_of_chunks_to_eat -= 1
        if number_of_chunks_to_eat <= 0:
            break

    return user_ids, item_ids, next_user_id-1, next_item_id-1, average_so_far

def convert_ids(chunk, chunk_number, user_ids, item_ids, next_user_id, next_item_id):
    for index, row in chunk.iterrows():
        if not (row[USER_ID_COLUMN] in user_ids):
            user_ids[row[USER_ID_COLUMN]] = next_user_id
            next_user_id += 1
        
        if not (row[ITEM_ID_COLUMN] in item_ids):
            item_ids[row[ITEM_ID_COLUMN]] = next_item_id
            next_item_id += 1

        if index%PRINT_EVERY == 0:
            print_verbose(f"digesting... index: {index} chunk: {chunk_number}")

    return next_user_id, next_item_id

def calculate_average(chunk, chunk_number, total_so_far, average_so_far):
    chunk_total = chunk.shape[0]
    chunk_accumulator = 0

    for index, row in chunk.iterrows():
        
        rating = get_rating(row)

        chunk_accumulator += rating

        if index%PRINT_EVERY == 0:
            print_verbose(f"calculating global average... index: {index} chunk: {chunk_number}")
    
    clear_verbose_print()

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

def  fit_model(dataset, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids):
    number_of_chunks_to_eat = NUMBER_OF_CHUNKS_TO_EAT

    for i, chunk in enumerate(dataset):
        for index, row in chunk.iterrows():
            
            user = user_ids[row[USER_ID_COLUMN]]
            item = item_ids[row[ITEM_ID_COLUMN]]

            item_vector = item_matrix[item, :]
            user_vector = user_matrix[user, :]

            error = get_rating(row) - predict(user, item, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias)

            if error > 200:
                print(f"Warning! Error is out of whack! Value: {error}")

            item_vector = item_vector + LEARNING_RATE * (error * user_vector -  EMBEDDING_REGULARIZATION * item_vector)
            user_vector = user_vector + LEARNING_RATE * (error * item_vector -  EMBEDDING_REGULARIZATION * user_vector)
            user_bias_vector[user] = user_bias_vector[user] + LEARNING_RATE * (error * user_bias_vector[user] - BIAS_REGULARIZATION * user_bias_vector[user])
            item_bias_vector[item] = item_bias_vector[item] + LEARNING_RATE * (error * item_bias_vector[item] - BIAS_REGULARIZATION * item_bias_vector[item])

            for n in range(0, NUMBER_OF_EMBEDDINGS):
                item_matrix[item, n] = item_vector[n]
                user_matrix[user, n] = user_vector[n]

            if index%PRINT_EVERY == 0:
                print_verbose(f"training... index: {index} chunk: {i}")

        clear_verbose_print()

        number_of_chunks_to_eat -= 1
        if number_of_chunks_to_eat <= 0:
            break

def  mean_generic_error(generic, dataset, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids):
    number_of_chunks_to_eat = NUMBER_OF_CHUNKS_TO_EAT
    accumulator = 0
    count = 0

    for i, chunk in enumerate(dataset):
        for index, row in chunk.iterrows():
            
            user = user_ids[row[USER_ID_COLUMN]]
            item = item_ids[row[ITEM_ID_COLUMN]]

            error = get_rating(row) - predict(user, item, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias)
            
            accumulator += generic(error)
            count += 1

            if index%PRINT_EVERY == 0:
                print_verbose(f"calculating mean error... index: {index} chunk: {i}")
        
        clear_verbose_print()

        number_of_chunks_to_eat -= 1
        if number_of_chunks_to_eat <= 0:
            break

    return accumulator / count

def mean_absolute_error(dataset, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids):
    return mean_generic_error(abs, dataset, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids)

def mean_square_error(dataset, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids):
    return mean_generic_error(lambda x: x**2, dataset, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids)

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

# TODO: These two classes share a lot of code that could probably be made a super-class but I'm not in the mood
class movielens_cross_validation:
    def __init__(self, file_path, number_of_chunks_to_eat, columns):
        self.file_path = file_path
        self.number_of_chunks_to_eat = number_of_chunks_to_eat
        self.columns = columns
        self.test_set_index = self.number_of_chunks_to_eat-1 # Start with the last chunk as the test-set
        self.chunks = pd.read_csv(FILE_PATH, usecols=columns)
        self.chunks = self.chunks.sample(frac=1).reset_index(drop=True)
        self.chunks = np.array_split(self.chunks, 5, axis=0)

    """ Makes the iterator return every chunk, reserving none for testing """
    def use_no_test_set(self):
        self.test_set_index = self.number_of_chunks_to_eat # This number is 1 higher than the index of the last chunk in self.chunks

    """ Changes the dataset to use the next chunk in line to be the test set. 
        Returns False if every chunk has already been used as test-set. """
    def next_cross_validation_distribution(self):
        self.test_set_index -= 1
        
        if(self.test_set_index < 0):
            return False
        else:
            return True

    def get_test_set(self):
        if self.test_set_index == self.number_of_chunks_to_eat:
            raise Exception("There is no test set because use_no_test_set was called. Call next_cross_validation_distribution to set the first chunk to be the test set again.")

        return self.chunks[self.test_set_index]

    def __iter__(self):
        self.next_chunk = 0
        self.chunks.__iter__()
        return self

    def __next__(self):
        if self.next_chunk == self.test_set_index:
            self.next_chunk += 1
            return self.__next__()
        
        if self.next_chunk >= len(self.chunks):
            raise StopIteration

        chunk = self.chunks[self.next_chunk]
        self.next_chunk += 1
        return chunk

class grundfos_network_drive_files:
    def __init__(self, file_path, number_of_files, credentials, columns):
        self.file_path = file_path
        self.number_of_files = number_of_files
        self.columns = columns
        self.username, self.password = credentials
        self.test_set_index = self.number_of_files-1 # Start by using the last file as the test-set

        self.files = []

        for i in range(1, number_of_files+1):
            file_path = self.file_path.format(i)
            network_path = getAAUfilename(file_path)

            with smbc.open_file(network_path, mode="r", username=self.username, password=self.password) as f:
                file = pd.read_csv(f, usecols=self.columns)
                self.files.append(file)

    """ Makes the iterator return every chunk, reserving none for testing """
    def use_no_test_set(self):
        self.test_set_index = self.number_of_files # This number is 1 higher than the index of the last file 

    """ Changes the dataset to use the next chunk in line to be the test set. 
        Returns False if every chunk has already been used as test-set. """
    def next_cross_validation_distribution(self):
        self.test_set_index -= 1
        
        if(self.test_set_index < 0):
            return False
        else:
            return True

    def get_test_set(self):
        if self.test_set_index == self.number_of_files:
            raise Exception("There is no test set because use_no_test_set was called. Call next_cross_validation_distribution to set the first chunk to be the test set again.")

        test_set = self.files[self.test_set_index]

        if RATING_COLUMN != None:
            test_set = test_set.query(f"{RATING_COLUMN}==1")

        return test_set

    def __iter__(self):
        self.next_file_index = 0
        return self

    def __next__(self):
        if self.next_file_index == self.test_set_index:
            self.next_file_index += 1
            return self.__next__()

        if self.next_file_index >= self.number_of_files:
            raise StopIteration
        
        chunk = self.files[self.next_file_index]

        self.next_file_index += 1
        return chunk

def get_test_set(test_dataframe):
	result = set()
	for _, row in test_dataframe.iterrows():
		result.add((row[USER_ID_COLUMN], row[ITEM_ID_COLUMN]))
	return result

def do_topk(user_matrix, item_matrix, test_set, user_ids, item_ids):
    actual_user_ids = user_ids.keys() # The reader is asked to recall that user_ids is a dict that maps the actual ids to our own made-up sequential integer ids
    actual_item_ids = item_ids.keys()

    item_tensor = tf.convert_to_tensor(item_matrix, dtype=np.float32)

    topk_predicter = tfrs.layers.factorized_top_k.BruteForce(k= 10)
    topk_predicter.index(item_tensor)
    
    predictions = []
    
    user_id = 0
    for batch in np.array_split(user_matrix, TOPK_BATCH_SIZE):
        tensor_batch = tf.convert_to_tensor(batch, dtype=np.float32)
        raw_predictions = topk_predicter(tensor_batch)
        for u in range(len(batch)):
            predictions.append((user_id, [(raw_predictions[0][u][j], raw_predictions[1][u][j].numpy()) for j in range(len(raw_predictions[0][u]))]))
            user_id+=1
    
    return topk.topKMetrics(predictions, test_set, actual_user_ids, actual_item_ids)

def train_and_evaluate(dataset, user_ids, item_ids, uid_max, iid_max, global_bias):
    number_of_users = uid_max + 1
    number_of_items = iid_max + 1

    print(f"number of users: {number_of_users}")
    print(f"number of items: {number_of_items}")
    print(f"global bias: {global_bias}", flush=True)
    
    user_matrix = np.random.random((number_of_users, NUMBER_OF_EMBEDDINGS)) * (1/NUMBER_OF_EMBEDDINGS)
    item_matrix = np.random.random((number_of_items, NUMBER_OF_EMBEDDINGS)) * (1/NUMBER_OF_EMBEDDINGS)
    user_bias_vector = np.zeros(number_of_users)
    item_bias_vector = np.zeros(number_of_items)

    test_dataframe = dataset.get_test_set()

    print("-"*16)
    print("Training:", flush=True)

    for i in range(1,  EPOCHS+1):
        fit_model(dataset, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids)

        current_time = datetime.now().strftime("%H:%M:%S")

        if i%EPOCH_ERROR_CALCULATION_FREQUENCY==0:
            err = mean_square_error(dataset, user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids)
            print (f"::::EPOCH {i:=3}::::    {current_time}    MSE: {err}", flush=True)
            test_set_err = mean_square_error([test_dataframe], user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids)
            print(f"             And on the test set MSE: {test_set_err}")
        else:
            print (f"::::EPOCH {i:=3}::::    {current_time}", flush=True)

    if bmThread != None:
        bmThread.active = 0 #deactivate the thread, will exit on the next while loop cycle
        bmThread.join()  # wait for it to exit on its own, since its daemon as a precaution
    
    result = {}

    if EVALUATE:
        print("-"*16)
        print("Evaluating...", flush=True)
        test_set = get_test_set(test_dataframe) # Set of (user, item) pairs
        print("Calculating MSE on test set", flush=True)
        test_set_err = mean_square_error([test_dataframe], user_matrix, item_matrix, user_bias_vector, item_bias_vector, global_bias, user_ids, item_ids)

        print("Calculating top-k results", flush=True)

        result = do_topk(user_matrix, item_matrix, test_set, user_ids, item_ids)
        result["mse"] = test_set_err

        print(f"Using config: {get_config()}")
        print(f"Got results: {result}")
        print("-"*16)

    return result

def get_data(): 
    credentials = (None, None)
    if GRUNDFOS:
        username = input("Username:")
        password = getpass()
        credentials = (username, password)

    if RATING_COLUMN!=None:
        columns = [USER_ID_COLUMN, ITEM_ID_COLUMN, RATING_COLUMN]
    else:
        columns = [USER_ID_COLUMN, ITEM_ID_COLUMN, TRANSACTION_COUNT_COLUMN, QUANTITY_SUM_COLUMN]

    if GRUNDFOS:
        dataset = grundfos_network_drive_files(FILE_PATH, NUMBER_OF_FILES, credentials, columns)
    else:
        dataset = movielens_cross_validation(FILE_PATH, NUMBER_OF_CHUNKS_TO_EAT, columns)
    
    return dataset

if __name__ == "__main__":
    print(get_config())
    
    print("Loading data...")
    dataset = get_data()
    dataset.use_no_test_set() # Don't hold any chunk back as the test set, so that every chunk can be digested.

    if BENCHMARK_FILE_NAME != None:
        bmThread = benchThread(1,1,BENCHMARK_FILE_NAME) #create the thread
        bmThread.start() #and start it
    else:
        bmThread = None

    print("-"*16)
    print("Digesting....", flush=True)
    user_ids, item_ids, uid_max, iid_max, global_bias = digest(dataset)

    dataset.next_cross_validation_distribution() # Use the first chunk as the test set
    
    results = []

    x_val=1
    while True:
        print("="*16)
        print(f"Cross validation {x_val} of 5")
        print("="*16)
        x_val+=1

        result = train_and_evaluate(dataset, user_ids, item_ids, uid_max, iid_max, global_bias)
        results.append(result)

        if not dataset.next_cross_validation_distribution():
            break
    
    result = topk.getAverage(results)

    print("="*16)
    print(f"Using config: {get_config()}")
    print(f"Got final results: {result}")
    print("::::ALL  DONE::::", flush=True)
