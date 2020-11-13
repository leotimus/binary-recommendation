import numpy as np
import pandas as pd

CHUNK_SIZE = 100
NUMBER_OF_CHUNKS_TO_EAT = 100
PRINT_EVERY = 500
EPOCHS = 10
USER_ID_COLUMN = "CUSTOMER_ID"
ITEM_ID_COLUMN = " MATERIAL"
RATING_COLUMN = " is_real"
LEARNING_RATE = 0.01
REGULARIZATION = 0.01

NUMBER_OF_FACTORS = 5

VERBOSE = False

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

            q_i = item_matrix[:, item]
            p_u = user_matrix[:, user]

            error = row[RATING_COLUMN] - np.dot(q_i, p_u)
            q_i = q_i + LEARNING_RATE * (error * p_u - REGULARIZATION * q_i)
            p_u = p_u + LEARNING_RATE * (error * q_i - REGULARIZATION * p_u)


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

            q_i = item_matrix[:, item]
            p_u = user_matrix[:, user]

            error = row[RATING_COLUMN] - np.dot(q_i, p_u)
            
            accumulator += np.absolute(error)
            count += 1

            if index%PRINT_EVERY == 0:
                print_verbose("calculating mean error... at index: {}".format(index))

        if number_of_chunks_to_eat <= 0:
            break
        else:
            number_of_chunks_to_eat -= 1

    return accumulator / count

        
#n = 5 # Number of factors
#
#get_error(r_ui, q_i, p_u)
#    return r_ui - dotp(q_i, p_u)
#
#train_model(R, Q, P)
#    for _ in range 0 .. number_of_training_iterations
#        for u in range 0.. height(R) :
#            for i in range 0 .. length(R):
#                error = get_error(R[u][i], Q[i], P[u])
#                
#                Q[i] = Q[i] + learning_rate * error * Q[i]
#                P[i] = P[i] + learning_rate * error * P[i]
#
#    return Q, P
#
## predict rating for item i and user u
#dotp(Q[i], P[u])



if __name__ == "__main__":
    file = r'/run/user/1000/gvfs/smb-share:server=cs.aau.dk,share=fileshares/IT703e20/CleanDatasets/with_0s/binary_MC_with_0s_populated1000.csv'
    data_chunks = pd.read_csv(file, chunksize=CHUNK_SIZE)

    print("Digesting....\n----------------")
    user_ids, item_ids, uid_max, iid_max = convert_ids(data_chunks)
    
    user_matrix = np.random.random((NUMBER_OF_FACTORS, uid_max + 1))
    item_matrix = np.random.random((NUMBER_OF_FACTORS, iid_max + 1))

    for i in range(0,  EPOCHS):
        print (f"::::EPOCH {i}::::")
        data_chunks = pd.read_csv(file, chunksize=CHUNK_SIZE)
        fit_model(data_chunks, user_matrix, item_matrix, user_ids, item_ids)

        data_chunks = pd.read_csv(file, chunksize=CHUNK_SIZE)

        err = mean_absolute_error(data_chunks, user_matrix, item_matrix, user_ids, item_ids)
        print(f"Error: {err}")