n = 5 # Number of factors

get_error(r_ui, q_i, p_u)
    return r_ui - dotp(q_i, p_u)

train_model(R, Q, P)
    for _ in range 0 .. number_of_training_iterations
        for u in range 0.. height(R) :
            for i in range 0 .. length(R):
                error = get_error(R[u][i], Q[i], P[u])
                
                Q[i] = Q[i] + learning_rate * error * Q[i]
                P[i] = P[i] + learning_rate * error * P[i]

    return Q, P

# predict rating for item i and user u
dotp(Q[i], P[u])