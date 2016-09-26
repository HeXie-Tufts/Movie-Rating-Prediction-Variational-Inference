import numpy as np
import math
from LoadData import get_data
import sys
import os


def load_split_data(data_size, test_p):
    # Load data and split into train set, test set randomly.
    # data_size is either "100k", "1m", "10m" or "20m".
    # test_p is a float between 0 - 1 indicating the portion of data hold out as test set
    print("split data randomly")
    # Load ratings, data is already permuted in get_data
    ratings = get_data(data_size)
    nb_users = int(np.max(ratings[:, 0]))
    nb_movies = int(np.max(ratings[:, 1]))
    # split test/train set
    test_size = int(len(ratings) * test_p)
    test_ratings = ratings[:test_size]
    train_ratings = ratings[test_size:]
    # train_ratings is sorted by user index
    train_ratings = train_ratings[train_ratings[:, 0].argsort()]
    # save test and train data in case more training is needed on this split
    np.save("Data/" + data_size + "_" + str(int(test_p * 100))+ "percent_test.npy", test_ratings)
    np.save("Data/" + data_size + "_" + str(int(test_p * 100))+ "percent_train.npy", train_ratings)
    # test_ratings and train_ratings are numpy array of user id | item id | rating
    return test_ratings, train_ratings, nb_users, nb_movies, len(train_ratings)


def load_split_data_even(data_size, test_p):
    # Load and split data by withheld a fixed number of samples from each user
    # For MovieLens dataset, make sure test_p not larger than 0.1
    print("Split data by hold out fixed number of rating from each user.")
    if test_p > 0.1:
        print("Test sample percentage could not be larger than 10%!")
        sys.exit()
    ratings = get_data(data_size)
    nb_users = int(np.max(ratings[:, 0]))
    nb_movies = int(np.max(ratings[:, 1]))
    ave_nb_rating = len(ratings)/nb_users
    nb_test = int(test_p * ave_nb_rating)
    # split test/train set
    ratings = ratings[ratings[:, 0].argsort()]
    test_ratings = np.array([[0,0,0]])
    train_ratings = np.array([[0,0,0]])
    i = 0
    j = 0
    while i < len(ratings):
        while j < len(ratings) and ratings[i][0] == ratings[j][0]:
            j = j + 1
        test_ratings = np.append(test_ratings, ratings[i:i + nb_test], axis = 0)
        train_ratings = np.append(train_ratings, ratings[i + nb_test:j], axis = 0)
        i = j
    test_ratings = test_ratings[1:]
    train_ratings = train_ratings[1:]
    # train_ratings is sorted by user index
    # train_ratings = train_ratings[train_ratings[:, 0].argsort()]
    # save test and train data in case needing to continue training
    # np.save("Data/" + data_size + "_" + str(int(test_p * 100))+ "percent_test.npy", test_ratings)
    # np.save("Data/" + data_size + "_" + str(int(test_p * 100))+ "percent_train.npy", train_ratings)
    # test_ratings and train_ratings are lists of user id | item id | rating
    return test_ratings, train_ratings, nb_users, nb_movies, len(train_ratings)


def predictRMSE(test_ratings, U, V):
    # predict movie ratings on test set and calculate RMSE
    RMSE = 0
    for rating in test_ratings:
        RMSE += np.square(rating[2] - np.dot(U[int(rating[0] - 1), :], V[int(rating[1] - 1), :]))
    RMSE = math.sqrt(RMSE / len(test_ratings))
    return RMSE


def predictMAE(test_ratings, U, V):
    # predict movie ratings on test set and calculate MAE
    MAE = 0
    for rating in test_ratings:
        MAE += abs(rating[2] - np.dot(U[int(rating[0] - 1), :], V[int(rating[1] - 1), :]))
    MAE /= len(test_ratings)
    return MAE

def trace(A, B):
    # Calculate the trace of the product of two symmetric matrices A and B
    return np.sum(A * B)


def train(data_size = "100k", nb_epoch = 10, test_p = 0.1, rank = 10, split = "random"):
    # Training of the variational inference matrix factorization model
    # Load data and split
    if split == "random":
        test_ratings, train_ratings, nb_users, nb_movies, nb_ratings = load_split_data(data_size, test_p)
    elif split == "byuser":
        test_ratings, train_ratings, nb_users, nb_movies, nb_ratings = load_split_data_even(data_size, test_p)
    # Initialize model parameters, the variances
    # The variances of V matrix, rhos, are held constant
    # The variances of U matrix, sigma square, are placed on the diagonal of a diagonal matrix, sigmasq_matrix
    tausq = 1
    sigmasq_matrix = np.identity(rank)
    # Initialize U and V randomly from N(0, 1), normal distribution with mean 0 and variance 1
    U = np.random.randn(nb_users, rank)
    V = np.random.randn(nb_movies, rank)

    Psi_list = np.zeros((nb_movies, rank, rank))
    np.copyto(Psi_list, [np.identity(rank) * rank])
    # Keep track of test set RMSE and train set RMSE
    RMSE_list = [0] * nb_epoch
    RMSE_train_list = [0] * nb_epoch
    with open("Result/" + data_size + "_" + str(rank) + "_" + \
                       str(nb_epoch) + "_Error.txt", "a") as myfile:
        myfile.write("\n" + split)
    # Start training
    for k in range(nb_epoch):
        print("nb_epoch: ", k)
        S_list =[np.identity(rank) * rank] * nb_movies
        t_list = [np.zeros((rank, ))] * nb_movies
        new_sigmasq = np.zeros((rank, rank))
        new_tausq = 0
        index_1 = 0
        index_2 = 0
        # update Q(u)
        for i in range(nb_users):
            # Update Phi_i and u_i
            Phi = sigmasq_matrix
            u_mean = np.zeros((rank, ))
            while index_1 < nb_ratings and train_ratings[index_1][0] == i + 1:
                j = int(train_ratings[index_1][1] - 1)
                rating = train_ratings[index_1][2]
                Phi = Phi + (1/tausq) * (Psi_list[j] + np.outer(V[j, :], V[j, :]))
                u_mean = u_mean + (1/tausq) * rating * V[j, :]
                index_1 = index_1 + 1
            Phi = np.linalg.inv(Phi)
            u_mean = np.dot(Phi, u_mean)
            U[i, :] = u_mean
            # Update S_j and t_j, add value to new_tausq for tausq update
            while index_2 < nb_ratings and train_ratings[index_2][0] == i + 1:
                j = int(train_ratings[index_2][1] - 1)
                rating = train_ratings[index_2][2]
                S_list[j] = S_list[j] + (1/tausq) * (Phi + np.outer(u_mean, u_mean))
                t_list[j] = t_list[j] + (1/tausq) * rating * u_mean
                new_tausq = new_tausq + rating**2 - 2 * rating * np.dot(U[i, :], V[j, :])\
                                    + trace((Phi + np.outer(u_mean, u_mean)), (Psi_list[j] + np.outer(V[j, :], V[j, :])))
                index_2 = index_2 + 1

            # Add value from Phi and u_mean to new_sigmasq
            for m in range(rank):
                new_sigmasq[m, m] = new_sigmasq[m, m] + Phi[m, m] + np.square(u_mean[m])

        # Update Psi_j and v_j
        for j in range(nb_movies):
            Psi_list[j] = np.linalg.inv(S_list[j])
            V[j, :] = np.dot(Psi_list[j], t_list[j])

        # Update the variances, update sigmasq
        sigmasq_matrix = 1.0/(nb_users -1) * new_sigmasq
        # Update tausq
        tausq = 1.0/(nb_ratings - 1) * new_tausq
        RMSE_list[k] = predictRMSE(test_ratings, U, V)
        RMSE_train_list[k] = predictRMSE(train_ratings, U, V)
        with open("Result/" + data_size + "_" + str(rank) + "_" + \
                        str(nb_epoch) + "_Error.txt", "a") as myfile:
            myfile.write("\n" + str(k+1) + "-test-" + str(RMSE_list[k]) + "-train-" + str(RMSE_train_list[k]))
    # Save the last U and V, sigma and tau
    np.save("Result/" + data_size + "_" + str(rank) + "_" + str(nb_epoch) + "_U.npy", U)
    np.save("Result/" + data_size + "_" + str(rank) + "_" + str(nb_epoch) + "_V.npy", V)
    np.save("Result/" + data_size + "_" + str(rank) + "_" + str(nb_epoch) + "_Psi.npy", Psi_list)
    np.save("Result/" + data_size + "_" + str(rank) + "_" + str(nb_epoch) + "_sigma.npy", sigmasq_matrix)
    with open("Result/" + data_size + "_" + str(rank) + "_" + str(nb_epoch) + "_tausq.txt", 'w') as myfile:
        myfile.write(str(tausq))

    print("done")
    return nb_epoch, RMSE_list


if __name__ == '__main__':
    train("10m", 10, 0.1, 5)
