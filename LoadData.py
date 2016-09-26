# Read and load movie rating data into a numpy array named ratings. 
# Each rating is in [userID, movieID, rating] format. 
# Assume MovieLens 100k, 1m, 10m and 20m data set are saved in Data/ml-100k, Data/ml-1m, 
# Data/ml-10m, and Data/ml-20m respectively.

import os
import numpy as np

def get_data(size):
    ratings = []
    if size == "100k":
        path = os.path.join("Data", "ml-100k", "u.data")
        print("Read movie lens 100k data set")
        f = open(path, "r")
        while (1):
            line = f.readline()
            if line == "":
                break
            ratings.append(line.split()[0:-1])
        f.close()
    if size == "1m" or size == "10m":
        path = os.path.join("Data", "ml-" + size, "ratings.dat")
        print("Read movie lens " + size + " data set")
        f = open(path, "r")
        while (1):
            line = f.readline()
            if line == "":
                break
            ratings.append(line.split("::")[0:-1])
        f.close()
    if size == "20m":
        path = os.path.join("Data", "ml-20m", "ratings.csv")
        print("Read movie lens 20m data set")
        f = open(path, "r")
        line = f.readline()
        while (1):
            line = f.readline()
            if line == "":
                break
            ratings.append(line.split(",")[0:-1])
        f.close()
    ratings = np.array(ratings, dtype = np.float32)
    # permute the ratings array
    ratings = np.random.permutation(ratings)
    print("Loading data done")
    return ratings


