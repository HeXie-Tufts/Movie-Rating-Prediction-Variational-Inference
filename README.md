# Movie-Rating-Prediction-Variational-Inference

Matrix Factorizaiton techniques are the most popular and effective choice for implementing Collaborative Filtering in Recommendation Systems. In the process of learning various Matrix Factorization algorithms, I implemented Variational Inference Matrix Factorizaiton introduced in:

Lim, Y. J., & Teh, Y. W. (2007). Variational Bayesian
approach to movie rating prediction. _Proceedings of
KDD Cup and Workshop._

When applying this implementation on MovieLens dataset and randomly select 10% data as test set. I achieved the following RMSE:

* MovieLens 100k, RMSE = 0.9039
* MovieLens 1m, RMSE = 0.8384
* MovieLens 10m, RMSE = 0.7812
