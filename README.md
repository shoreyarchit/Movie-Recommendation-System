# Goal
The aim of the this project is to build a system that recommends top 10 movies that are similar to movie watched by a user and are highy rated by other users as well.

# Dataset
The dataset used here is Movie Lens dataset and follwing is brief summary of dataset:

This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.

# Technique
I have employed use of Word2Vec technique to train the model. Instead of using words and creating word embeddings, I have used Movies and created movie embeddings vector and trained it over. So, for two movies to be similar to each other, their movie embedding vector will essentially be similar.

And lastly we have optimized the cost using Optimizer in Pytorch.


