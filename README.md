# Music Recommendation System #
This repository hosts my code to [Kaggle KKBox's Music Recommendation Challenge](https://www.kaggle.com/c/kkbox-music-recommendation-challenge/ "Kaggle kkbox music recommendation challenge").

## Description ##
The goal is to predict repetitively listening behavior - whether a user listens to a particular song _again_ within 30 days of first listening. 

The data covers KKBOX user information such as age and gender, music metadata such as time length and names of artist/composer/lyricist. KKBOX provides the music listen history of selected users as training data. 

## Approach ##
This recommender is a hybrid of content-based recommendations and collaborative filtering. It heavily relies on music metadata. I train a `doc2vec` model to embed the names of each song's artist, composer, and lyricist. Other metainformation such as genre is also vectorized. Then I use matrix factorization, where both user and item (song) metadata are incorporated, to train a `LightFM` (latent representation recommendation) model and then calculate recommendation scores. 

This approach has two computationally expensive steps: 
1. The `doc2vec` model takes about a day vectorizing the artist/composer/lyricist names.
2. The matrix factorization algorithm requires a user features matrix, a song features matrix, and an implicit feedback user-song matrix (based on only observed repetitive listening); the latter two take a long time to construct. 

## Ongoing Work ##
I choose `doc2vec` as I would like to preserve the order of the tokenized words. This appears to be important as when multiple artists work together on a song, all of their names are given in the "artist" field (same for the composer/lyricist fields). "Jane Smith | John Doe" are different artists from "Jane Doe | John Smith". However, literature ([Kenter et al. 2016](http://aclweb.org/anthology/P/P16/P16-1089.pdf) shows that an average of `word2vec` vectors has strong performance on short text similarity tasks. Replacing `doc2vec` with `word2vec` may save us a substantial amount of time. 

## Potential Work ##
This data can also be used to recommend songs to a user that has never listened to them before. However, this recommender is set up to go over a list of songs for each user, calculate the recommendation scores for these songs, and then pick the ones with highest recommendation scores. This approach applies to the challenge as the testing data consists of specific user-song pairs and I only need to calculate recommendation scores for these pairs. Using this recommender over the entire song database before making recommendations for one individual user becomes computationally prohibitive. To extend this recommender to recommend "new" songs we need to first make a limited list. The list could be the top X most-played songs among all users or the top X most-played songs among an individual's favorite genres. 

## Requirements ##
`Python` 3, `gensim`, `nltk`, `LightFM`, `sci-kit learn`. 