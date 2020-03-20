# Naive-Bayes
Suppose that we're building an app that recommends movies. We've scraped a large set of reviews off the web, but we would like to recommend only movies with positive reviews. 
I use the Naive Bayes algorithm to train a binary sentiment classifier with a dataset of movie reviews. 
In naive_bayes.py, I learn a bag of words (unigram) model that will classify a review as positive or negative based on the words it contains. 
In naive_bayes_mixture.py, I combine the unigram and bigram models to achieve better performance on review classification.


## Requirements:
```
python3
pygame
```
## Running:

```
python3 mp3.py --training ../MP3_data_zip/train --development ../MP3_data_zip/dev --stemming False --lower_case True --laplace 0.1 --pos_prior 0.8
```

```
python3 mp3_mixture.py --training ../MP3_data_zip/train --development ../MP3_data_zip/dev --stemming False --lower_case True --bigram_lambda=0.5 --unigram_smoothing=0.1 --bigram_smoothing=0.1 --pos_prior 0.6
```

```
python3 mp3_tf_idf.py --training ../Mp3_data_zip/train --development ../Mp3_data_zip/dev
```
