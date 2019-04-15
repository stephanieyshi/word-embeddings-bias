# Evaluation Metrics
There are several evaluation metrics, which tend to relate to vector distance, that have been used in literature to evaluate the results of debiasing. Since there is literature that shows that existing attempts to debias word embeddings have been insufficient, we believe it is best to use a variety of evaluation metrics to compare the results that have been done so far.

## Metric 1: Projection
### Direct Bias
We first aim to measure direct bias, as described in Man is to Computer Programmer (Bolukbasi et al. 2016). They use the following metric:

<img src="https://latex.codecogs.com/gif.latex?\text{DirectBias}_c \frac{1}{|N|} \sum_{w \in N} |\cos(\hat{w}, g)|^c" />

where <img src="https://latex.codecogs.com/gif.latex?c" /> is a strictness parameter, where <img src="https://latex.codecogs.com/gif.latex?g" />  is a learned gender direction, and where <img src="https://latex.codecogs.com/gif.latex?N" />  is the set of gender neutral words.

### Indirect Bias



## Metric 2: Pearson Correlation

## Metric 3: WEAT

## Metric 4: Clustering
