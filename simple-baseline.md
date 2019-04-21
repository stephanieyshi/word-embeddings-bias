# Simple Baseline
Our simple baseline measures direct bias and indirect bias on undebiased (in their original form) embeddings. We also measure its performance on word similarity and analogy-solving tasks.

## Direct Bias
We find that the Google News embeddings set has a direct bias of 8%.

## Indirect Bias
We find the following indirect bias on the following word pairs:
* receptionist, softball: 0.6723428949119995
* waitress, softball: 0.31784272010598474
* homemaker, softball: 0.38374112744622535
* businessman, football: 0.17007832577452814
* maestro, football: 0.4158048130010961

## Clustering Accuracy
We find that the original embeddings produce a clustering accuracy of 99.4%.

## Spearman Correlation for Word Similarity Task
The Spearman correlation on the word similarity task using the WordSimilarity-353 test set is 0.6857.

## Analogy Task Accuracy
The embeddings produced an accuracy of 68% on the analogy task using the Google analogy test set.
