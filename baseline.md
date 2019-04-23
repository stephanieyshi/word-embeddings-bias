# Baseline
Our baseline measures direct bias and indirect bias on debiased embeddings along with its performance on word similarity and analogy-solving tasks. The following files are needed to run the debiasing file in baseline.py:

* original Google News embeddings: embeddings/w2v_gnews_small.txt
* definitional pairs file: data/definitional_pairs.json
* gender-specific words file: data/gender_specific_full.json
* pairs of gender-equivalent words file: data/equalize_pairs.json

The baseline can be executed by running `python3 baseline.py` in terminal. The resulting debiased embeddings should end up in: embeddings/debiased_w2v_gnews_small.txt

## Results 
### Direct Bias
We find that the debiased Google News embeddings set has a direct bias of 1.5%.

### Indirect Bias
We find the following indirect bias on the following word pairs:
* receptionist, softball: 5.274e-15
* waitress, softball: -0.047
* homemaker, softball: 6.974e-15
* businessman, football: -0.110
* maestro, football: -3.717e-15

### Pearson Correlation
When compared to the biases of the original embeddings, we observed a Pearson correlation of 0.4711 with a p-value of 4.438e-19.

### Clustering Accuracy
We find that the debiased embeddings produce a clustering accuracy of 74.2%.

### Spearman Correlation for Word Similarity Task
The Spearman correlation on the word similarity task using the WordSimilarity-353 test set is 0.6826.

### Analogy Task Accuracy
The embeddings produced an accuracy of 71% on the analogy task using the Google analogy test set.
