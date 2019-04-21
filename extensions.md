# Extensions

## Extension 1: BERT
Instead of using context-free embeddings such as word2vec or GLoVe, we'll investigate BERT, which is a contextual model. We'll obtain pre-trained contextual embeddings, which are fixed contextual representations of each input token generated from the hidden layers of the pre-trained BERT model. We can then investigate the nature of gender bias in such embeddings, apply our de-biasing algorithm, and examine any residual bias in the de-biased embeddings. Questions we can investigate include:
- Does changing the input context to the pre-trained model change the degree and nature of bias in the embeddings (consider "Morgan is a programmer. His wife is a homemaker" vs "Morgan is a homemaker. His wife is a programmer.")?
- How good is BERT at solving analogies or performing on SQuAD compared to context-free embeddings?
- Is it possible to fine tune BERT as an additional pre-processing step to further remove bias from the input embeddings?

## Extension 2: Political bias
We could also investigate political bias in word embeddings. To do this, take a news source that has a high perceived level of bias (eg. Breitbart News) and collect a corpus of articles for this news source (eg. [here](https://www.kaggle.com/snapcrack/all-the-news)). Then, use Facebook [fastText](https://fasttext.cc/docs/en/unsupervised-tutorial.html) to generate word embeddings of this corpus. We'll also need to create definitional word pairs, equalize word pairs, gender-specific words, and other relevant politically-relevant word sets, similar to how we did so for gender bias. Once we've done this, we can investigate the nature of political bias in such embeddings, apply our de-biasing algorithm, and examine any residual bias in the de-biased embeddings. Questions we can investigate include:
- Does our de-biasing algorithm's performance differ significantly between the two types of bias (gender vs political)?
- If we perform a sentence-generation task using our de-biased embeddings, would the resulting phrases resemble that of a more lesser-biased news source such as NPR or BBC?
