# Data

Our data comes from a number of sources.  The first source is from the *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings* paper. Since there is no gold standard, we do not have any test or gold standard datasets.

### Debiasing Word Embeddings Paper:

- definition_pairs.json contains a list of *definitional* gender word pairs including (woman, man), (girl, boy), (she, he), (female, male).  In particular, these are the 10 pairs of words that the researchers use to define gender direction.

- equalize_pairs.json contains a list of *gender-equivalent* word pairs including (father, mother), (monastery, convent), (spokesman, spokeswoman).  These are crowdsourced words.

- gender\_specific\_full.json contains a list of 1441 gender-specific words including actress, prince, ladies, bride, fraternity, etc.

- gender\_specific_seed.json contains a list of 218 gender-specific seed words including maternity, matriarch, businessman, businesswoman, etc.

- professions.json contains a list of professions and their respective direction towards a particular gender (positive values for male, negative for female)

- GoogleNews-vectors-negative300.bin.gz is the (zipped) file containing the existing embeddings the paper debiases.  The files can be downloaded [here](https://drive.google.com/drive/folders/0B5vZVlu2WoS5dkRFY19YUXVIU2M).

### Other Sources:

- Word embeddings from the New York Times Annotated Corpus (Sandhaus, 2008) trained using the GLoVe algorithm. These files can be downloaded [here](http://stanford.edu/~nkgarg/NYTembeddings/).

- Embeddings trained from a combination of the Wikipedia 2014 dump and the Gigaword 5 corpus using the GLoVe algorithm. The 6B token, 400K vocab version can be downloaded [here](http://nlp.stanford.edu/data/glove.6B.zip). The 42B token, 1.9M vocab version can be downloaded [here](http://nlp.stanford.edu/data/glove.42B.300d.zip). The 840B, 2.2M vocab version can be downloaded [here](http://nlp.stanford.edu/data/glove.840B.300d.zip).

- Twitter embeddings trained using the GLoVe algorithm. These files can be downloaded [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip).

- Word embeddings for historical English text (Hamilton, Leskovec, and Jurafsky, 2018) spanning all decades from 1800 to 2000. These are derived from Google N-Gram word2vec vectors and can be downloaded [here](http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip).

- Brown corpus case-insensitive word2vec embeddings. It can be downloaded [here](https://data.world/jaredfern/brown-corpus).

- Fasttext 1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset. It can be downloaded [here](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip).
