# Data

Our data comes from a number of sources.  The first source is from the *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings* paper. Since there is no gold standard, we do not have any test or gold standard datasets.

### Debiasing Word Embeddings Paper:

- definition_pairs.json contains a list of *definitional* gender word pairs including (woman, man), (girl, boy), (she, he), (female, male).  In particular, these are the 10 pairs of words that the researchers use to define gender direction.

- equalize_pairs.json contains a list of *gender-equivalent* word pairs including (father, mother), (monastery, convent), (spokesman, spokeswoman).  These are crowdsourced words.

- gender\_specific\_full.json contains a list of 1441 gender-specific words including actress, prince, ladies, bride, fraternity, etc.

- gender\_specific_seed.json contains a list of 218 gender-specific seed words including maternity, matriarch, businessman, businesswoman, etc.

- professions.json contains a list of professions and their respective direction towards a particular gender (positive values for male, negative for female)

- GoogleNews-vectors-negative300.bin.gz is the (zipped) file containing the existing embeddings the paper debiases.  The files can be downloaded [here](https://drive.google.com/drive/folders/0B5vZVlu2WoS5dkRFY19YUXVIU2M).
