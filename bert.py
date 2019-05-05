from bert_embedding import BertEmbedding
import json
import numpy as np
from scoring.scoring import *
import matplotlib.pyplot as plt

def read_words(filename):
    words = list()
    with open(filename) as f:
        words = f.read().split('\n')[:-1]

    return words

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def write_embeddings_to_file(embedding_dict, filename):
    with open(filename, 'w') as f:
        for word in embedding_dict:
            f.write(word + " ")
            vector = ["{0:.7f}".format(val) for val in embedding_dict[word]]
            vector_string = " ".join(vector)
            f.write(vector_string + "\n")

def bert_to_dict(result):
    embedding_dict = dict()
    for i in range(len(result)):
        word = result[i][0][0]
        embedding = result[i][1][0]
        embedding_dict[word] = embedding

    return embedding_dict

if __name__ == '__main__':
    word_similarity_file = 'data/combined.csv'
    similarity_pairs = read_wordsim(word_similarity_file)
    all_words = set()
    for key in similarity_pairs:
        all_words.add(key[0])
        all_words.add(key[1])

    similarity_pairs = {key:similarity_pairs[key] for key in similarity_pairs}


    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
    result = bert_embedding(list(all_words))

    embedding_dict = bert_to_dict(result)
    similarity_dict = get_similarities(embedding_dict, list(similarity_pairs.keys()))

    word_pairs = similarity_pairs.keys()

    X = [similarity_pairs[pair] for pair in word_pairs]
    y = [similarity_dict[pair] for pair in word_pairs]

    plt.scatter(X, y)
    plt.show()

    spearman = stats.spearmanr([similarity_pairs[pair] for pair in word_pairs], 
                               [similarity_dict[pair] for pair in word_pairs])
    print(spearman)