from bert_embedding import BertEmbedding
import json
import numpy as np

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
        print(word)
        embedding = result[i][1][0]
        embedding_dict[word] = embedding

    return embedding_dict

if __name__ == '__main__':
    gender_neutral_words = read_words('gender_neutral_words.txt')
    gender_specific_words = read_json('gender_specific_full.json')
    all_words = gender_neutral_words + gender_specific_words
    all_words = [word for word in all_words if '_' not in word]

    bert_embedding = BertEmbedding()
    result = bert_embedding(all_words)

    embedding_dict = bert_to_dict(result)
    write_embeddings_to_file(embedding_dict, 'bert_small.txt')