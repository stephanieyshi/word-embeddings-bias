from pymagnitude import *
import json

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

if __name__ == '__main__':
    # data files
    neutral_words_file = 'data/gender_neutral_words.txt'
    specific_words_file = 'data/gender_specific_full.json'
    embeddings_file = 'embeddings/glove.twitter.27B.100d.magnitude'

    print("Collecting Data...")
    gender_neutral_words = read_words(neutral_words_file)
    gender_specific_words = read_json(specific_words_file)
    all_words = gender_neutral_words + gender_specific_words
    vectors = Magnitude(embeddings_file)

    print("Filling Dictionary...")
    embedding_dict = { word:vectors.query(word) for word in all_words }

    print("Writing to File...")
    write_embeddings_to_file(embedding_dict, 'embeddings/glove_small.txt')

    