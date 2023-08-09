# import numpy as np
# from gensim.models import KeyedVectors
# from scipy.spatial.distance import cosine

# # Assuming you have downloaded the GloVe embeddings and converted them to word2vec format:
# MODEL_PATH = "glove.6B.300d.word2vec.txt"
# model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)

# def get_vector(text):
#     words = [word for word in text.split() if word in model.vocab]
#     if words:
#         return np.mean(model[words], axis=0)
#     else:
#         return np.zeros((model.vector_size,))

# def get_most_related_authors_glove(query, authors, num=3):
#     query_vector = get_vector(query)

#     similarities = {}
#     for author in authors: 
#         author_vector = get_vector(author["id"])
#         similarity = 1 - cosine(query_vector, author_vector)
#         similarities[author["id"]] = similarity

#     sorted_authors = sorted(similarities.keys(), key=lambda author: -similarities[author])
#     return sorted_authors[:num]
