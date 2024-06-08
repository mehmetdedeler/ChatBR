import gensim
from nltk import word_tokenize
from scipy import spatial

embedding_path = "../utils/embed_model/GoogleNews-vectors-negative300.bin"
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=True)


text_1 = "小明是一个好人"
text_2 = "坏人"


similarity = 1 - spatial.distance.cosine(
    word2vec_model.get_mean_vector(word_tokenize(text_1)),
    word2vec_model.get_mean_vector(word_tokenize(text_2))
)

print(similarity)
