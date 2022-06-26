from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tokenizers import tokenize_into_words


tfidf_vect = TfidfVectorizer(
                    tokenizer = tokenize_into_words,
                    lowercase=False,
                )

vectorize = lambda Text: tfidf_vect.fit_transform(Text)
similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

def tfidf_cos_sim(pair_obj):
    src_text = pair_obj.read_src()
    sus_text = pair_obj.read_sus()
    src_vec, sus_vec = vectorize( [src_text , sus_text] )

    sim = cosine_similarity( src_vec , sus_vec)

    return sim[0][0]


