from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle

from read_pan import get_corpus
from os.path import join as path_join
from sklearn.metrics.pairwise import cosine_similarity

Maxlen = 1000
no_of_src_docs = 3230
no_of_sus_docs = 1827
ps = PorterStemmer()

def generate_cleaned_data_frame():
    # Get the corupus of text
    src_names , susp_names, src_corpus , sus_corpus = get_corpus()

    # get no of src_docs and sus_docs
    no_of_src_docs = len(src_corpus)
    no_of_sus_docs = len(sus_corpus)

    documents = src_corpus + sus_corpus
    doc_names = src_names + susp_names

    # documents = documents[:100]

    print(f" Read {len(documents)} documents!")

    

    documents_df=pd.DataFrame(
                        list(zip(doc_names, documents)),
                        columns=[ 'doc_names' , 'documents']
                )

    stop_words_l = set(stopwords.words('english'))
    preprocess = lambda x : ' '.join( ps.stem(w.lower()) for w in word_tokenize(x) if w not in stop_words_l)

    # print(preprocess)
    documents_df['documents_cleaned'] = documents_df.documents.apply( preprocess )

    documents_df.to_csv("clean_data.csv")
    
    print('Pre-proccessed Documents:' , len(documents_df))


def get_sim_method(doc_embeddings_path:str = path_join("data" ,"document_embeddings")):
    with open(doc_embeddings_path, 'rb') as f:
        document_embeddings = pickle.load(f)    

    def glove_tfidf_cos_sim(pair_obj):
        src_ind = int(pair_obj.src_name[-9:-4]) - 1
        sus_ind = int(pair_obj.sus_name[-9:-4]) + no_of_src_docs - 1

        sim = cosine_similarity( [document_embeddings[src_ind]] , [document_embeddings[sus_ind]])

        return sim[0][0]
        
    return glove_tfidf_cos_sim


if __name__ == '__main__':
    generate_cleaned_data_frame()
    pass
