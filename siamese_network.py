import pickle
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from os.path import join as path_join

ps = PorterStemmer()
s = tf.qint16

def get_siamese_sim():
    '''
        Creates and returns function comparing two texts using
        siamese network.
    '''

    with open(path_join( 'data' , f'keras_tokenizer'), 'rb') as f:
        tok = pickle.load(f)

    with open(path_join( 'data' ,f'embedding_matrix_siamese'), 'rb') as f:
        embedding_matrix = pickle.load(f)

    stop_words_l = set(stopwords.words('english'))
    preprocess = lambda x : ' '.join( ps.stem(w.lower()) for w in word_tokenize(x) if w not in stop_words_l)
    model = tf.keras.models.load_model('saved_model/my_model')

    def siamese_sim(src_text, sus_text):
        sequences = tok.texts_to_sequences([src_text , sus_text])
        sequences = np.asarray( [pad_sequences(sequences, maxlen=500, padding='post')] )
        preds = model.predict([ sequences[:,0], sequences[:,1] ])
        return 1 - round(float(preds[0][0]) , 4)

    return siamese_sim


if __name__ == "__main__":
    # txt1 = "My name is Mrunank. I like chess."
    # txt2 = "Me is Mrunank. I love chess and enjoy tactics."
    # siamese_sim = get_siamese_sim()
    # print( "Similarity" , siamese_sim(txt1 , txt2) )
    pass