from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from string import punctuation

penn2wordnet_tag = {
    "NN" : 'n' , "NNS": 'n', "NNP" :'n' , "NNPS":'n',
    "VBG":'v', "VBN" :	'v', "VBP" :'v', "VBZ":	'v',
    "VBD": 'v',"VB" : 'v',
    "JJ":'a' , "JJR":'a' , "JJS" :'a',
    "RB":'r', "RBS":'r' , "RBR":'r'
}


def tokenize_into_words(text:str):
    """
    Removes punctuations and get Lemmatized tokens of string 
    `text`. Lemmatization is context based.
    """

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)

    lemmatized_tokens = []
    tagged_words = pos_tag(tokens) 

    for word,tag in tagged_words:
        if word not in punctuation:
            wordnet_tag = penn2wordnet_tag.get(tag , "n")
            token = lemmatizer.lemmatize(word , pos = wordnet_tag)
            lemmatized_tokens.append(token)

    return lemmatized_tokens


if __name__ == "__main__":
    sample_string = "My name is Mrunank. I really like playing chess."
    tokens = tokenize_into_words(sample_string)
    print(tokens)

    




