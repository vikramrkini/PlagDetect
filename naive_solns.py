
from tokenizers import tokenize_into_words

def Jaccard_Similarity(token_list1: list , token_list2:list):
    """
    Calculates Jaccard Similarity between two text documents given their
    """
    s1 = set(token_list1)
    s2 = set(token_list2)
    score = len(s1 & s2) / len(s1 | s2)
    return round( score , 5)


if __name__ == '__main__':
    str1 = "My name is Mrunank. I like playing chess."
    str2 = "We are playing chess."

    t1 = tokenize_into_words(str1)
    t2 = tokenize_into_words(str2)
    print(t1,t2,sep='\n')

    sim = Jaccard_Similarity(t1 , t2)
    print("Jaccard Similarity:" , sim)
    pass