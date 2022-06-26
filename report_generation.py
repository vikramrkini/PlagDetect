import pandas as pd
from read_pan import foldernames , read_pair_file
from tfidf import tfidf_cos_sim
from glove import get_sim_method

def gen_report(similarity_method , method_name = "TfIDF"  , normalize = True):
    """
        Generate the statistics on similarity scores lists
        for all classes
    """

    similarity_lists = []

    for plag_class in range(1,6):

        print("Working on class:" + str(plag_class) )
        pair_list = read_pair_file(plag_class)   
        scores = []

        for pair_obj in pair_list:
            sim = similarity_method(pair_obj)
            scores.append( round(sim,5))

        similarity_lists.append(scores)

    f = open(f'Reports\\Report_{method_name}.txt' , 'w')

    header = f"\nMethod = {method_name}"
    
    f.write(header)

    if normalize:
        mine , maxe = 1 , 0
        s = 0
        count = 0
        for similarity_scores in similarity_lists:
            mine = min( mine , min(similarity_scores))
            maxe = max( maxe , max(similarity_scores))
            s += sum(similarity_scores)
            count += len(similarity_scores)

        mean = s / count

        for similarity_scores in similarity_lists:
            n = len(similarity_scores)

            for i in range(n):
                x = similarity_scores[i]
                # similarity_scores[i] = (x - mean) / (maxe - mine)
                similarity_scores[i] = (x - mine) / (maxe - mine)

    for plag_class , similarity_scores in enumerate(similarity_lists):
        f.write(f"\n\nPlag Class : {foldernames[plag_class+1]}\n")
        s = pd.Series(similarity_scores)
        data = str(s.describe())
        f.write(data)

    f.close()



if __name__ == '__main__':
    glove_tfidf_cos_sim = get_sim_method()
    gen_report( glove_tfidf_cos_sim, method_name = "glove + TFIDF" )

    