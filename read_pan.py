from os.path import join as path_join, exists as path_exists
from os import mkdir , listdir
from shutil import rmtree
import xml.etree.ElementTree as ET
from pprint import pprint
from nltk import corpus
import numpy as np

pan_data_path = r"data\pan13-text-alignment-training-corpus-2013-01-21"

foldernames = {
    1 : "01-no-plagiarism",
    2 : "02-no-obfuscation",
    3 : "03-random-obfuscation",
    4 : "04-translation-obfuscation",
    5 : "05-summary-obfuscation"
}

class PAN_Entry:
    def __init__(self,src_name, sus_name, xml_name,plag_class) -> None:
        self.src_name = src_name
        self.sus_name = sus_name
        self.xml_name = xml_name
        self.plag_class = plag_class
        
    def read_src(self):
        ''' Read src file and return its contents'''
        file_path = path_join(pan_data_path , "src" ,  self.src_name)
        src_file = open(file_path , encoding="mbcs")
        return src_file.read()

    def read_sus(self):
        ''' Read sus file and return its contents'''
        file_path = path_join(pan_data_path , "susp" , self.sus_name)
        sus_file = open(file_path,encoding="mbcs")
        return sus_file.read()

    def read_xml(self):
        ''' 
        Read xml file and return its contents.
        Also calculate percentage plagiarism
        '''
        path = path_join(pan_data_path , foldernames[self.plag_class] , self.xml_name)

        tree = ET.parse(path)
        self.src_size = len(self.read_src())
        self.src_plag_size = 0
        
        # root is a dictionary type object
        root = tree.getroot()

        for feature in root:
            # print()
            # pprint(feature.tag)
            self.src_plag_size += int( feature.attrib['this_length'] )
            # pprint(feature.attrib)
        
        self.plag_score = self.src_plag_size / self.src_size
        print('Plag_score:',self.plag_score)

def explore(pair_list:list , number:int):
    """
        For exploring and analyzing the data 
    """

    # Delete the explore folder
    if path_exists("Explore"):
        rmtree("Explore")

    # Make Explore directory
    mkdir("Explore")

    for i in range(number):
        pair_obj = pair_list[i]

        # Create the files for exploration purpose
        txtfile = open(  path_join ( "Explore", f"source_{i}.txt") , "w" ,encoding="mbcs" )
        susfile = open( path_join ( "Explore", f"susp_{i}.txt") , "w" ,encoding="mbcs" )


        susfile.write(pair_obj.sus_name+"\n")
        sus_contents = pair_obj.read_sus()
        susfile.write(sus_contents)

        txtfile.write(pair_obj.src_name+'\n')
        src_contents = pair_obj.read_src()
        txtfile.write(src_contents)

def read_pair_file(plag_class:int)->list:
    """
    Reads a "pairs" file and returns a list containing pan entry 
    objects.
    folder_no : the class of file that has to be accessed.
    """
    path = path_join(pan_data_path , foldernames[plag_class] , "pairs")
    pair_file = open(path ,'r')

    pair_list = []
    for pair in pair_file.readlines():
       sus_name , src_name = pair.strip().split() 
       xml_name = sus_name[:-4] + '-' +  src_name[:-4] + '.xml'
       obj = PAN_Entry(src_name , sus_name , xml_name , plag_class)
       pair_list.append(obj)

    return pair_list
 

def get_corpus():
    '''
        Returns a two lists of strings containing all src
        and sus documents respectively (src_corpus, sus_corpus)
    '''
    src_corpus = []
    sus_corpus = []

    src_files = listdir( path_join(pan_data_path , "src") )
    sus_files = listdir( path_join(pan_data_path , "susp") )
    
    for fname in src_files:
        file_path = path_join(pan_data_path , "src" ,  fname)
        src_file = open(file_path , encoding="mbcs")
        src_corpus.append( src_file.read())

    for fname in sus_files:
        file_path = path_join(pan_data_path , "susp" ,  fname)
        src_file = open(file_path , encoding="mbcs")
        sus_corpus.append( src_file.read())
    
    return src_files , sus_files , src_corpus , sus_corpus



if __name__ == '__main__':
    # plagiarism class you want to explore
    # plag_class = 2

    # pair_list = read_pair_file(plag_class)   

    # for pair_obj in pair_list[1:5]:
    #     pair_obj.read_xml()

    # explore(pair_list, 5)

    get_corpus()
    pass