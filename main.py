from string import punctuation,ascii_lowercase,digits,count
from textblob import TextBlob
from stop_words import get_stop_words
import sys

from model import LDA2Vec

import os
import os.path
import pickle
import time
import shelve

import chainer
from chainer import cuda
from chainer import serializers
import chainer.optimizers as O
import numpy as np

from lda2vec import utils
from lda2vec import prepare_topics, print_top_words_per_topic, topic_coherence
from lda2vec import preprocess
from lda2vec import Corpus
from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood
from lda2vec.utils import move
from L2V import l2v


from chainer import Chain
import chainer.links as L
import chainer.functions as F



def getDOCproperties(text, filename):
    #print "entered doc properties"
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    filters = {}
    vowels_count = 0
    consonants_count = 0
    ascii_lower_count = 0
    digits_count = 0
    for i in text:
        #print "entered for loop in doc properties"
        if i in ascii_lowercase:
            ascii_lower_count += 1
        if i in digits:
            digits_count += 1
        if i in vowels:
            vowels_count += 1
        if i in consonants:
            consonants_count += 1
    filters["filename"] = filename
    #print "filename done"
    filters["avg_word_len"] = np.mean([len(i) for i in text.split()])
    #print "avg word len done"
    filters["avg_line_len"] = np.mean([len(i) for i in text.split("\n")])
    #print "avg line len done"
    filters["englishness"] = vowels_count/(consonants_count+0.001)
    #print "englishness done"
    filters["alphabets/numbers"] = ascii_lower_count/(digits_count+0.001)
    #print "alpha/num done"
        
    return filters




def preprocess_and_filter(properties, txt):

    if properties.get('avg_word_len')>4 and       \
       properties.get('avg_word_len')<6.5 and      \
        properties.get('englishness')>0.5 and      \
        properties.get('alphabets/numbers')>10 and \
        properties.get('avg_line_len')>15:

            txt = "\n".join(line for line in txt.split("\n") if len(line)>100)
            txt = "".join(char for char in txt if char not in punctuation+digits)
            preprocessed = unicode(" ".join(word for word in TextBlob(txt).words if word not in stop_words and word in w2v.vocab))
            
            if len(preprocessed)>250:
                

                return preprocessed
            else:
                return None



def read_write_labeled_corpus(folder_path=None, preprocessed=True):
    
    if preprocessed:
        documents_filtered = []
        folder_name = os.path.basename(os.path.normpath(folder_path))
        folder_path = "clean/" + folder_name
        for root, dirs, files in os.walk(folder_path):

            for f in files:
                if ".txt" in f:
                    file_path = os.path.join(root, f)
                    with open(file_path,"r") as ff:
                        file_content = ff.read()

                    documents_filtered.append(unicode(file_content, errors="ignore"))
        return None, documents_filtered
    
    if not preprocessed:
        documents_filtered = []
        document_properties = {}
        #documents = {}
        folder_name = os.path.basename(os.path.normpath(folder_path))
        #print "folder name",folder_name
        if not os.path.isdir("clean"):os.mkdir("clean") 

        for root, dirs, files in os.walk(folder_path):

            for f in files:
                filename, ext = os.path.splitext(f)

                if ext in (".pdf",".txt",".doc"):

                    try:
                        #print "entered try"
                        clean_root = root.replace(folder_path, "clean/"+folder_name+"/")
                        #print "clean root :", clean_root
                        if not os.path.isdir(clean_root) : os.makedirs(clean_root)
                        file_path = os.path.join(root, f)
                        #print "file path :", file_path
                        clean_path = os.path.join(clean_root, filename+".txt")
                        #print "clean path :",clean_path
                        file_content = parser.from_file(file_path)["content"].lower()
                        #print "file  parsed :",f,"\n"
                        properties = getDOCproperties(file_content, f)
                        #print "properties done"
                        document_properties[f] = properties
                        preprocessed = preprocess_and_filter(properties, file_content)

                        if preprocessed is not None:
                            #print "file preprocessed :",f
                            documents_filtered.append(preprocessed)
                            with open(clean_path,"w") as ff:
                                ff.write(preprocessed.encode("utf-8"))
                    except Exception as e: print(str(e),f)
        return document_properties, documents_filtered


# if __name__ == "__main__":

# 	stop_words = get_stop_words('english')

# 	from gensim.models.word2vec import KeyedVectors
# 	fn_wordvc = "../lda2vec-tf/GoogleNews-vectors-negative300.bin"
# 	w2v = KeyedVectors.load_word2vec_format(fn_wordvc, binary=True)

# 	folder = sys.argv[1]

# 	epochs = sys.argv[2]

# 	train_properties, train_filtered = read_write_labeled_corpus(folder, preprocessed=False)

# 	model = l2v(docs=train_filtered, word2vec_model=w2v)

# 	model.train(epochs=epochs)

	







