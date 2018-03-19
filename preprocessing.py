from nltk.corpus import stopwords
import pandas as pd
import numpy as np


def create_bow(sentence, vocab_list, gram):
    word_list = tokenize(sentence, gram)
    bow = np.zeros(len(vocab_list))
    
    for word in word_list:
            bow[vocab_list[word]] = 1
    return bow

def rm_stopwords(word_list):
    return [word for word in word_list if word not in stopwords.words('english')]

def tokenize(sent, grams):
    words_list = rm_stopwords(sent.split())
    sent_tok = []
    for gram in range(1, grams + 1):
        for i in range(len(words_list) + 1 - gram):
            sent_tok.append("-".join(words_list[i:i + gram]))
    return sent_tok
    
"""
Loads the raw data from SST1 and manually converts into SST2
See: https://nlp.stanford.edu/sentiment/treebank.html
"""
def build_vocab(gram):
    
    #Load data
    df_sent = pd.read_csv('Dataset/datasetSentences.txt', delimiter="\t")
    df_label = pd.read_csv('Dataset/sentiment_labels.txt', delimiter="|")
    df_splitlabel = pd.read_csv('Dataset/datasetSplit.txt', delimiter=",")
    df_dictionary = pd.read_csv('Dataset/dictionary.txt', delimiter="|", names =['sentence', 'phrase ids'])
    df = pd.merge(df_sent, df_dictionary, on='sentence', how='left')
    df = pd.merge(df, df_splitlabel, on='sentence_index')
    df = pd.merge(df, df_label, on='phrase ids', how='left')
    
    #transform SST1 into SST2 by removing neutral reviews and making
    #classification binary
    def classify(sent_value):
        classification = 0
        if sent_value <= .4:
            classification = -1
        elif sent_value > .6:
            classification = 1
        else:
            classification = 3
            
        return classification
    
    df['label'] = df['sentiment values'].apply(classify)
    
    #drop all neutral reviews
    df = df[df['label'] != 3]
    
    word_count = 0
    vocab_list = {}
    
    #Create vocab set 
    vocab_set = set()
    for sentence in df['sentence']:
        word_list = tokenize(sentence, gram)
        vocab_set.update(word_list)
    
    #Assign each word a unique index
    for word in vocab_set:
        vocab_list[word] = word_count
        word_count +=1
    
    df_train = df[df['splitset_label'] == 1]
    df_val = df[df['splitset_label'] == 2]
    df_test = df[df['splitset_label'] == 3]
    
    return vocab_list, df_train, df_val, df_test



