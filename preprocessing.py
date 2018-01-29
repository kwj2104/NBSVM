from nltk.corpus import stopwords
import pandas as pd
from collections import Set


pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)



def rm_stopwords(word_list):
    return [word for word in word_list if word not in stopwords.words('english')]

def tokenize(sent, gram):
    words_list = rm_stopwords(sent.split())
    
    sent_tok = []
    
    for i in range(len(words_list) + 1 - gram):
        sent_tok.append("-".join(words_list[i:i + gram]))

    return sent_tok
    
def build_vocab(gram):
    
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
            classification = 2
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
    
    #create set 
    vocab_set = set()
    for sentence in df['sentence']:
        word_list = tokenize(sentence, gram)
        vocab_set.update(word_list)
        
    for word in vocab_set:
        vocab_list[word] = word_count
        word_count +=1  
    
    df_train = df[df['splitset_label'] == 1]
    df_test = df[df['splitset_label'] == 3]
    
    return vocab_list, df_train, df_test
