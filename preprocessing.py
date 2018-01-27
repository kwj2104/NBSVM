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
    
def build_vocab():
    
    df_sent = pd.read_csv('Dataset/datasetSentences_test.txt', delimiter="\t")
    df_label = pd.read_csv('Dataset/datasetSplit_test.txt', delimiter=",")
    df = pd.merge(df_sent,df_label, on='sentence_index', how='outer')

    word_count = 0
    vocab_list = {}
    
    #create set 
    vocab_set = set()
    for sentence in df['sentence']:
        word_list = tokenize(sentence, 1)
        vocab_set.update(word_list)
        
    for word in vocab_set:
        vocab_list[word] = word_count
        word_count +=1  
    
    return vocab_list, df

        