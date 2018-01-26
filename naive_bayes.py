import torchtext
import tt_helper as th
import itertools 
import numpy as np
import Set
from nltk.corpus import stopwords


TEXT, LABEL, train_iter, val_iter, test_iter = th.SST_preprocessing()

vocab_len = len(TEXT.vocab)

#item = next(iter(train_iter))

#for i in range(10):
    #sent_indices =  item.text[:, i]
    #print(type(sent_indices.data[0]))
    #print(item.label[i])
    
    
def train_nb(train_iter): 

    pos_list = np.zeros(vocab_len)
    neg_list = np.zeros(vocab_len)
    pos_count = 0
    total_count = 0
    for item in itertools.islice(iter(train_iter), 100):
        for i in range(10):
            word_set = Set()
            
            for word_index in item.text[:, i].data:
                word_set.add(word_index)
            

            if(LABEL.vocab.itos[item.label.data[0]] == "positive"):
                pos_count = pos_count + 1
                for word_index in word_set:
                    pos_list[word_index] += 1
            else:
                for word_index in word_set:
                    neg_list[word_index] += 1
            
            total_count = total_count + 1
            
    neg_count = total_count - pos_count
    
    return (pos_list/pos_count, neg_list/neg_count, pos_count/total_count)


#print(train_nb(train_iter))

