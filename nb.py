import preprocessing as pp
import numpy as np



vocab_list, df = pp.build_vocab()

def create_bow(sentence, vocab_list):
    word_list = pp.tokenize(sentence)
    bow = np.zeros(len(vocab_list))
    
    for word in word_list:
            bow[vocab_list[word]] = 1
    return bow

def train_nb(vocab_list, df):
    
    #find prior = total positive examples/total examples 
    total_sents = df.shape[0]
    pos_sents = df.loc[df['splitset_label'] == 1].shape[0]
    neg_sents = total_sents - pos_sents
    
    #initiate counts for word appearance conditional on label == 1 and label ==2
    #alpha is laplacian smoothing parameter
    alpha = 1
    pos_list = np.ones(len(vocab_list)) * alpha
    neg_list = np.ones(len(vocab_list)) * alpha
    
    for sentence, label in zip(df['sentence'].values, df['splitset_label']):
        bow = create_bow(sentence, vocab_list)   
      
        if label == 1:
            pos_list += bow
        else:
            neg_list +=bow
            
            
    #Calculate log-count ratio
    r = np.log(pos_list/pos_list.abs().sum())/(neg_list/neg_list.abs().sum())
    b = pos_sents/neg_sents
    
    return r, b

def predict_nb(sentence, r, b, vocab_list):
    bow = create_bow(sentence, vocab_list)
    result = np.sign(np.dot(bow, r.T) + b)
    
    if result > 0:
        return 1
    else:
        return 2
            
            

