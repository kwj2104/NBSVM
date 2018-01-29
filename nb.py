import preprocessing as pp
import numpy as np

GRAM = 1


def create_bow(sentence, vocab_list, gram):
    word_list = pp.tokenize(sentence,gram)
    bow = np.zeros(len(vocab_list))
    
    for word in word_list:
            bow[vocab_list[word]] = 1
    return bow

def train_nb(vocab_list, df, gram):
    
    #find prior = total positive examples/total examples 
    total_sents = df.shape[0]
    pos_sents = df.loc[df['label'] == 1].shape[0]
    neg_sents = total_sents - pos_sents
    
    #print("Total Sentences: ", total_sents)
    
    
    #initiate counts for word appearance conditional on label == 1 and label ==2
    #alpha is laplacian smoothing parameter
    alpha = 1
    pos_list = np.ones(len(vocab_list)) * alpha
    neg_list = np.ones(len(vocab_list)) * alpha
    
    for sentence, label in zip(df['sentence'].values, df['label']):
        bow = create_bow(sentence, vocab_list, gram)   
      
        if label == 1:
            pos_list += bow
        else:
            neg_list +=bow
            
            
    #Calculate log-count ratio
    #print(vocab_list.keys())
    #print(pos_list)
    #print(neg_list)
    r = np.log(pos_list/abs(pos_list).sum())/(neg_list/abs(neg_list).sum())
    b = pos_sents/neg_sents
    
    #save parameters
    np.save('trained_r.npy', r)
    np.save('trained_b.npy', b)
    
    return r, b

def predict_nb(sentence, r, b, vocab_list):
    bow = create_bow(sentence, vocab_list)
    result = np.sign(np.dot(bow, r.T) + b)
    
    if result > 0:
        return 1
    else:
        return 2
    
def main():
    print("preprocessing...")
    #vocab_list, df_train, df_test = pp.build_vocab(GRAM)
    print("training...")
    #train_nb(vocab_list, df_train, GRAM)
    print("done")
    
    print(np.load('trained_r.npy').shape)

main()
    
            
            

