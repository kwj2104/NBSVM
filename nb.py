import preprocessing as pp
import numpy as np
import sklearn as sk

GRAM = 2
C = 1


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
    
    print("Total Sentences: ", total_sents)
    print("Positive Sentences: ", pos_sents)
    
    
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
    x = (pos_list/abs(pos_list).sum())
    y = (neg_list/abs(neg_list).sum())

    r = np.log(x/y)

    #r = np.log(pos_list/abs(pos_list).sum())/(neg_list/abs(neg_list).sum())
    b = np.log(pos_sents/neg_sents)

    #save parameters
    np.save('trained_r.npy', r)
    np.save('trained_b.npy', b)
    
    return r, b

def train_svm(vocab_list, df, gram, c):
    clf = sk.LinearSVC(C=c, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=None, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
    
    pass

def predict_nb(df_test, r, b, vocab_list, gram):
    
    total_sents = df_test.shape[0]
    total_score = 0
    
    for sentence, label in zip(df_test['sentence'].values, df_test['label']):
        predicted_label = 0
        
        bow = create_bow(sentence, vocab_list, gram)   
        result = np.sign(np.dot(bow, r.T) + b)
        if result > 0:
            predicted_label = 1
        else:
            predicted_label = 2
        
        if predicted_label == label:
            total_score += 1
            
    return total_score/total_sents
        
    
        
    
def main():
    print("preprocessing...")
    vocab_list, df_train, df_test = pp.build_vocab(GRAM)
    #print(df_train['sentence'])
    print("training...")
    r, b = train_nb(vocab_list, df_train, GRAM)
    print("done")
    
    #r = np.load('trained_r.npy')
    #b = np.load('trained_b.npy')
    print(predict_nb(df_test, r, b, vocab_list, GRAM))

main()
    
            
            

