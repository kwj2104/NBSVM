import preprocessing as pp
import numpy as np
from sklearn.svm import LinearSVC

#Hyperparameters
GRAM = 1
C = 1
BETA = .25


def create_bow(sentence, vocab_list, gram):
    word_list = pp.tokenize(sentence,gram)
    bow = np.zeros(len(vocab_list))
    
    for word in word_list:
            bow[vocab_list[word]] = 1
    return bow

def train_nb(vocab_list, df):
    
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
        bow = create_bow(sentence, vocab_list, GRAM)   
      
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

# Train SVM with L2 Regularization
def train_svm(vocab_list, df_train, gram, c):
    clf = LinearSVC(C=c, class_weight=None, dual=False, fit_intercept=True,
     loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
    
    X = np.array([create_bow(sentence, vocab_list, gram)  for sentence in df_train['sentence'].values])
    #y = np.where(df_train['label'].values == 1, 1, -1)
    y = df_train['label'].values
    
    clf.fit(X, y)
    
    svm_coef = clf.coef_
    svm_intercept = clf.intercept_
    
    #np.save('svm_coef.npy', svm_coef)
    #np.save('svm_intercept.npy', svm_intercept)
    
    return svm_coef, svm_intercept
    
    

#Use for MNB and SVM
def predict(df_test, w, b, vocab_list):
    
    total_sents = df_test.shape[0]
    total_score = 0
    
    for sentence, label in zip(df_test['sentence'].values, df_test['label']):
        predicted_label = 0
        
        bow = create_bow(sentence, vocab_list, GRAM)   
        result = np.sign(np.dot(bow, w.T) + b)
        if result > 0:
            predicted_label = 1
        else:
            predicted_label = -1
        
        if predicted_label == label:
            total_score += 1
            
    return total_score/total_sents

def predict_nbsvm(df_test, svm_coef, svm_intercept, r, b, vocab_list):
    total_sents = df_test.shape[0]
    total_score = 0
    
    for sentence, label in zip(df_test['sentence'].values, df_test['label']):
        predicted_label = 0
        bow = r * create_bow(sentence, vocab_list, GRAM)  
        w_bar = (abs(svm_coef).sum())/len(vocab_list)
        w_prime = (1 - BETA)*(w_bar) + (BETA * svm_coef)
        result = np.sign(np.dot(bow, w_prime.T) + svm_intercept)
        if result > 0:
            predicted_label = 1
        else:
            predicted_label = -1
        
        if predicted_label == label:
            total_score += 1
            
    return total_score/total_sents
    
            
        
    
def main():
    print("preprocessing...")
    vocab_list, df_train, df_test = pp.build_vocab(GRAM)

    #Train SVM
    print("training svm...")
    svm_coef, svm_intercept = train_svm(vocab_list, df_train, GRAM, C)
    #train_svm(vocab_list, df_train, GRAM, C)
    
    #print("predicting...")
    #print(predict(df_test, svm_coef, svm_intercept, vocab_list))
    

    print("training NB...")
    r, b = train_nb(vocab_list, df_train)
    #print("done")

    #print("predicting...")
    #print(predict(df_test, r, b, vocab_list))
    
    #Predict Using NBSVM
    print("Predict Using NBSVM:")
    print(predict_nbsvm(df_test, svm_coef, svm_intercept, r, b, vocab_list))

main()
    

    
            
            

