from preprocessing import create_bow, build_vocab
import numpy as np
from sklearn.svm import LinearSVC
import argparse as argparse

"""
Module for Multinomial Naive Bayes Classifier MNBC, and the 
Naive Bayes Support Vector Machine interpolation, NBSVM.

This implements the model using hyperparamters described 
by Wang and Manning at: 
http://www.aclweb.org/anthology/P/P12/P12-2.pdf#page=118
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gram", type=int, default=1)
    parser.add_argument("--C", type=float, default=1)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--alpha", type=float, default=1)

    return parser.parse_args()

args = parse_args()

"""
Trains the Multinomial Naive Bayes Model
"""
def train_nb(vocab_list, df):
    
    #find prior = total positive examples/total examples 
    total_sents = df.shape[0]
    pos_sents = df.loc[df['label'] == 1].shape[0]
    neg_sents = total_sents - pos_sents
    
    #initiate counts for word appearance conditional on label == 1 and label == 2
    #alpha is laplacian smoothing parameter
    pos_list = np.ones(len(vocab_list)) * args.alpha
    neg_list = np.ones(len(vocab_list)) * args.alpha
    
    for sentence, label in zip(df['sentence'].values, df['label']):
        bow = create_bow(sentence, vocab_list, args.gram)
      
        if label == 1:
            pos_list +=bow
        else:
            neg_list +=bow
            
    #Calculate log-count ratio
    x = (pos_list/abs(pos_list).sum())
    y = (neg_list/abs(neg_list).sum())
    r = np.log(x/y)
    b = np.log(pos_sents/neg_sents)
    
    return r, b

"""
Trains the (linear-kernel) SVM with L2 Regularization
"""
def train_svm(vocab_list, df_train, c, r):
    clf = LinearSVC(C=c, class_weight=None, dual=False, fit_intercept=True,
     loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
    
    X = np.array([(r * create_bow(sentence, vocab_list, args.gram))  for sentence in df_train['sentence'].values])
    y = df_train['label'].values
   
    clf.fit(X, y)   
    svm_coef = clf.coef_
    svm_intercept = clf.intercept_
    
    return svm_coef, svm_intercept, clf

"""
Predict classification with MNB
"""
def predict(df_test, w, b, vocab_list):
    total_sents = df_test.shape[0]
    total_score = 0
    
    for sentence, label in zip(df_test['sentence'].values, df_test['label']):      
        bow = create_bow(sentence, vocab_list, args.gram)

        result = np.sign(np.dot(bow, w.T) + b)
        if result == label:
            total_score +=1      
            
    return total_score/total_sents

"""
Predict classification with NB-SVM
"""
def predict_nbsvm(df_test, svm_coef, svm_intercept, r, b, vocab_list):
    total_sents = df_test.shape[0]
    total_score = 0
    
    for sentence, label in zip(df_test['sentence'].values, df_test['label']):
        bow = r * create_bow(sentence, vocab_list, args.gram)  
        w_bar = (abs(svm_coef).sum())/len(vocab_list)
        w_prime = (1 - args.beta)*(w_bar) + (args.beta * svm_coef)
        result = np.sign(np.dot(bow, w_prime.T) + svm_intercept)
        if result == label:
            total_score +=1  
            
    return total_score/total_sents


    
if __name__ == "__main__":
    print("Building Dataset...")
    vocab_list, df_train, df_val, df_test = build_vocab(args.gram)
    
    print("Training Multinomial Naive Bayes...")
    r, b = train_nb(vocab_list, df_train)

    #Train SVM
    print("Training LinearSVM...")
    svm_coef, svm_intercept, clf = train_svm(vocab_list, df_train, args.C, r)
    
    #Test Models
    accuracy = predict_nbsvm(df_test, svm_coef, svm_intercept, r, b, vocab_list)
    print("Test using NBSVM ({:.4f}-gram):".format(args.gram))
    print("Beta: {} Accuracy: {}".format(args.beta, accuracy))
    
    mnb_acc = predict(df_test, r, b, vocab_list)
    print("Test using MNB ({:.4f}-gram):".format(args.gram))
    print("Accuracy: {}".format(mnb_acc))
    
    


    



    

    
            
            
