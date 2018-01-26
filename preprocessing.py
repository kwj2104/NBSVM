from nltk.corpus import stopwords


def rm_stopwords(word_list):
    return [word for word in word_list if word not in stopwords.words('english')]

def tokenize(sent, gram):
    words_list = rm_stopwords(sent.split())
    
    sent_tok = []
    
    for i in range(len(words_list) + 1 - gram):
        sent_tok.append("-".join(words_list[i:i + gram]))

    return sent_tok
    
def build_vocab(path):
    
    #vocab_list = Counter()
    
    for sentence in open(path).xreadlines():
        pass

            
    
#stopWords = set(stopwords.words('english'))
#print(stopWords)
    
    
print(tokenize("this is a test", 1))

        