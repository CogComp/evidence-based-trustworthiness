import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.spatial.distance import cosine

def jd_sim(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    int1 = np.intersect1d(l1, l2)
    uni1 = np.union1d(l1,l2)
    if len(uni1) == 0:
        return 0
    else:
        return float(len(int1))/float(len(uni1))

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

glove = loadGloveModel('glove.6B.50d.txt')

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)

def sent2emb(sent):
    sent = sent.lower()
    tokens = word_tokenize(sent)#sent.decode('utf8'))
    vectors = []
    for tok in tokens:
        if tok in glove:
            vectors.append(glove[tok])
    if len(vectors) == 0:
        return np.zeros(50)
    else:
        return np.mean(vectors, axis=0)

def paragraph2emb(paragraph, sent_compare):
    sent_compare_emb = sent2emb(sent_compare)
    sents = sent_tokenize(paragraph)
    sent_vec = {}
    for sent in sents:
        sent_emb = sent2emb(sent)
        sent_vec.append((sent, 1 - cosine(sent_emb, sent_compare_emb)))
    sent_vec = sorted(sent_vec.items(), key = lambda d:d[1], reverse=True)
    return sent2emb(sent_vec[0][0])


