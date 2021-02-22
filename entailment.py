from allennlp import predictors
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from scipy.spatial.distance import cosine
import pickle
import numpy as np
import codecs

n_features = 10000
topk = 20
directory = "dataset/emergent/"
files = "url-versions-2015-06-14.csv"
news = pd.read_csv(directory + files)
print(news.columns)

news = news[['claimHeadline', 'articleHeadline', 'articleBody']]

def evidence_search(topk, news, vflag):
    claim_evidence = {}
    sentences = []
    for iter, row in news.iterrows():
        if pd.isna(row['claimHeadline']):
            continue
        if 'Claim:' in row['claimHeadline']:
            claim = row['claimHeadline'][7:]
        else:
            claim = row['claimHeadline']

        if claim not in claim_evidence:
            claim_evidence[claim] = []
            sentences.append(claim)

        if pd.isna(row['articleHeadline']) == False:
            claim_evidence[claim].append(row['articleHeadline'])
            sentences.append(row['articleHeadline'])

        if pd.isna(row['articleBody']) == False:
            body = sent_tokenize(row['articleBody'])
            claim_evidence[claim] = claim_evidence[claim] + body
            sentences = sentences + body

    print("There are ", len(sentences), " sentences. \n")

    if vflag == True:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        joblib.dump(vectorizer, 'tfidf_model.joblib')

    vectorizer = joblib.load('tfidf_model.joblib')
    s_count = 0
    claim_short_evidence = {}
    for claim in claim_evidence.keys():
        real_topk = topk
        claim_short_evidence[claim] = []
        claim_vector = vectorizer.transform([claim]).toarray()[0]
        candidate_evidence = claim_evidence[claim]
        candidate_evidence_vector = vectorizer.transform(candidate_evidence).toarray()
        rank_temp = []
        for i in range(len(candidate_evidence)):
            sim = cosine(claim_vector, candidate_evidence_vector[i])
            if np.isnan(sim):
                continue
            sim = float(1) - sim
            rank_temp.append((i, sim))
        rank_temp = sorted(rank_temp, key=lambda d: d[1], reverse=True)
        if len(rank_temp) <= topk:
            real_topk = len(rank_temp)
        for i in range(real_topk):
            claim_short_evidence[claim].append(candidate_evidence[rank_temp[i][0]])
        s_count = s_count + real_topk

    pickle.dump(claim_short_evidence, open("claim_evidence.p", "wb"))
    print("There are ", str(s_count), " pairs to be predicted!")
    return claim_short_evidence

claim_evidence = evidence_search(topk, news, True)

#claim_evidence = pickle.load( open( "claim_evidence.p", "rb" ) )

#model = PretrainedModel("https://s3-us-west-2.amazonaws.com/allennlp/models/esim-elmo-2018.05.17.tar.gz",'textual-entailment').predictor()
#predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/esim-elmo-2018.05.17.tar.gz",'textual-entailment')
#predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")
predictor = Predictor.from_path("newmodel_adapt20_with_n_1/model.tar.gz")

with codecs.open("claim_evidence_score_adapt_top20_with_n_1.txt","w") as outFp:
    for claim in claim_evidence.keys():
        for evidence in claim_evidence[claim]:
            outFp.write(claim + "\t" + evidence + "\t")
            log_prob = predictor.predict( hypothesis= claim, premise=evidence)["label_probs"]
            for p in log_prob:
                outFp.write(str(p) + "\t")
            outFp.write("\n")
    outFp.close()


