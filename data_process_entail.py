import pandas as pd
import codecs
import copy
from numpy import nan
import numpy as np
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from utility import sent2emb
import spacy
import pickle
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from trust_models import Baseline

evi_noise_ratio = 0.25
claim_evidence_entail_file =  "claim_evidence_score_adapt_top20_with_n_1 2.txt" #"claim_evidence_score_adapt_top20.txt"
nlp = spacy.load('en_core_web_sm')
directory = "dataset/emergent/"
files = "url-versions-2015-06-14.csv"
news = pd.read_csv(directory + files)
print(news.columns)
news = news[['claimHeadline', 'claimTruthiness', 'articleUrl', 'articleHeadline', 'articleId', 'articleBody', 'articleStance','articleHeadlineStance']]

def spearsman_source(source_gt, source_res):
    vec_gt = []
    vec_res = []
    for i in source_gt.keys():
        vec_gt.append(source_gt[i])
        vec_res.append(source_res[i])
    return spearmanr(vec_gt, vec_res)

def pearson_source(source_gt, source_res):
    vec_gt = []
    vec_res = []
    for i in source_gt.keys():
        vec_res.append(source_res[i])
        vec_gt.append(source_gt[i])
    return pearsonr(vec_gt, vec_res)

def precision_source(gt, result):
    acc = 0
    acc_total = 0
    for i in gt.keys():
        if i not in result:
            acc_total = acc_total + 1
        else:
            if result[i] == gt[i]:
                acc = acc + 1
            acc_total = acc_total + 1
    return float(acc)/float(acc_total)

def get_claim_ground_truth(news):
    claim_label = {}
    for iter, row in news.iterrows():
        if 'Claim:' in row['claimHeadline']:
            claim = row['claimHeadline'][7:]
        else:
            claim = row['claimHeadline']
        if row['claimTruthiness'].lower() == 'false':
            claim_label[claim] = -1
        elif row['claimTruthiness'].lower() == 'true':
            claim_label[claim] = 1
        elif row['claimTruthiness'].lower() == 'unknown':
            continue
        else:
            print(row['claimTruthiness'])
    print('There are', len(claim_label.keys()), 'claims')
    return claim_label

def get_source_ground_truth(news, claims_collection):
    source_claim = {}
    source_truthworthiness = {}

    for iter, row in news.iterrows():
        if 'Claim:' in row['claimHeadline']:
            claim = row['claimHeadline'][7:]
        else:
            claim = row['claimHeadline']

        if claim not in claims_collection:
            continue

        url = row['articleUrl']
        url = url.split('/')
        if len(url) < 3:
            continue

        source_url = url[2]
        if source_url not in source_truthworthiness:
            source_truthworthiness[source_url] = {}
            source_truthworthiness[source_url][0] = 0
            source_truthworthiness[source_url][1] = 0

        if source_url not in source_claim:
            source_claim[source_url] = {}
            source_claim[source_url][1] = []
            source_claim[source_url][-1] = []


        if row['articleStance'] == 'for':
            articles = 1
            source_claim[source_url][1].append(claim)
        elif row['articleStance'] == 'against':
            articles = -1
            source_claim[source_url][-1].append(claim)
        elif row['articleStance'] == 'observing':
            articles = 1
            source_claim[source_url][1].append(claim)
        else:
            articles = 0

        if row['claimTruthiness'] == 'true':
            claims = 1
        elif row['claimTruthiness'] == 'false':
            claims = -1
        else:
            claims = 0

        if claims * articles != 0:
            source_truthworthiness[source_url][1] = source_truthworthiness[source_url][1] + 1

        if claims * articles == 1:
            source_truthworthiness[source_url][0] = source_truthworthiness[source_url][0] + 1

    source_trust_gt = {}
    for i in source_truthworthiness.keys():
        if source_truthworthiness[i][1] == 0:
            source_trust_gt[i] = 0
        else:
            source_trust_gt[i] = float(source_truthworthiness[i][0]) / float(source_truthworthiness[i][1])

    return source_claim, source_trust_gt

def get_claim_evidence_entailment(filename):
    claim_evidence_support = {}
    with codecs.open(filename, "r") as inFp:
        tuples = inFp.readlines()
        tuples = [t.strip(" \n").split('\t')[:-1] for t in tuples]
        tuples = [t for t in tuples if len(t) == 5]
#        random.shuffle(tuples)
#        len_tuple = int(float(len(tuples))/float(2))

        for t in tuples:
            label_score = []
            for tid in range(2, 5):
                label_score.append((tid, float(t[tid])))
            label_score = sorted(label_score, key = lambda d:d[1], reverse=True)
            claim_evidence_support[t[0] + ";" + t[1]] = [label_score[0][0] - 2, label_score[0][1]]

        claim_keys = list(claim_evidence_support.keys())
        random.shuffle(claim_keys)
        len_keys = int(float(evi_noise_ratio) * float(len(claim_keys)))
        claim_keys = claim_keys[:len_keys]
        for ck in claim_keys:
            if claim_evidence_support[ck][0] == 1:
                claim_evidence_support[ck][0] = random.choice([-1,0])
            else:
                claim_evidence_support[ck][0] = random.choice([1,0])

        inFp.close()
    return claim_evidence_support

def source_claim_evidence_noisy_collector(news, source_truthworthiness, claim_label, claim_evidence):
    id = 0
    content_feature2 = {}
    source_claim_evidence_noisy = {}
    source_claim_evidence_example = {}
    for iter, row in news.iterrows():
        url = row['articleUrl']
        url = url.split('/')
        if len(url) < 3:
            continue

        source_url = url[2]

        if source_url not in source_truthworthiness:
            continue

        if 'Claim:' in row['claimHeadline']:
            claim = row['claimHeadline'][7:]
        else:
            claim = row['claimHeadline']

        if claim not in claim_label.keys():
            continue

        if source_url not in source_claim_evidence_noisy:
            source_claim_evidence_noisy[source_url] = {}
        if source_url not in source_claim_evidence_example:
            source_claim_evidence_example[source_url] = {}

        if claim not in source_claim_evidence_noisy[source_url]:
            source_claim_evidence_noisy[source_url][claim] = {}
            source_claim_evidence_noisy[source_url][claim][1] = []
            source_claim_evidence_noisy[source_url][claim][-1] = []
            source_claim_evidence_noisy[source_url][claim][0] = []

        if claim not in source_claim_evidence_example[source_url]:
            source_claim_evidence_example[source_url][claim] = {}
            source_claim_evidence_example[source_url][claim][1] = []
            source_claim_evidence_example[source_url][claim][-1] = []
            source_claim_evidence_example[source_url][claim][0] = []

        if claim not in content_feature2:
            content_feature2[claim] = id
            id = id + 1

        if type(row['articleHeadline']) != float:
            claim_ahl = claim + ";" + row['articleHeadline']
            if claim_ahl in claim_evidence:
                if row['articleHeadline'] not in content_feature2:
                    content_feature2[row['articleHeadline']] = id
                    id = id + 1

                entail_label = claim_evidence[claim_ahl][0]
                entail_score = claim_evidence[claim_ahl][1]
                if entail_label == 0:
                    source_claim_evidence_noisy[source_url][claim][1].append((row['articleHeadline'], entail_score))
                elif entail_label == 1:
                    source_claim_evidence_noisy[source_url][claim][-1].append((row['articleHeadline'], entail_score))
                else:
                    source_claim_evidence_noisy[source_url][claim][0].append((row['articleHeadline'], entail_score))

            if row['articleHeadlineStance'] == 'for':
                source_claim_evidence_example[source_url][claim][1].append((row['articleHeadline'],))
            elif row['articleHeadlineStance'] == 'against':
                source_claim_evidence_example[source_url][claim][-1].append((row['articleHeadline'],))
            elif row['articleHeadlineStance'] == 'observing':
                source_claim_evidence_example[source_url][claim][1].append((row['articleHeadline'],))
            else:
                source_claim_evidence_example[source_url][claim][0].append((row['articleHeadline'],))

        if type(row['articleBody']) != float:
            abd = sent_tokenize(row['articleBody'])
            for abs in abd:
                claim_abs = claim + ";" + abs
                if claim_abs in claim_evidence:
                    if abs not in content_feature2:
                        content_feature2[abs] = id
                        id = id + 1
                    entail_label = claim_evidence[claim_abs][0]
                    entail_score = claim_evidence[claim_abs][1]
                    if entail_label == 0:
                        source_claim_evidence_noisy[source_url][claim][1].append((abs, entail_score))
                    elif entail_label == 1:
                        source_claim_evidence_noisy[source_url][claim][-1].append((abs, entail_score))
                    else:
                        source_claim_evidence_noisy[source_url][claim][0].append((abs, entail_score))

    source_claim_evidence_noisy_temp = copy.deepcopy(source_claim_evidence_noisy)
    for i in source_claim_evidence_noisy_temp.keys():
        for j in source_claim_evidence_noisy_temp[i].keys():
            num_total = 0
            for k in source_claim_evidence_noisy_temp[i][j].keys():
                num_total = num_total + len(source_claim_evidence_noisy_temp[i][j][k])
            if num_total == 0:
                del source_claim_evidence_noisy[i][j]
    return source_claim_evidence_noisy, source_claim_evidence_example, content_feature2

claim_label = get_claim_ground_truth(news)
source_claim, source_truthworthiness = get_source_ground_truth(news, claim_label)
claim_evidence = get_claim_evidence_entailment(claim_evidence_entail_file)
source_claim_evidence_noisy, source_claim_evidence_example, content_feature2 = source_claim_evidence_noisy_collector(news, source_truthworthiness, claim_label, claim_evidence)

print('There are', len(source_truthworthiness.keys()), 'sources')
print('There are', len(content_feature2) , 'sentences.')



test1 = Baseline(source_claim, source_claim_evidence_noisy, claim_label, source_truthworthiness)
print("majority vote:")
cr1, sr1 = test1.majority_vote_claim_only()
print('precision :', precision_source(claim_label, cr1))
print('spearsman score :', spearsman_source(source_truthworthiness, sr1))
temp_total = sum(sr1.values())
for i in sr1.keys():
    sr1[i] = float(sr1[i])/float(temp_total)
print('pearsonr score :', pearson_source(source_truthworthiness, sr1))

#print("\n majority vote evidence:")
#r2, sr2, sr3 = test1.majority_vote_claim_evidence()
#print("precision: ", precision_source(claim_label, cr2))
#print("spearsman score: ", spearsman_source(source_truthworthiness, sr2))
#print("spearsman score extra: ", spearsman_source(source_truthworthiness, sr3))
#temp_total = sum(sr2.values())
#for i in sr2.keys():
#    sr2[i] = float(sr2[i])/float(temp_total)
#print("pearson score: ", pearson_source(source_truthworthiness, sr2))

#temp_total = sum(sr3.values())
#for i in sr3.keys():
#    sr3[i] = float(sr3[i])/float(temp_total)
#print('pearsonr score extra: ', pearson_source(source_truthworthiness, sr3))



#acc = 0
#for i in claim_label.keys():
#    if i in cr1 and i not in cr2:
#        if claim_label[i] == cr1[i]:
#            acc + 1
#    elif i in cr2 and i not in cr1:
#        if claim_label[i] == cr2[i]:
#            acc + 1
#    elif i in cr1 and i in cr2:
#        if claim_label[i] == cr1[i] or claim_label[i] == cr2[i]:
#            acc = acc + 1
#print('union precision: ', float(acc)/float(len(claim_label.keys())))


#
# with codecs.open("new_training_data_headline_with_n.txt", "w") as outFp:
#     for i in source_claim_evidence_example.keys():
#         for j in source_claim_evidence_example[i].keys():
#             for k in source_claim_evidence_example[i][j].keys():
#                 if k == claim_label[j]:
#                     for evi in source_claim_evidence_example[i][j][k]:
#                         outFp.write(evi[0] + "\t" + j + "\t" + str(k) + "\n")
#     outFp.close()
#
content2 = []
content_feature_array = sorted(content_feature2.items(), key = lambda d:d[1])
for i,j in content_feature_array:
    content2.append(i)
print(len(content2))

content = []
emb_rep = []
ent_rep = []

for i,j in content_feature_array:
    emb_rep.append(sent2emb(i))
    doc = nlp(i)
    temp_ent = []
    for ent in doc.ents:
        temp_ent = temp_ent + ent.text.split(" ")
    ent_rep.append(temp_ent)

pickle.dump(ent_rep,open('ent_rep_inference20.p','wb'))

vectorize2 = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
vectorize2.fit(content2)
tfidf_rep = vectorize2.transform(content2).toarray()
pickle.dump(tfidf_rep, open('tfidf_rep_inference20.p','wb'))
pickle.dump(emb_rep, open('glove_rep_inference20.p', 'wb'))

# #print('There are ', tuple_num, ' tuples.')
# print('There are ', len(tfidf_rep), ' tfidf.')
# print('There are ', len(emb_rep), ' embedding.')
# print('There are ', len(ent_rep), ' groups of entities.')
# print source_claim_evidence_noisy
# print claim_evidence
# for i in claim_evidence.keys():
#    print ent_rep[content_feature2[i]]
#    for j in claim_evidence[i]:
#        print ent_rep[content_feature2[j]]
#        print jd_sim(ent_rep[content_feature2[i]], ent_rep[content_feature2[j]])
#    print "\n"
#
# print np.array(tfidf_rep[10])
