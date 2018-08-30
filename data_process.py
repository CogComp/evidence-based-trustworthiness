import pandas as pd
import codecs
from numpy import nan
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

n_features = 10000
directory = "dataset/emergent/"
files = "url-versions-2015-06-14.csv"
news = pd.read_csv(directory + files)
print(news.columns)
news = news[['claimHeadline', 'claimTruthiness', 'articleUrl', 'articleHeadline', 'articleId', 'articleBody', 'articleStance']]

#['claimHeadline', 'ClaimTruthiness', 'articleUrl', 'articleHeadline', 'articleBody', 'articleStance']
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
    elif row['claimTruthiness'] == 'unknown':
        continue
    else:
        print(row['claimTruthiness'])
print('There are', len(claim_label.keys()), 'claims')
print(set(news['articleStance'].tolist()))

source_evidence_claim = {}
content_feature = {}
tuple_num = 0
for iter, row in news.iterrows():
    url = row['articleUrl']
    url = url.split('/')
    if len(url) < 3:
        continue

    if type(row['articleBody']) == float:
        continue
    if type(row['articleStance']) == float:
        continue
    if type(row['articleHeadline']) == float:
        continue

    source_url = url[2]
    if source_url not in source_evidence_claim:
        source_evidence_claim[source_url] = {}

    if 'Claim:' in row['claimHeadline']:
        claim = row['claimHeadline'][7:]
    else:
        claim = row['claimHeadline']
    if claim not in claim_label:
        continue

    if claim not in source_evidence_claim[source_url]:
        source_evidence_claim[source_url][claim] = {}
    if row['articleStance'] == 'observing' or row['articleStance'] == 'for':
        source_evidence_claim[source_url][claim][1] = []
        source_evidence_claim[source_url][claim][1].append((row['articleHeadline'].lower(), row['articleBody'].lower()))
        tuple_num = tuple_num + 1
    elif row['articleStance'] == 'against' or row['articleStance'] == 'ignoring':
        source_evidence_claim[source_url][claim][-1] = []
        source_evidence_claim[source_url][claim][-1].append((row['articleHeadline'].lower(), row['articleBody'].lower()))
        tuple_num = tuple_num + 1
    else:
        print('Unobserved Stance Value!')

    content_feature[row['articleBody']] = 0
    content_feature[claim] = 0
    content_feature[row['articleHeadline']] = 0

print('There are', len(source_evidence_claim.keys()), 'sources')

content = []
for i in content_feature.keys():
    content = content + sent_tokenize(i.decode('utf-8'))

vectorize = TfidfVectorizer(max_features=n_features, stop_words='english')
vectorize.fit(content)
print('There are ', tuple_num, ' tuples')
