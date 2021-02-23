#from data_process import source_evidence_claim
#from data_process import vectorize
#from data_process_full_conn import claim_label, tfidf_rep, source_claim_evidence_noisy, content_feature2, emb_rep, jd_sim, ent_rep
#from data_process_full_conn import vectorize2
from data_process_entail import claim_label, source_claim, source_claim_evidence_noisy, content_feature2, source_truthworthiness
from utility import jd_sim
from scipy.special import expit
from scipy.spatial.distance import cosine
from decimal import *
getcontext().prec = 6
import numpy as np
import pickle
import codecs
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
import copy

ent_rep = pickle.load(open('ent_rep_inference20.p','rb'))
emb_rep = pickle.load(open('glove_rep_inference20.p', 'rb'))
tfidf_rep = pickle.load(open('tfidf_rep_inference20.p', 'rb'))
print(tfidf_rep.shape)
smooth_para = 0.0000001

cweights = 0.95
const_para = 100


class TrustEvidence:

    def __init__(self, data1, data2, itr, gt, sgt):
        # Data Part
        self.source_claim = data1
        self.source_item_claim_evi = data2
        self.ground_truth = gt
        self.source_gt = sgt
#        self.source_num = 0

        # Hyper Parameter
        self.iteration_num = itr
        self.total_claim = 1000

        # Model Parameter
        self.Accuracy = {}
        self.item_val = {}
        self.item_val_dis = {}

    def __initialization__(self):

        #intialization for Source
        for i in self.source_item_claim_evi.keys():
            self.Accuracy[i] = 0.6

        # Intialization for Item Value Distribution
        for i in self.source_claim.keys():
            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    if k not in self.item_val_dis:
                        self.item_val_dis[k] = {}
                        self.item_val_dis[k][1] = -1
                        self.item_val_dis[k][-1] = -1
                        self.item_val_dis[k][0] = -1
                    if k not in self.item_val:
                        self.item_val[k] = ""
        for i in self.source_item_claim_evi.keys():
            for j in self.source_item_claim_evi[i].keys():
                if j not in self.item_val_dis:
                    self.item_val_dis[j] = {}
                    self.item_val_dis[j][1] = -1
                    self.item_val_dis[j][-1] = -1
                    self.item_val_dis[j][0] = -1
                if j not in self.item_val:
                    self.item_val[j] = ""


        #for i in self.source_item_claim_evi.keys():
        #    for j in self.source_item_claim_evi[i].keys():
        #        if j not in self.item_val_dis:
        #            self.item_val_dis[j] = {}
        #        for k in self.source_item_claim_evi[i][j].keys():
        #            self.item_val_dis[j][k] = -1
        #        if j not in self.item_val:
        #            self.item_val[j] = ""

    def precision(self, result):
        acc = 0
        acc_total = 0
        for i in self.ground_truth.keys():
            if i not in result:
                acc_total = acc_total + 1
            else:
                if result[i] == self.ground_truth[i]:
                    acc = acc + 1
                acc_total = acc_total + 1
        return float(acc)/float(acc_total)

    def error_analysis(self, result):
        value = {}
        for i in result.keys():
            if result[i] != self.ground_truth[i]:
                if result[i] not in value:
                    value[result[i]] = 0
                value[result[i]] = value[result[i]] + 1
        return value

    def label_analysis(self, result):
        value_dis = {}
        for i in result.keys():
            if result[i] not in value_dis:
                value_dis[result[i]] = 0
            value_dis[result[i]] = value_dis[result[i]] + 1
        return value_dis

    def one_label_accuracy(self):
        value = {}
        for i in self.ground_truth.keys():
            if self.ground_truth[i] not in value:
                value[self.ground_truth[i]] = 0
            value[self.ground_truth[i]] = value[self.ground_truth[i]] + 1
        for i in value.keys():
            value[i] = float(value[i])/float(len(self.ground_truth.keys()))
        return value

    def spearsman_source(self):
        vec_gt = []
        vec_res = []
        for i in self.source_gt.keys():
            vec_res.append(self.Accuracy[i])
            vec_gt.append(self.source_gt[i])
        return spearmanr(vec_gt, vec_res)

    def pearson_source(self):
        temp_source = copy.deepcopy(self.Accuracy)
        temp_total = sum(temp_source.values())
        for i in temp_source.keys():
            temp_source[i] = float(temp_source[i])/float(temp_total)

        vec_gt = []
        vec_res = []
        for i in self.source_gt.keys():
            vec_res.append(temp_source[i])
            vec_gt.append(self.source_gt[i])
        return pearsonr(vec_gt, vec_res)

class SimpleLCA(TrustEvidence):

    def __init__(self, data1, data2, itr, gt, sgt):
        TrustEvidence.__init__(self, data1, data2, itr, gt, sgt)

    def __initialization__(self):
        TrustEvidence.__initialization__(self)

    def precision(self, result):
        return TrustEvidence.precision(self, result)

    def spearsman_source(self):
        return TrustEvidence.spearsman_source(self)

    def pearson_source(self):
        return TrustEvidence.pearson_source(self)

    def error_analysis(self, result):
        return TrustEvidence.error_analysis(self, result)

    def label_analysis(self, result):
        return TrustEvidence.label_analysis(self, result)

    def one_label_accuracy(self):
        return TrustEvidence.one_label_accuracy(self)

    def ini_item_val_dis(self):
        for i in self.item_val_dis.keys(): # claim_id
            for j in self.item_val_dis[i].keys():
                self.item_val_dis[i][j] = -1

    def ini_item_val(self):
        for i in self.item_val.keys(): # claim_id
            self.item_val[i] = ""

    def ini_source(self):
        for i in self.Accuracy.keys():
            self.Accuracy[i] = 0.6

    def EstepLayer(self):
        for i in self.source_claim.keys():
            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    for l in self.item_val_dis[k].keys():
                        if l == j:
                            if self.item_val_dis[k][l] == -1:
                                self.item_val_dis[k][l] = self.Accuracy[i]
                            else:
                                self.item_val_dis[k][l] = self.item_val_dis[k][l] * self.Accuracy[i]
                        else:
                            if self.item_val_dis[k][l] == -1:
                                self.item_val_dis[k][l] = float(float(1) - self.Accuracy[i])/float(self.total_claim)
                            else:
                                self.item_val_dis[k][l] = self.item_val_dis[k][l] * float(float(1) - self.Accuracy[i])/float(self.total_claim)
        # Normalization
        #print(self.item_val_dis)
        for i in self.item_val_dis.keys(): # item_id
            sum_total = 0
            for j in self.item_val_dis[i].keys():
                sum_total = sum_total + self.item_val_dis[i][j]
            if sum_total != 0:
                for j in self.item_val_dis[i].keys():
                    self.item_val_dis[i][j] = float(self.item_val_dis[i][j])/float(sum_total)

        for i in self.item_val_dis.keys():
            #self.item_val_dis[i][0] = 0
            temp_rank = sorted(self.item_val_dis[i].items(), key = lambda d:d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]

    def MstepLayer(self): # Update Accuracy
        for i in self.source_claim.keys():
            temp = 0
            temp_total = 0
            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    if j == self.item_val[k]:
                        temp = temp + self.item_val_dis[k][j]
                    temp_total = temp_total + 1
            if temp_total != 0:
                self.Accuracy[i] = float(temp)/float(temp_total)
            else:
                self.Accuracy[i] = 0

    def SimLCA(self):
        self.__initialization__()
        for iter in range(self.iteration_num):
            self.ini_item_val_dis()
            self.ini_item_val()
            self.EstepLayer()
            self.ini_source()
            self.MstepLayer()
        return self.item_val
        #print sorted(self.item_val.items())
        #for i in self.item_evi_dis.keys():
        #    temp = sorted(self.item_evi_dis[i].items(), key = lambda d:d[1], reverse = True)
        #    self.item_evi[i] = (temp[0][0], temp[1][0], temp[2][0], temp[3][0], temp[4][0])

        #return self.Accuracy, self.item_val, self.item_evi

class SimpleLCA_evidence(TrustEvidence):

    def __init__(self, data, itr, gt):
        TrustEvidence.__init__(self, data, itr, gt)

    def __initialization__(self):
        TrustEvidence.__initialization__(self)

    def precision(self, result):
        return TrustEvidence.precision(self, result)

    def error_analysis(self, result):
        return TrustEvidence.error_analysis(self, result)

    def label_analysis(self, result):
        return TrustEvidence.label_analysis(self, result)

    def one_label_accuracy(self):
        return TrustEvidence.one_label_accuracy(self)

    def ini_item_val_dis(self):
        for i in self.item_val_dis.keys(): # claim_id
            for j in self.item_val_dis[i].keys():
                self.item_val_dis[i][j] = -1

    def ini_item_val(self):
        for i in self.item_val.keys(): # claim_id
            self.item_val[i] = ""

    def ini_source(self):
        for i in self.Accuracy.keys():
            self.Accuracy[i] = 0.6

    def EstepLayer(self):
        for i in self.source_item_claim_evi.keys(): # source
            for j in self.source_item_claim_evi[i].keys(): # claim_id
                for k in self.source_item_claim_evi[i][j].keys():
                    for l in self.item_val_dis[j].keys():
                        if l == k:
                            if self.item_val_dis[j][l] == -1:
                                self.item_val_dis[j][l] = self.Accuracy[i]
                            else:
                                self.item_val_dis[j][l] = self.item_val_dis[j][l] * self.Accuracy[i]
                        else:
                            if self.item_val_dis[j][l] == -1:
                                self.item_val_dis[j][l] = float(1 - self.Accuracy[i])#/float(self.total_claim)
                            else:
                                self.item_val_dis[j][l] = self.item_val_dis[j][l] * float(1 - self.Accuracy[i])#/float(self.total_claim)
        # Normalization

        for i in self.item_val_dis.keys(): # item_id
            sum_total = 0
            for j in self.item_val_dis[i].keys():
                sum_total = sum_total + self.item_val_dis[i][j]
            if sum_total != 0:
                for j in self.item_val_dis[i].keys():
                    self.item_val_dis[i][j] = float(self.item_val_dis[i][j])/float(sum_total)

        for i in self.item_val_dis.keys():
            temp_rank = sorted(self.item_val_dis[i].items(), key = lambda d:d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]

    def MstepLayer(self): # Update Accuracy
        for i in self.source_item_claim_evi.keys():
            temp = 0
            temp_total = 0
            for j in self.source_item_claim_evi[i].keys():
                for k in self.source_item_claim_evi[i][j].keys():
                    if k == self.item_val[j]:
                        temp = temp + 1
                    temp_total = temp_total + 1
            if temp_total != 0:
                self.Accuracy[i] = float(temp)/float(temp_total)
            else:
                self.Accuracy[i] = 0

    def SimLCA(self):
        self.__initialization__()
        for iter in range(self.iteration_num):
            self.ini_item_val_dis()
            self.ini_item_val()
            self.EstepLayer()
            self.ini_source()
            self.MstepLayer()
        return self.item_val
        #print sorted(self.item_val.items())
        #for i in self.item_evi_dis.keys():
        #    temp = sorted(self.item_evi_dis[i].items(), key = lambda d:d[1], reverse = True)
        #    self.item_evi[i] = (temp[0][0], temp[1][0], temp[2][0], temp[3][0], temp[4][0])

        #return self.Accuracy, self.item_val, self.item_evi

class EvidenceLCA_Combine(SimpleLCA):
    def __init__(self, data1, data2, itr, gt, sgt):
        SimpleLCA.__init__(self, data1, data2, itr, gt, sgt)
        self.Recall = {}
        self.Qf = {}
        self.Extractor = {}
        self.Extractor['cfd'] = 0.5
        self.Extractor['tfidf'] = 0.5
        self.Extractor['glove'] = 0.5
        self.Extractor['entity'] = 0.5
        self.Extractor['const'] = 0.5
        for i in self.Accuracy.keys():
            self.Accuracy[i] = 0.6

    def __initialization__(self):
        SimpleLCA.__initialization__(self)

        for i in self.Accuracy.keys():
            self.Recall[i] = 0.1
        for i in self.Accuracy.keys():
            self.Qf[i] = 0.067
        for i in self.Extractor.keys():
            self.Extractor[i] = 0.5

    def precision(self, result):
        return SimpleLCA.precision(self, result)

    def spearsman_source(self):
        return SimpleLCA.spearsman_source(self)

    def pearson_source(self):
        return SimpleLCA.pearson_source(self)

    def ini_item_val(self):
        SimpleLCA.ini_item_val(self)

    def ini_item_val_dis(self):
        SimpleLCA.ini_item_val_dis(self)

    def ini_source(self):
        SimpleLCA.ini_source(self)
        for i in self.Accuracy.keys():
            self.Accuracy[i] = 0.6
        for i in self.Recall.keys():
            self.Recall[i] = 0.1
        for i in self.Qf.keys():
            self.Qf[i] = 0.067

    def EstepLayer(self):
        for i in self.Recall.keys():
            self.Recall[i] = float(self.Accuracy[i] * self.Qf[i])/float(float(1) - self.Accuracy[i] + smooth_para)

        for i in self.source_item_claim_evi.keys():  # source
            for j in self.source_item_claim_evi[i].keys():  # claim_id
                claim_vector = tfidf_rep[content_feature2[j]]
                claim_ent = ent_rep[content_feature2[j]]
                claim_emb = emb_rep[content_feature2[j]]

                temp_total = 0
                for k in self.source_item_claim_evi[i][j].keys():
                    temp_total = temp_total + len(self.source_item_claim_evi[i][j][k])


                for k in self.source_item_claim_evi[i][j].keys():
                    exp_total = 0
                    total = 0
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        if evi not in content_feature2:
                            continue
                        total = total + 1
                        headline_vector = tfidf_rep[content_feature2[evi]]
                        headline_emb = emb_rep[content_feature2[evi]]
                        headline_ent = ent_rep[content_feature2[evi]]

                        cos1 = 1 - cosine(claim_vector, headline_vector)
                        cos2 = 1 - cosine(claim_emb, headline_emb)
                        cos3 = jd_sim(claim_ent, headline_ent)

                        if np.isnan(cos1):
                            cos1 = 0
                        if np.isnan(cos2):
                            cos2 = 0
                        if np.isnan(cos3):
                            cos3 = 0

                        exp_score = self.Extractor['tfidf'] * cos1 + self.Extractor['glove'] * cos2 + self.Extractor['entity'] * cos3 + \
                                          self.Extractor['cfd'] * float(escore) + self.Extractor['const'] * float(1)
                        exp_total = exp_total + exp_score

                    if total == 0:
                        exp_score =  expit(0)
                    else:
                        exp_score =  expit(float(exp_total)/float(total))
                    for l in self.item_val_dis[j].keys():
                        if l == k:
                            if self.item_val_dis[j][l] == -1:
                                self.item_val_dis[j][l] = 10 * float(len(self.source_item_claim_evi[i][j][k]))/float(temp_total) * np.log(self.Accuracy[i] * exp_score + smooth_para)
                            else:
                                self.item_val_dis[j][l] = self.item_val_dis[j][l] + 10 * float(len(self.source_item_claim_evi[i][j][k]))/float(temp_total) * np.log(self.Accuracy[i] * exp_score + smooth_para)
                        else:
                            if self.item_val_dis[j][l] == -1:
                                self.item_val_dis[j][l] = 10 * float(len(self.source_item_claim_evi[i][j][k]))/float(temp_total) * np.log(float(1 - self.Accuracy[i] * exp_score) / float(self.total_claim) + smooth_para)
                            else:
                                self.item_val_dis[j][l] = self.item_val_dis[j][l] + 10 * float(len(self.source_item_claim_evi[i][j][k]))/float(temp_total) * np.log(float(1 - self.Accuracy[i] * exp_score) / float(self.total_claim) + smooth_para)


        #print(self.item_val_dis)

        for i in self.item_val_dis.keys():
            temp_rank = sorted(self.item_val_dis[i].items(), key=lambda d: d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]

        print(self.precision(self.item_val))

        for i in self.source_claim.keys():
            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    for l in self.item_val_dis[k].keys():
                        if l == j:
                            if self.item_val_dis[k][l] == -1:
                                self.item_val_dis[k][l] = np.log(self.Accuracy[i] + smooth_para)
                            else:
                                self.item_val_dis[k][l] = self.item_val_dis[k][l] + np.log(self.Accuracy[i] + smooth_para)
                        else:
                            if self.item_val_dis[k][l] == -1:
                                self.item_val_dis[k][l] = np.log(float(float(1) - self.Accuracy[i])/float(self.total_claim) + smooth_para)
                            else:
                                self.item_val_dis[k][l] = self.item_val_dis[k][l] + np.log(float(float(1) - self.Accuracy[i])/float(self.total_claim) + smooth_para)

        # Normalization
        #print(self.item_val_dis)
        for i in self.item_val_dis.keys():  # item_id
            sum_total = 0
            for j in self.item_val_dis[i].keys():
                if self.item_val_dis[i][j] != -1:
                    sum_total = sum_total + np.exp(self.item_val_dis[i][j])

            if sum_total != 0:
                for j in self.item_val_dis[i].keys():
                    if self.item_val_dis[i][j] != -1:
                        self.item_val_dis[i][j] = np.exp(float(self.item_val_dis[i][j])) / float(sum_total)
            self.item_val_dis[i][0] = 0

        for i in self.item_val_dis.keys():
            temp_rank = sorted(self.item_val_dis[i].items(), key=lambda d: d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]

        print(self.precision(self.item_val))
        print('\n')

    def compute_gradient_alpha(self):

        gradient_Alpha = {}
        gradient_Alpha['glove'] = 0
        gradient_Alpha['tfidf'] = 0
        gradient_Alpha['entity'] = 0
        gradient_Alpha['cfd'] = 0
        gradient_Alpha['const'] = 0

        for ft in gradient_Alpha.keys():
            for i in self.source_item_claim_evi.keys():
                for j in self.source_item_claim_evi[i].keys():
                    total = 0
                    for k in self.source_item_claim_evi[i][j].keys():
                        if k != 0:
                            total = total + len(self.source_item_claim_evi[i][j][k])

                    claim_vector = tfidf_rep[content_feature2[j]]
                    claim_ent = ent_rep[content_feature2[j]]
                    claim_emb = emb_rep[content_feature2[j]]
                    for k in self.source_item_claim_evi[i][j].keys():
                        exp_total = 0
                        temp_total = 0
                        distance_total = {}
                        distance_total['tfidf'] = 0
                        distance_total['glove'] = 0
                        distance_total['entity'] = 0
                        distance_total['cfd'] = 0
                        distance_total['const'] = 0

                        for evi, escore in self.source_item_claim_evi[i][j][k]:
                            if evi not in content_feature2:
                                continue
                            temp_total = temp_total + 1

                            headline_vector = tfidf_rep[content_feature2[evi]]
                            headline_emb = emb_rep[content_feature2[evi]]
                            headline_ent = ent_rep[content_feature2[evi]]

                            distance_all = {}
                            cos1 = 1 - cosine(claim_vector, headline_vector)
                            cos2 = 1 - cosine(claim_emb, headline_emb)
                            cos3 = jd_sim(claim_ent, headline_ent)
                            distance_all['tfidf'] = cos1
                            distance_all['glove'] = cos2
                            distance_all['entity'] = cos3
                            distance_all['cfd'] = float(escore)
                            distance_all['const'] = float(1)
                            for key_f in distance_all.keys():
                                distance_total[key_f] = distance_total[key_f] + distance_all[key_f]
                            if np.isnan(cos1):
                                cos1 = 0
                            if np.isnan(cos2):
                                cos2 = 0
                            if np.isnan(cos3):
                                cos3 = 0

                            exp_total = exp_total + self.Extractor['tfidf'] * cos1 + self.Extractor['glove'] * cos2 + self.Extractor[
                                    'entity'] * cos3 + self.Extractor['cfd'] * float(escore) + self.Extractor['const'] * float(1)
                        if temp_total != 0:
                            exp_score = expit(float(exp_total)/float(temp_total))
                        else:
                            exp_score = expit(0)

                        if k == self.item_val[j]:
                            bsyd = float(1)
                        else:
                            bsyd = float(0)

                        for key_f in distance_total.keys():
                            if temp_total != 0:
                                distance_total[key_f] = float(distance_total[key_f])/float(temp_total)
                            else:
                                distance_total[key_f] = 0

                        item1 = float(bsyd) * float(1 - exp_score) * distance_total[ft]

                        if float(self.Accuracy[i]) * float(exp_score) == 1:
                            item2 = float(1 - float(bsyd)) * float(exp_score - 1) * distance_total[ft]
                        else:
                            item2 = float(1 - float(bsyd)) * float(self.Accuracy[i] * exp_score) / float(float(self.Accuracy[i]) * float(exp_score) - float(1)) * float(1 - exp_score) * float(distance_total[ft])
                        gradient_Alpha[ft] = gradient_Alpha[ft] + float(len(self.source_item_claim_evi[i][j][k]))/float(total) * float(self.item_val_dis[j][k]) * (
                            float(item1) + float(item2))
        return gradient_Alpha

    def compute_gradient_Ps(self):
        gradient_P = {}
        for i in self.source_item_claim_evi.keys():
            gradient_iter = 0
            for j in self.source_item_claim_evi[i].keys():
                claim_vector = tfidf_rep[content_feature2[j]]
                claim_ent = ent_rep[content_feature2[j]]
                claim_emb = emb_rep[content_feature2[j]]
                total = 0
                for k in self.source_item_claim_evi[i][j].keys():
                    if k != 0:
                        total = total + len(self.source_item_claim_evi[i][j][k])

                for k in self.source_item_claim_evi[i][j].keys():
                    exp_temp = 0
                    temp_total = 0
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        if evi not in content_feature2:
                            continue
                        temp_total = temp_total + 1
                        headline_vector = tfidf_rep[content_feature2[evi]]
                        headline_ent = ent_rep[content_feature2[evi]]
                        headline_emb = emb_rep[content_feature2[evi]]

                        cos1 = 1 - cosine(claim_vector, headline_vector)
                        cos2 = jd_sim(claim_ent, headline_ent)
                        cos3 = 1 - cosine(claim_emb, headline_emb)

                        if np.isnan(cos1):
                            cos1 = 0
                        if np.isnan(cos2):
                            cos2 = 0
                        if np.isnan(cos3):
                            cos3 = 0

                        in_sigma = {}
                        in_sigma['tfidf'] = cos1
                        in_sigma['entity'] = cos2
                        in_sigma['glove'] = cos3
                        in_sigma['cfd'] = escore
                        in_sigma['const'] = 1

                        for l in in_sigma.keys():
                            if np.isnan(in_sigma[l]):
                                in_sigma[l] = 0

                        sigma_sim = 0
                        for l in self.Extractor.keys():
                            sigma_sim = sigma_sim + self.Extractor[l] * float(in_sigma[l])

                        exp_temp = exp_temp + sigma_sim

                    if temp_total == 0:
                        exp_score = expit(0)
                    else:
                        exp_score = expit(float(exp_temp)/float(temp_total))

                    if k == self.item_val[j]:
                        bsyd = 1
                    else:
                        bsyd = 0

                    item1 = float(1)/float(self.Accuracy[i]) * float(bsyd)
                    item2 = float(1 - bsyd) * exp_score/float(float(self.Accuracy[i]* exp_score) - float(1))
                    gradient_iter = gradient_iter + self.item_val_dis[j][k] * float(len(self.source_item_claim_evi[i][j][k]))/float(total) * (item1 + item2)

            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    if j == self.item_val[k]:
                        bsyd = 1
                    else:
                        bsyd = 0
                    item1 = float(bsyd) / float(self.Accuracy[i])
                    item2 = float(1-bsyd)/float(self.Accuracy[i] - 1)
                    gradient_iter = gradient_iter + self.item_val_dis[k][j] * float(item1 + item2)

            gradient_P[i] = gradient_iter
        return gradient_P

    def MstepLayer(self, lr1, lr1b, lr2, ext_pre):
        temp_total1 = {}
        for i in self.Accuracy.keys():
            temp_total1[i] = 0
            self.Accuracy[i] = 0
            self.Recall[i] = 0

        for i in self.source_claim.keys():
            acc = 0
            temp = 0
            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    if j == self.item_val[k]:
                        acc = acc + 1
                    temp = temp + 1
            if temp == 0:
                self.Accuracy[i] = 0
            else:
                self.Accuracy[i] = float(acc)/float(temp)
        #
        #     self.Recall[i] = float(acc)/float(len(self.ground_truth.keys()))

        # for i in range(50):
        #     pre_alpha = {}
        #     for m in self.Accuracy.keys():
        #         pre_alpha[m] = self.Accuracy[m]
        #
        #     gradient_alpha = self.compute_gradient_Ps()
        #
        #     lr = lr1 #/ float(i + 1)
        #     for m in gradient_alpha.keys():
        #         self.Accuracy[m] = self.Accuracy[m] + float(lr) * gradient_alpha[m]
        #
        #     delta_extractor = {}
        #     for m in self.Accuracy.keys():
        #         delta_extractor[m] = abs(self.Accuracy[m] - pre_alpha[m])
        #     delta = sum(delta_extractor.values())
        #
        #     if delta < lr1b:
        #         break
        #     print("iteration acc", i)


        # for i in self.source_item_claim_evi.keys():
        #     for j in self.source_item_claim_evi[i].keys():
        #         if self.item_val[j] == 0:
        #             continue
        #         totalk = 0
        #         for k in self.source_item_claim_evi[i][j].keys():
        #             if k == 0:
        #                 continue
        #             totalk = totalk + len(self.source_item_claim_evi[i][j][k])
        #         claim_number = claim_number + totalk
        #
        #         for k in self.source_item_claim_evi[i][j].keys():
        #             if self.item_val[j] == 0:
        #                 continue
        #             if k == self.item_val[j]:
        #                 self.Recall[i] = self.Recall[i] + len(self.source_item_claim_evi[i][j][k])

        #for i in self.Recall.keys():
        #    self.Recall[i] = float(self.Recall[i])/float(claim_number)



        for i in range(50):
            pre_alpha = {}
            for m in self.Extractor.keys():
                pre_alpha[m] = self.Extractor[m]

            gradient_alpha = self.compute_gradient_alpha()

            lr = lr2 #/ float(i + 1)
            for m in gradient_alpha.keys():
                self.Extractor[m] = self.Extractor[m] + float(lr) * gradient_alpha[m]

            delta_extractor = {}
            for m in self.Extractor.keys():
                delta_extractor[m] = abs(self.Extractor[m] - pre_alpha[m])
            delta = sum(delta_extractor.values())

            if delta < ext_pre:
                break
            print("iteration extractor", i)
        #print(self.Accuracy)
        #print(self.Extractor)

    def LCA(self):
        self.__initialization__()
        pre_accuracy = {}
        for i in self.Accuracy.keys():
            pre_accuracy[i] = 0.6

        lr1 = 0.000001
        lr1_b = 0.0001
        lr2 = 0.0001
        lr2_b = 0.0001
        lr3 = 0.001
        lr3_b = 0.001

        for iter in range(self.iteration_num):
            self.ini_item_val_dis()
            self.ini_item_val()
            self.EstepLayer()
            self.ini_source()
            for i in self.Extractor.keys():
                self.Extractor[i] = 0.5
            self.MstepLayer(lr1, lr1_b, lr2, lr2_b)  # (0.0001, 0.1)

            print(self.precision(self.item_val))
            print(self.spearsman_source())
            print(self.pearson_source())
            print("\n")

class EvidenceLCA_FullConnect(SimpleLCA):

    def __init__(self, data, itr, gt):
        SimpleLCA.__init__(self, data, itr, gt)
        self.Extractor = {}
        for i in claim_label.keys():
            if i not in content_feature2:
                continue
            self.Extractor[i] = {}
            self.Extractor[i]['tfidf'] = 0
            self.Extractor[i]['glove'] = 0

    def __initialization__(self):
        # Intialization for source trustworthiness
        self.source_num = len(self.source_item_claim_evi.keys())
        for i in self.source_item_claim_evi.keys():
            self.Accuracy[i] = 0.5

        # Intialization for Item Value Distribution
        for i in claim_label.keys():
            if i not in content_feature2:
                continue
            if i not in self.item_val_dis:
                self.item_val_dis[i] = {}
            if i not in self.item_val:
                self.item_val[i] = -1
            self.item_val_dis[i][1] = -1
            self.item_val_dis[i][-1] = -1

        for i in self.Extractor.keys():
            for j in self.Extractor[i].keys():
                self.Extractor[i][j] = 1

    def precision(self, result):
        return SimpleLCA.precision(self, result)

    def ini_item_val(self):
        SimpleLCA.ini_item_val(self)

    def ini_item_val_dis(self):
        SimpleLCA.ini_item_val_dis(self)

    def ini_source(self):
        SimpleLCA.ini_source(self)

    def compute_gradient_Hs(self):
        gradient_H = {}
        for i in self.source_item_claim_evi.keys():
            gradient_iter = 0
            for j in self.source_item_claim_evi[i].keys():

                for ahl, ahb in self.source_evi[i][j]:
                    article_vector = tfidf_rep[content_feature2[ahl]]
                    article_emb = sent2emb(ahl)
                    for k in claim_label.keys():
                        if k not in content_feature2:
                            continue
                        claim_vector = tfidf_rep[content_feature2[k]]

                        claim_emb = sent2emb(k)

                        if j == self.item_val[k]:
                            bsyd = 1
                        else:
                            bsyd = 0

                        in_sigma = {}
                        in_sigma['tfidf'] = 1 - cosine(claim_vector, article_vector)
                        in_sigma['glove'] = 1 - cosine(claim_emb, article_emb)

                        for l in in_sigma.keys():
                            if np.isnan(in_sigma[l]):
                                in_sigma[l] = 0

                        sigma_sim = 0
                        for l in self.Extractor[k].keys():
                            sigma_sim = sigma_sim + self.Extractor[k][l] * in_sigma[l]

                        sigma_sim = expit(sigma_sim)

                        temp_item = bsyd - self.Accuracy[i] * sigma_sim
                    #if self.Accuracy[i] != 0:
                        if float(self.Accuracy[i] * (1-self.Accuracy[i]*sigma_sim)) == 0:
                            temp_item = float(temp_item)/(float((self.Accuracy[i] + 0.00001) * (1-(self.Accuracy[i] + 0.00001)*sigma_sim)))
                        else:
                            temp_item = float(temp_item)/float(self.Accuracy[i] * (1-self.Accuracy[i]*sigma_sim))

                        temp_item = temp_item * float(self.item_val_dis[k][j])

                        if np.isnan(temp_item):
                            temp_item = 0
                            #print "Here nan"
                            #print self.Accuracy[i], in_sigma, self.Extractor[j]
                            #print(claim_vector.shape)
                            #print(headline_vector.shape)
                            #print(sum(claim_vector))
                            #print(sum(headline_vector))
                            #print('\n')
                        gradient_iter = gradient_iter + temp_item

            gradient_H[i] = gradient_iter
        return gradient_H

    def compute_gradient_alpha(self):

        item_source = {}
        for i in self.source_evi.keys():
            for j in claim_label.keys():
                if j not in content_feature2:
                    continue
                if j not in item_source:
                    item_source[j] = {}
                item_source[j][i] = 0

        gradient_Alpha = {}
        for i in item_source.keys():
            gradient_Alpha[i] = {}
            gradient_Alpha[i]['glove'] = 0
            gradient_Alpha[i]['tfidf'] = 0
            if i not in content_feature2:
                continue

            claim_vector = tfidf_rep[content_feature2[i]]
            claim_emb = sent2emb(i)
            for j in item_source[i].keys():
                for k in self.source_evi[j].keys():
                    for ahl, ahb in self.source_evi[j][k]:
                        headline_vector = tfidf_rep[content_feature2[ahl]]
                        headline_emb = sent2emb(ahl)

                        sim = {}
                        sim['tfidf'] = 1 - cosine(claim_vector, headline_vector)
                        sim['glove'] = 1 - cosine(claim_emb, headline_emb)

                        for l in sim.keys():
                            if np.isnan(sim[l]):
                                sim[l] = 0

                        sigma_sim= 0
                        for l in self.Extractor[i].keys():
                            sigma_sim = sigma_sim + self.Extractor[i][l] * sim[l]
                        sigma_sim = expit(sigma_sim)

                        if k == self.item_val[i]:
                            bsyd = 1
                        else:
                            bsyd = 0

                        for l in self.Extractor[i].keys():
                            temp_grad = (1 - sigma_sim) * sim[l] * (bsyd - self.Accuracy[j] * sigma_sim)
                            temp_grad = float(temp_grad) / float(1 - self.Accuracy[j] * sigma_sim)
                            temp_grad = temp_grad * self.item_val_dis[i][k]
                            gradient_Alpha[i][l] = gradient_Alpha[i][l] + temp_grad

        return gradient_Alpha

    def EstepLayer(self):
        for i in self.source_evi.keys(): # source
            for j in self.source_evi[i].keys():
                for ahl, ahb in self.source_evi[i][j]:
                    article_vector = tfidf_rep[content_feature2[ahl]]
                    #article_vector = vectorize2.transform([ahl]).toarray().flatten()
                    article_emb = sent2emb(ahl)

                    for k in claim_label.keys(): # claim_id

                        if k not in content_feature2:
                            continue

                        claim_vector = tfidf_rep[content_feature2[k]]
                        claim_emb = sent2emb(k)

                        for l in self.item_val_dis[k].keys():
                            cos1 = 1 - cosine(claim_vector, article_vector)
                            cos2 = 1 - cosine(claim_emb, article_emb)
                            #print cos1, cos2

                            if np.isnan(cos1):
                                cos1 = 0
                            if np.isnan(cos2):
                                cos2 = 0

                            if l == j:
                                if self.item_val_dis[k][l] == -1:
                                    self.item_val_dis[k][l] = self.Accuracy[i] * expit(self.Extractor[k]['tfidf'] * cos1 + self.Extractor[k]['glove'] * cos2)
                                else:
                                    self.item_val_dis[k][l] = self.item_val_dis[k][l] * self.Accuracy[i] * expit(self.Extractor[k]['tfidf'] * cos1 + self.Extractor[k]['glove'] * cos2)
                            else:
                                if self.item_val_dis[k][l] == -1:
                                    self.item_val_dis[k][l] = float(1 - self.Accuracy[i] * expit(self.Extractor[k]['tfidf'] * cos1 + self.Extractor[k]['glove'] * cos2))
                                else:
                                    self.item_val_dis[k][l] = self.item_val_dis[k][l] * float(float(1) - self.Accuracy[i] * expit(self.Extractor[k]['tfidf'] * cos1 + self.Extractor[k]['glove'] * cos2))
                            #print self.item_val_dis[k][l]
        # Normalization
        #print self.item_val_dis
        exit()
        for i in self.item_val_dis.keys(): # item_id
            sum_total = 0
            for j in self.item_val_dis[i].keys():
                sum_total = sum_total + self.item_val_dis[i][j]
            if sum_total != 0:
                for j in self.item_val_dis[i].keys():
                    self.item_val_dis[i][j] = float(self.item_val_dis[i][j])/float(sum_total)

        for i in self.item_val_dis.keys():
            temp_rank = sorted(self.item_val_dis[i].items(), key = lambda d:d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]

    def MstepLayer(self, lr1, lr2, pre, ext_pre): # Update Accuracy
        for i in range(100):
            pre_accuracy = {}
            for m in self.Accuracy.keys():
                pre_accuracy[m] = self.Accuracy[m]
            gradient_H = self.compute_gradient_Hs()
            lr = lr1 / float(i + 1)
            for m in gradient_H.keys():
                self.Accuracy[m] = self.Accuracy[m] - float(lr) * gradient_H[m]
            delta_accuracy = {}
            for m in self.Accuracy.keys():
                delta_accuracy[m] = abs(pre_accuracy[m] - self.Accuracy[m])
            delta = sum(delta_accuracy.values())
            print("iteration source", i)
            if delta < pre:
                break
        for m in self.Accuracy.keys():
            if self.Accuracy[m] < 0:
                self.Accuracy[m] = 0
            if self.Accuracy[m] > 1:
                self.Accuracy[m] = 1

        for i in range(100):
            pre_alpha = {}
            for m in self.Extractor.keys():
                pre_alpha[m] = {}
                for n in self.Extractor[m].keys():
                    pre_alpha[m][n] = self.Extractor[m][n]

            gradient_alpha = self.compute_gradient_alpha()

            lr = lr2 / float(i + 1)
            for m in gradient_alpha.keys():
                for n in gradient_alpha[m].keys():
                    self.Extractor[m][n] = self.Extractor[m][n] - float(lr) * gradient_alpha[m][n]

            delta_extractor = {}
            for m in self.Extractor.keys():
                for n in self.Extractor[m].keys():
                    delta_extractor[m + ':' + n] = abs(self.Extractor[m][n] - pre_alpha[m][n])
            delta = sum(delta_extractor.values())

            if delta < ext_pre:
                break
            print("iteration extractor", i)

        #acc_sum = sum(self.Accuracy.values())
        #for i in self.Accuracy.keys():
        #    self.Accuracy[i] = expit(self.Accuracy[i])
        #for i in self.Extractor.keys():
        #    self.Extractor[i] = expit(self.Extractor[i])
        print(self.Accuracy)
        print(self.Extractor)

    def LCA(self):
        self.__initialization__()
        pre_accuracy = {}
        for i in self.Accuracy.keys():
            pre_accuracy[i] = 0.5
        lr1 = 0.0001
        lr1_b = 0.01
        lr2 = 0.1
        lr2_b = 1
        for iter in range(self.iteration_num):
            self.ini_item_val_dis()
            self.ini_item_val()
            self.EstepLayer()
            self.ini_source()

            for i in self.Extractor.keys():
                for j in self.Extractor[i].keys():
                    self.Extractor[i][j] = 0
            self.MstepLayer(lr1, lr2, lr1_b, lr2_b) #(0.0001, 0.1)
            print(data.precision(self.item_val))
        return self.item_val

class SimpleLCA_nosiy(TrustEvidence):

    def __init__(self, data, itr, gt):
        TrustEvidence.__init__(self, data, itr, gt)

    def __initialization__(self):
        TrustEvidence.__initialization__(self)

    def precision(self, result):
        return TrustEvidence.precision(self, result)

    def error_analysis(self, result):
        return TrustEvidence.error_analysis(self, result)

    def label_analysis(self, result):
        return TrustEvidence.label_analysis(self, result)

    def one_label_accuracy(self):
        return TrustEvidence.one_label_accuracy(self)

    def ini_item_val_dis(self):
        for i in self.item_val_dis.keys(): # claim_id
            for j in self.item_val_dis[i].keys():
                self.item_val_dis[i][j] = -1

    def ini_item_val(self):
        for i in self.item_val.keys(): # claim_id
            self.item_val[i] = ""

    def ini_source(self):
        for i in self.Accuracy.keys():
            self.Accuracy[i] = 0.6

    def EstepLayer(self):
        for i in self.source_item_claim_evi.keys(): # source
            for j in self.source_item_claim_evi[i].keys(): # claim_id
                evidence_number = 0
                for k in self.source_item_claim_evi[i][j].keys():
                    if k == 0:
                        continue
                    evidence_number = evidence_number + len(self.source_item_claim_evi[i][j][k])
                if evidence_number == 0:
                    continue
                for k in self.source_item_claim_evi[i][j].keys():
                    support_number = len(self.source_item_claim_evi[i][j][k])
                    for l in self.item_val_dis[j].keys():
                        if l == k:
                            if self.item_val_dis[j][l] == -1:
                                self.item_val_dis[j][l] = np.power(self.Accuracy[i], float(support_number)/float(evidence_number))
                            else:
                                self.item_val_dis[j][l] = self.item_val_dis[j][l] * np.power(self.Accuracy[i], float(support_number)/float(evidence_number))
                        else:
                            if self.item_val_dis[j][l] == -1:
                                self.item_val_dis[j][l] = np.power(float(1 - self.Accuracy[i]), float(support_number)/float(evidence_number))
                            else:
                                self.item_val_dis[j][l] = self.item_val_dis[j][l] * np.power(float(1 - self.Accuracy[i]), float(support_number)/float(evidence_number))

        #print(self.item_val_dis)
        # Normalization
        for i in self.item_val_dis.keys(): # item_id
            sum_total = 0
            for j in self.item_val_dis[i].keys():
                sum_total = sum_total + self.item_val_dis[i][j]
            if sum_total != 0:
                for j in self.item_val_dis[i].keys():
                    self.item_val_dis[i][j] = float(self.item_val_dis[i][j])/float(sum_total)
            self.item_val_dis[i][0] = 0


        for i in self.item_val_dis.keys():
            temp_rank = sorted(self.item_val_dis[i].items(), key = lambda d:d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]

    def MstepLayer(self): # Update Accuracy
        for i in self.source_item_claim_evi.keys():
            temp = 0
            temp_total = 0
            for j in self.source_item_claim_evi[i].keys():
                evidence_number = 0
                for k in self.source_item_claim_evi[i][j].keys():
                    if k != 0:
                        evidence_number = evidence_number + len(self.source_item_claim_evi[i][j][k])

                if evidence_number == 0:
                    temp_total = temp_total + 1 #evidence_number
                    continue

                for k in self.source_item_claim_evi[i][j].keys():
                    if k == self.item_val[j]:
                        temp = temp + float(len(self.source_item_claim_evi[i][j][k]))/float(evidence_number)
                temp_total = temp_total + 1 #evidence_number

            if temp_total != 0:
                self.Accuracy[i] = float(temp)/float(temp_total)
            else:
                self.Accuracy[i] = 0

    def SimLCA(self):
        self.__initialization__()
        for iter in range(self.iteration_num):
            self.ini_item_val_dis()
            self.ini_item_val()
            self.EstepLayer()
            self.ini_source()
            self.MstepLayer()
            print(self.precision(self.item_val))
        return self.item_val
        #print sorted(self.item_val.items())
        #for i in self.item_evi_dis.keys():
        #    temp = sorted(self.item_evi_dis[i].items(), key = lambda d:d[1], reverse = True)
        #    self.item_evi[i] = (temp[0][0], temp[1][0], temp[2][0], temp[3][0], temp[4][0])

        #return self.Accuracy, self.item_val, self.item_evi

class EvidenceLCA_noisy(SimpleLCA):

    def __init__(self, data, itr, gt):
        SimpleLCA.__init__(self, data, itr, gt)
        self.Extractor = {}
        self.Extractor['cfd'] = 1
        self.Extractor['tfidf'] = 1
        self.Extractor['glove'] = 1
        self.Extractor['entity'] = 1
        self.Extractor['const'] = 1

    def __initialization__(self):
        SimpleLCA.__initialization__(self)
        for i in self.Extractor.keys():
            self.Extractor[i] = 1

    def precision(self, result):
        return SimpleLCA.precision(self, result)

    def ini_item_val(self):
        SimpleLCA.ini_item_val(self)

    def ini_item_val_dis(self):
        SimpleLCA.ini_item_val_dis(self)

    def ini_source(self):
        SimpleLCA.ini_source(self)

    def compute_gradient_Hs(self):
        gradient_H = {}
        for i in self.source_item_claim_evi.keys():
            gradient_iter = 0
            for j in self.source_item_claim_evi[i].keys():
                claim_vector = tfidf_rep[content_feature2[j]]
                claim_ent = ent_rep[content_feature2[j]]
                claim_emb = emb_rep[content_feature2[j]]

                for k in self.source_item_claim_evi[i][j].keys():
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        headline_vector = tfidf_rep[content_feature2[evi]]
                        headline_ent = ent_rep[content_feature2[evi]]
                        headline_emb = emb_rep[content_feature2[evi]]

                        if k == self.item_val[j]:
                            bsyd = 1
                        else:
                            bsyd = 0

                        in_sigma = {}
                        in_sigma['tfidf'] = 1 - cosine(claim_vector, headline_vector)
                        in_sigma['entity'] = jd_sim(claim_ent, headline_ent)
                        in_sigma['glove'] = 1 - cosine(claim_emb, headline_emb)
                        in_sigma['cfd'] = escore
                        in_sigma['const'] = 1

                        for l in in_sigma.keys():
                            if np.isnan(in_sigma[l]):
                                in_sigma[l] = 0

                        sigma_sim = 0
                        for l in self.Extractor.keys():
                            sigma_sim = sigma_sim + self.Extractor[l] * float(in_sigma[l])

                        sigma_sim = expit(sigma_sim)

                        temp_item = bsyd - self.Accuracy[i] * sigma_sim
                    # if self.Accuracy[i] != 0:
                        #if float(self.Accuracy[i] * (1 - self.Accuracy[i] * sigma_sim)) == 0:
                        temp_item = float(temp_item) / (float((self.Accuracy[i] + 0.0000001) * (1 - (self.Accuracy[i] + 0.0000001) * sigma_sim)))
                        #else:
                        #    temp_item = float(temp_item) / float(self.Accuracy[i] * (1 - self.Accuracy[i] * sigma_sim))
                    # else:
                    #    temp_item = 0

                    # if self.item_val[j] == 1:
                    #    temp_item = temp_item
                    # else:
                    #    temp_item = 0
                        temp_item = temp_item * float(self.item_val_dis[j][k])

                        if np.isnan(temp_item):
                            temp_item = 0
                            print("Here nan")
                            #print self.Accuracy[i], in_sigma, self.Extractor[j]
                            print(claim_vector.shape)
                            print(headline_vector.shape)
                            print(sum(claim_vector))
                            print(sum(headline_vector))
                            print('\n')

                        gradient_iter = gradient_iter + temp_item
            # print gradient_iter
            gradient_H[i] = gradient_iter
        return gradient_H

    def compute_gradient_alpha(self):

        item_source = {}
        for i in self.source_item_claim_evi.keys():
            for j in self.source_item_claim_evi[i].keys():
                if j not in item_source:
                    item_source[j] = {}
                item_source[j][i] = 0

        gradient_Alpha = {}

        gradient_Alpha['glove'] = 0
        gradient_Alpha['tfidf'] = 0
        gradient_Alpha['entity'] = 0
        gradient_Alpha['cfd'] = 0
        gradient_Alpha['const'] = 0
        for i in item_source.keys():

            claim_vector = tfidf_rep[content_feature2[i]]
            claim_ent = ent_rep[content_feature2[i]]
            claim_emb = emb_rep[content_feature2[i]]

            for j in item_source[i].keys():
                for k in self.source_item_claim_evi[j][i].keys():
                    for evi, escore in self.source_item_claim_evi[j][i][k]:
                        headline_vector = tfidf_rep[content_feature2[evi]]
                        headline_ent = ent_rep[content_feature2[evi]]
                        headline_emb = emb_rep[content_feature2[evi]]

                        sim = {}
                        sim['tfidf'] = 1 - cosine(claim_vector, headline_vector)
                        sim['entity'] = jd_sim(claim_ent, headline_ent)
                        sim['glove'] = 1 - cosine(claim_emb, headline_emb)
                        sim['cfd'] = escore
                        sim['const'] = 1

                        for l in sim.keys():
                            if np.isnan(sim[l]):
                                sim[l] = 0

                        sigma_sim = 0
                        for l in self.Extractor.keys():
                            sigma_sim = sigma_sim + self.Extractor[l] * float(sim[l])
                        sigma_sim = expit(sigma_sim)

                        if k == self.item_val[i]:
                            bsyd = 1
                        else:
                            bsyd = 0

                        for l in self.Extractor.keys():
                            temp_grad = (1 - sigma_sim) * sim[l] * (bsyd - self.Accuracy[j] * sigma_sim)
                            temp_grad = float(temp_grad) / float(1 - self.Accuracy[j] * sigma_sim + 0.0000001)
                            temp_grad = temp_grad * self.item_val_dis[i][k]
                            gradient_Alpha[l] = gradient_Alpha[l] + temp_grad
        return gradient_Alpha

    def EstepLayer(self):
        for i in self.source_item_claim_evi.keys():  # source
            for j in self.source_item_claim_evi[i].keys():  # claim_id
                claim_vector = tfidf_rep[content_feature2[j]]
                claim_ent = ent_rep[content_feature2[j]]
                claim_emb = emb_rep[content_feature2[j]]

                for k in self.source_item_claim_evi[i][j].keys():
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        if evi not in content_feature2:
                            continue
                        headline_vector = tfidf_rep[content_feature2[evi]]
                        headline_emb = emb_rep[content_feature2[evi]]
                        headline_ent = ent_rep[content_feature2[evi]]

                        cos1 = 1 - cosine(claim_vector, headline_vector)
                        cos2 = 1 - cosine(claim_emb, headline_emb)
                        cos3 = jd_sim(claim_ent, headline_ent)

                        if np.isnan(cos1):
                            cos1 = 0
                        if np.isnan(cos2):
                            cos2 = 0

                        exp_score = expit(self.Extractor['tfidf'] * cos1 + self.Extractor['glove'] * cos2 + self.Extractor['entity'] * cos3 + \
                                          self.Extractor['cfd'] * float(escore) + self.Extractor['const'] * float(1))
                        for l in self.item_val_dis[j].keys():
                            if l == k:
                                if self.item_val_dis[j][l] == -1:
                                    self.item_val_dis[j][l] = np.log(self.Accuracy[i] * exp_score + 0.0000001)
                                else:
                                    self.item_val_dis[j][l] = self.item_val_dis[j][l] + np.log(self.Accuracy[i] * exp_score + 0.0000001)
                            else:
                                if self.item_val_dis[j][l] == -1:
                                    self.item_val_dis[j][l] = np.log(float(1 - self.Accuracy[i] * exp_score)/float(1000) + 0.0000001)
                                else:
                                    self.item_val_dis[j][l] = self.item_val_dis[j][l] + np.log(float(1 - self.Accuracy[i] * exp_score)/float(1000) + 0.0000001)
        # Normalization
        #print(self.item_val_dis)
        for i in self.item_val_dis.keys():  # item_id
            sum_total = 0
            for j in self.item_val_dis[i].keys():
                sum_total = sum_total + np.exp(self.item_val_dis[i][j])

            if sum_total != 0:
                for j in self.item_val_dis[i].keys():
                    self.item_val_dis[i][j] = np.exp(float(self.item_val_dis[i][j])) / float(sum_total)
            self.item_val_dis[i][0] = 0

        print(self.item_val_dis)

        for i in self.item_val_dis.keys():
            temp_rank = sorted(self.item_val_dis[i].items(), key=lambda d: d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]

    def MstepLayer_App(self, lr1, lr2, pre, ext_pre):  # Update Accuracy
        temp_total = {}
        for i in self.Accuracy.keys():
            temp_total[i] = 0
            self.Accuracy[i] = 0
        for i in self.source_item_claim_evi.keys():
            claim_number = 0
            for j in self.source_item_claim_evi[i].keys():
                if self.item_val[j] == 0:
                    continue
                claim_number = claim_number + 1
                tempk = {}
                totalk = 0
                for k in self.source_item_claim_evi[i][j].keys():
                    if k == 0:
                        continue
                    totalk = totalk + len(self.source_item_claim_evi[i][j][k])
                if totalk == 0:
                    continue

                for k in self.source_item_claim_evi[i][j].keys():
                    tempk[k] = float(len(self.source_item_claim_evi[i][j][k]))/float(totalk)
                    if k == self.item_val[j]:
                        self.Accuracy[i] = self.Accuracy[i] + 1#tempk[k]
            if claim_number == 0:
                continue
            self.Accuracy[i] = float(self.Accuracy[i])/float(claim_number)

        for i in range(100):
            pre_alpha = {}
            for m in self.Extractor.keys():
                pre_alpha[m] = {}
                for n in self.Extractor[m].keys():
                    pre_alpha[m][n] = self.Extractor[m][n]

            gradient_alpha = self.compute_gradient_alpha()

            lr = lr2 #/ float(i + 1)
            for m in gradient_alpha.keys():
                for n in gradient_alpha[m].keys():
                    self.Extractor[m][n] = self.Extractor[m][n] + float(lr) * gradient_alpha[m][n]

            delta_extractor = {}
            for m in self.Extractor.keys():
                for n in self.Extractor[m].keys():
                    delta_extractor[m + ':' + n] = abs(self.Extractor[m][n] - pre_alpha[m][n])
            delta = sum(delta_extractor.values())

            if delta < ext_pre:
                break
            print("iteration extractor", i)
        print(self.Accuracy)
        print(self.Extractor)

    def MstepLayer(self, lr1, lr2, pre, ext_pre):  # Update Accuracy
        for i in range(100):
            pre_accuracy = {}
            for m in self.Accuracy.keys():
                pre_accuracy[m] = self.Accuracy[m]
            gradient_H = self.compute_gradient_Hs()
            lr = lr1 #/ float(i + 1)
            for m in gradient_H.keys():
                self.Accuracy[m] = self.Accuracy[m] + float(lr) * gradient_H[m]
            delta_accuracy = {}
            for m in self.Accuracy.keys():
                delta_accuracy[m] = abs(pre_accuracy[m] - self.Accuracy[m])
            delta = sum(delta_accuracy.values())
            print("iteration source", i)
            if delta < pre:
                break
        for m in self.Accuracy.keys():
            if self.Accuracy[m] < 0:
                self.Accuracy[m] = 0
            if self.Accuracy[m] > 1:
                self.Accuracy[m] = 1

        for i in range(100):
            pre_alpha = {}
            for m in self.Extractor.keys():
                pre_alpha[m] = self.Extractor[m]

            gradient_alpha = self.compute_gradient_alpha()

            lr = lr2 #/ float(i + 1)
            for m in gradient_alpha.keys():
                self.Extractor[m] = self.Extractor[m] + float(lr) * gradient_alpha[m]

            delta_extractor = {}
            for m in self.Extractor.keys():
                delta_extractor[m] = abs(self.Extractor[m] - pre_alpha[m])
            delta = sum(delta_extractor.values())

            if delta < ext_pre:
                break
            print("iteration extractor", i)

        #for i in self.Extractor.keys():
        #    temp_sum = float(sum(self.Extractor[i].values()))
        #    if temp_sum != 0:
        #        for j in self.Extractor[i].keys():
        #            self.Extractor[i][j] = float(self.Extractor[i][j])/float(temp_sum)
        #    else:
        #        for j in self.Extractor[i].keys():
        #            self.Extractor[i][j] = float(self.Extractor[i][j])/float(len(self.Extractor[i]))


        # acc_sum = sum(self.Accuracy.values())
        # for i in self.Accuracy.keys():
        #    self.Accuracy[i] = expit(self.Accuracy[i])
        # for i in self.Extractor.keys():
        #    self.Extractor[i] = expit(self.Extractor[i])
        print(self.Accuracy)
        print(self.Extractor)

    def LCA(self):
        self.__initialization__()
        pre_accuracy = {}
        for i in self.Accuracy.keys():
            pre_accuracy[i] = 0.6
        lr1 = 0.0001
        lr1_b = 0.001
        lr2 = 0.1
        lr2_b = 0.001
        for iter in range(self.iteration_num):
            self.ini_item_val_dis()
            self.ini_item_val()
            self.EstepLayer()
            self.ini_source()
            for i in self.Extractor.keys():
                self.Extractor[i] = 1
            self.MstepLayer(lr1, lr2, lr1_b, lr2_b)  # (0.0001, 0.1)
            print(data.precision(self.item_val))
            evidencelca = []
            gt_trust = []
            for i in self.Accuracy.keys():
                evidencelca.append(self.Accuracy[i])
                gt_trust.append(source_trust_gt[i])

            # app_accuracy = {}
            # temp_total = {}
            # for i in self.Accuracy.keys():
            #     temp_total[i] = 0
            #     app_accuracy[i] = 0
            #
            # for i in self.source_item_claim_evi.keys():
            #     claim_number = 0
            #     for j in self.source_item_claim_evi[i].keys():
            #         if self.item_val[j] == 0:
            #             continue
            #         claim_number = claim_number + 1
            #         tempk = {}
            #         totalk = 0
            #         for k in self.source_item_claim_evi[i][j].keys():
            #             if k == 0:
            #                 continue
            #             totalk = totalk + len(self.source_item_claim_evi[i][j][k])
            #         if totalk == 0:
            #             continue
            #
            #         for k in self.source_item_claim_evi[i][j].keys():
            #             tempk[k] = float(len(self.source_item_claim_evi[i][j][k])) / float(totalk)
            #             if k == self.item_val[j]:
            #                 app_accuracy[i] = app_accuracy[i] + 1  # tempk[k]
            #     if claim_number == 0:
            #         continue
            #     app_accuracy[i] = float(app_accuracy[i]) / float(claim_number)
            #     for i in app_accuracy.keys():
            #         evidencelca.append(app_accuracy[i])
            #         gt_trust.append(source_trust_gt[i])

            print("pearson", spearmanr(evidencelca, gt_trust))

        return self.item_val

class EvidenceLCA_noisy_recall(SimpleLCA):

    def __init__(self, data1, data2, itr, gt, sgt):
        SimpleLCA.__init__(self, data1, data2, itr, gt, sgt)
        self.Recall = {}
        self.Qf = {}
        self.Extractor = {}
        self.Extractor['cfd'] = 0.5
        self.Extractor['tfidf'] = 0.5
        self.Extractor['glove'] = 0.5
        self.Extractor['entity'] = 0.5
        self.Extractor['const'] = 0.5
        for i in self.Accuracy.keys():
            self.Accuracy[i] = 0.6

    def __initialization__(self):
        SimpleLCA.__initialization__(self)
        for i in self.Accuracy.keys():
            self.Recall[i] = 0.1
        for i in self.Accuracy.keys():
            self.Qf[i] = 0.067
        for i in self.Extractor.keys():
            self.Extractor[i] = 0.5

    def precision(self, result):
        return SimpleLCA.precision(self, result)

    def spearsman_source(self):
        return SimpleLCA.spearsman_source(self)

    def pearson_source(self):
        return SimpleLCA.pearson_source(self)

    def ini_item_val(self):
        SimpleLCA.ini_item_val(self)

    def ini_item_val_dis(self):
        SimpleLCA.ini_item_val_dis(self)

    def ini_source(self):
        SimpleLCA.ini_source(self)
        for i in self.Accuracy.keys():
            self.Accuracy[i] = 0.6
        for i in self.Recall.keys():
            self.Recall[i] = 0.1
        for i in self.Qf.keys():
            self.Qf[i] = 0.067

    def compute_gradient_Hs(self):
        gradient_H = {}
        for i in self.source_claim.keys():
            gradient_iter = 0
            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    if j == self.item_val[k]:
                        bsyd = 1
                    else:
                        bsyd = 0
                    gd = self.item_val_dis[k][j] * float(bsyd) / float(self.Accuracy[i]) + float(bsyd - 1) / float(1 - float(self.Accuracy[i]))

                    constraints = np.log((self.Recall[i] + smooth_para) / float(self.Recall[i] + self.Qf[i] + smooth_para)) - float(1) - np.log(self.Accuracy[i] + smooth_para)
                    if np.isnan(constraints):
                        constraints = 0
                    gradient_iter = gradient_iter + gd + const_para * float(constraints)
            # print gradient_iter
            gradient_H[i] = gradient_iter
        return gradient_H

    def compute_gradient_Ps(self):
        gradient_H = {}
        for i in self.source_item_claim_evi.keys():
            gradient_iter = 0
            for j in self.source_item_claim_evi[i].keys():
                claim_vector = tfidf_rep[content_feature2[j]]
                claim_ent = ent_rep[content_feature2[j]]
                claim_emb = emb_rep[content_feature2[j]]

                for k in self.source_item_claim_evi[i][j].keys():
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        headline_vector = tfidf_rep[content_feature2[evi]]
                        headline_ent = ent_rep[content_feature2[evi]]
                        headline_emb = emb_rep[content_feature2[evi]]

                        if k == self.item_val[j]:
                            bsyd = 1
                        else:
                            bsyd = 0

                        in_sigma = {}
                        in_sigma['tfidf'] = 1 - cosine(claim_vector, headline_vector)
                        in_sigma['entity'] = jd_sim(claim_ent, headline_ent)
                        in_sigma['glove'] = 1 - cosine(claim_emb, headline_emb)
                        in_sigma['cfd'] = escore
                        in_sigma['const'] = 1

                        for l in in_sigma.keys():
                            if np.isnan(in_sigma[l]):
                                in_sigma[l] = 0

                        sigma_sim = 0
                        for l in self.Extractor.keys():
                            sigma_sim = sigma_sim + self.Extractor[l] * float(in_sigma[l])

                        sigma_sim = expit(sigma_sim)

                        temp_item = bsyd - self.Accuracy[i] * sigma_sim
                    # if self.Accuracy[i] != 0:
                        #if float(self.Accuracy[i] * (1 - self.Accuracy[i] * sigma_sim)) == 0:
                        temp_item = float(temp_item) / (float((self.Accuracy[i] + 0.0000001) * (1 - (self.Accuracy[i] + 0.0000001) * sigma_sim)))
                        #else:
                        #    temp_item = float(temp_item) / float(self.Accuracy[i] * (1 - self.Accuracy[i] * sigma_sim))
                    # else:
                    #    temp_item = 0

                    # if self.item_val[j] == 1:
                    #    temp_item = temp_item
                    # else:
                    #    temp_item = 0
                        temp_item = temp_item * float(self.item_val_dis[j][k])

                        if np.isnan(temp_item):
                            temp_item = 0
                            print("Here nan")
                            #print self.Accuracy[i], in_sigma, self.Extractor[j]
                            print(claim_vector.shape)
                            print(headline_vector.shape)
                            print(sum(claim_vector))
                            print(sum(headline_vector))
                            print('\n')

                        gradient_iter = gradient_iter + temp_item
            # print gradient_iter
            gradient_H[i] = gradient_iter
        return gradient_H

    def compute_gradient_alpha(self):
        Acc_Temp = {}
        for i in self.Accuracy.keys():
            if self.Recall[i] + self.Qf[i] == 0:
                Acc_Temp[i] = 0
            else:
                Acc_Temp[i] = float(self.Recall[i])/float(self.Recall[i] + self.Qf[i])

        gradient_Alpha = {}
        gradient_Alpha['glove'] = 0
        gradient_Alpha['tfidf'] = 0
        gradient_Alpha['entity'] = 0
        gradient_Alpha['cfd'] = 0
        gradient_Alpha['const'] = 0

        for ft in gradient_Alpha.keys():
            for i in self.source_item_claim_evi.keys():
                for j in self.source_item_claim_evi[i].keys():
                    claim_vector = tfidf_rep[content_feature2[j]]
                    claim_ent = ent_rep[content_feature2[j]]
                    claim_emb = emb_rep[content_feature2[j]]
                    for k in self.source_item_claim_evi[i][j].keys():
                        if k == self.item_val[j]:
                            bsyd = 1
                        else:
                            bsyd = 0
                        for evi, escore in self.source_item_claim_evi[i][j][k]:
                            if evi not in content_feature2:
                                continue

                            headline_vector = tfidf_rep[content_feature2[evi]]
                            headline_emb = emb_rep[content_feature2[evi]]
                            headline_ent = ent_rep[content_feature2[evi]]

                            distance_all = {}
                            cos1 = 1 - cosine(claim_vector, headline_vector)
                            cos2 = 1 - cosine(claim_emb, headline_emb)
                            cos3 = jd_sim(claim_ent, headline_ent)
                            distance_all['tfidf'] = cos1
                            distance_all['glove'] = cos2
                            distance_all['entity'] = cos3
                            distance_all['cfd'] = float(escore)
                            distance_all['const'] = float(1)

                            if np.isnan(cos1):
                                cos1 = 0
                            if np.isnan(cos2):
                                cos2 = 0
                            if np.isnan(cos3):
                                cos3 = 0

                            exp_score = expit(self.Extractor['tfidf'] * cos1 + self.Extractor['glove'] * cos2 + self.Extractor['entity'] * cos3 + \
                                          self.Extractor['cfd'] * float(escore) + self.Extractor['const'] * float(1))

                            item1 = float(bsyd) * float(1-exp_score) * distance_all[ft]
                            item2 = float(1-bsyd) * float(Acc_Temp[i])/float(float(Acc_Temp[i]) * float(exp_score) - float(1)) * float(1-exp_score) * float(distance_all[ft])
                            #print(item1, item2, float(self.item_val_dis[j][k]) * (float(item1) + float(item2)))
                            gradient_Alpha[ft] = gradient_Alpha[ft] + float(self.item_val_dis[j][k]) * (float(item1) + float(item2))

        return gradient_Alpha

    def compute_gradient_Qs(self):
        gradient_Q = {}
        for i in self.source_item_claim_evi.keys():
            gradient_iter = 0
            for j in self.source_item_claim_evi[i].keys():
                claim_vector = tfidf_rep[content_feature2[j]]
                claim_ent = ent_rep[content_feature2[j]]
                claim_emb = emb_rep[content_feature2[j]]

                for k in self.source_item_claim_evi[i][j].keys():
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        headline_vector = tfidf_rep[content_feature2[evi]]
                        headline_ent = ent_rep[content_feature2[evi]]
                        headline_emb = emb_rep[content_feature2[evi]]

                        if k == self.item_val[j]:
                            bsyd = 1
                        else:
                            bsyd = 0

                        in_sigma = {}
                        cos1 = 1 - cosine(claim_vector, headline_vector)
                        cos2 = jd_sim(claim_ent, headline_ent)
                        cos3 = 1 - cosine(claim_emb, headline_emb)

                        if np.isnan(cos1):
                            cos1 = 0
                        if np.isnan(cos2):
                            cos2 = 0
                        if np.isnan(cos3):
                            cos3 = 0

                        in_sigma['tfidf'] = cos1
                        in_sigma['entity'] = cos2
                        in_sigma['glove'] = cos3
                        in_sigma['cfd'] = escore
                        in_sigma['const'] = 1

                        for l in in_sigma.keys():
                            if np.isnan(in_sigma[l]):
                                in_sigma[l] = 0

                        sigma_sim = 0
                        for l in self.Extractor.keys():
                            sigma_sim = sigma_sim + self.Extractor[l] * float(in_sigma[l])
                        sigma_sim = expit(sigma_sim)

                        QR = float(self.Qf[i] + self.Recall[i])
                        item1 = -float(self.Recall[i] * sigma_sim)
                        item2 = self.Recall[i] * sigma_sim * QR - np.power(QR, 2)
                        gradient_iter = gradient_iter + self.item_val_dis[j][k] * float(1 - bsyd) * float(item1)/float(item2)
            constraints =  - self.Accuracy[i] / float(self.Qf[i] + self.Recall[i])
            gradient_Q[i] = gradient_iter + float(const_para) * constraints
        return gradient_Q

    def compute_gradient_Rs(self):
        gradient_R = {}
        for i in self.source_item_claim_evi.keys():
            gradient_iter = 0
            for j in self.source_item_claim_evi[i].keys():
                claim_vector = tfidf_rep[content_feature2[j]]
                claim_ent = ent_rep[content_feature2[j]]
                claim_emb = emb_rep[content_feature2[j]]

                for k in self.source_item_claim_evi[i][j].keys():
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        headline_vector = tfidf_rep[content_feature2[evi]]
                        headline_ent = ent_rep[content_feature2[evi]]
                        headline_emb = emb_rep[content_feature2[evi]]

                        if k == self.item_val[j]:
                            bsyd = 1
                        else:
                            bsyd = 0

                        cos1 = 1 - cosine(claim_vector, headline_vector)
                        cos2 = jd_sim(claim_ent, headline_ent)
                        cos3 = 1 - cosine(claim_emb, headline_emb)

                        if np.isnan(cos1):
                            cos1 = 0
                        if np.isnan(cos2):
                            cos2 = 0
                        if np.isnan(cos3):
                            cos3 = 0

                        in_sigma = {}
                        in_sigma['tfidf'] = cos1
                        in_sigma['entity'] = cos2
                        in_sigma['glove'] = cos3
                        in_sigma['cfd'] = escore
                        in_sigma['const'] = 1

                        for l in in_sigma.keys():
                            if np.isnan(in_sigma[l]):
                                in_sigma[l] = 0

                        sigma_sim = 0
                        for l in self.Extractor.keys():
                            sigma_sim = sigma_sim + self.Extractor[l] * float(in_sigma[l])

                        sigma_sim = expit(sigma_sim)

                        QR = float(self.Qf[i] + self.Recall[i])

                        item1 = float(bsyd)/float(self.Recall[i])
                        item2u = float(1 - bsyd) * self.Qf[i] * sigma_sim
                        item2b = self.Recall[i] * sigma_sim * QR - np.power(QR, 2)

                        gradient_iter = gradient_iter + self.item_val_dis[j][k] * (item1 + float(item2u)/float(item2b))

            #if self.Accuracy[i] != 0:
            constraints = self.Accuracy[i] * self.Qf[i] / (float(self.Recall[i]) * float(self.Recall[i] + self.Qf[i]))#1 + np.log(self.Recall[i]) - float(self.Recall[i])/float(self.Accuracy[i]) - np.log(self.Accuracy[i])
            gradient_R[i] = gradient_iter + float(const_para) * constraints
        return gradient_R

    def MstepLayer_Grad(self, lr1, lr2, lr3, lr1_b, lr2_b, lr3_b):  # Update Accuracy

        # for i in self.source_claim.keys():
        #     acc = 0
        #     temp = 0
        #     for j in self.source_claim[i].keys():
        #         for k in self.source_claim[i][j]:
        #             if j == self.item_val[k]:
        #                 acc = acc + 1#self.item_val_dis[k][j]
        #             temp = temp + 1
        #     if temp == 0:
        #         self.Accuracy[i] = 0
        #     else:
        #         self.Accuracy[i] = float(acc) / float(temp)

        for i in range(50):
            pre_acc = {}
            for m in self.Accuracy.keys():
                pre_acc[m] = self.Accuracy[m]
            gradient_A = self.compute_gradient_Hs()

            lr = lr1
            for m in gradient_A.keys():
                self.Accuracy[m] = self.Accuracy[m] + float(lr) * gradient_A[m]
        #print(self.Accuracy)

        for i in range(50):
            pre_recall = {}
            for m in self.Recall.keys():
                pre_recall[m] = self.Recall[m]
            gradient_R = self.compute_gradient_Rs()

            lr = lr1
            for m in gradient_R.keys():
                self.Recall[m] = self.Recall[m] + float(lr) * gradient_R[m]

            #delta_extractor = {}
            #for m in self.Recall.keys():
            #    delta_extractor[m] = abs(self.Recall[m] - pre_recall[m])
            #delta = sum(delta_extractor.values())

            #if delta < lr1_b:
            #    break

        for i in range(50):
            pre_q = {}
            for m in self.Qf.keys():
                pre_q[m] = self.Qf[m]
            gradient_Q = self.compute_gradient_Qs()
            lr = lr3
            for m in gradient_Q.keys():
                self.Qf[m] = self.Qf[m] + float(lr) * gradient_Q[m]

            #delta_extractor = {}
            #for m in self.Recall.keys():
            #    delta_extractor[m] = abs(self.Qf[m] - pre_q[m])
            #delta = sum(delta_extractor.values())

            #if delta < lr3_b:
            #    break

        for i in range(30):
            pre_alpha = {}
            for m in self.Extractor.keys():
                pre_alpha[m] = self.Extractor[m]

            gradient_alpha = self.compute_gradient_alpha()

            lr = lr2 #/ float(i + 1)
            for m in gradient_alpha.keys():
                self.Extractor[m] = self.Extractor[m] + float(lr) * gradient_alpha[m]
            #print(self.Extractor)

            #delta_extractor = {}
            #for m in self.Extractor.keys():
            #    delta_extractor[m] = abs(self.Extractor[m] - pre_alpha[m])
            #delta = sum(delta_extractor.values())

            #if delta < lr2_b:
            #    break
            #print("iteration extractor", i)
        print("Interation:\n")
        print(self.Recall)
        print(self.Qf)
        print(self.Extractor)

    def EstepLayer_Grad(self):
        Acc_Extractor = {}
        for i in self.Recall.keys():
            if self.Qf[i] + self.Recall[i] == 0:
                Acc_Extractor[i] = 0
            else:
                Acc_Extractor[i] = float(self.Recall[i])/float(self.Recall[i] + self.Qf[i])

        for i in self.source_item_claim_evi.keys():  # source
            for j in self.source_item_claim_evi[i].keys():  # claim_id
                claim_vector = tfidf_rep[content_feature2[j]]
                claim_ent = ent_rep[content_feature2[j]]
                claim_emb = emb_rep[content_feature2[j]]

                for k in self.source_item_claim_evi[i][j].keys():
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        if evi not in content_feature2:
                            continue
                        headline_vector = tfidf_rep[content_feature2[evi]]
                        headline_emb = emb_rep[content_feature2[evi]]
                        headline_ent = ent_rep[content_feature2[evi]]

                        cos1 = 1 - cosine(claim_vector, headline_vector)
                        cos2 = 1 - cosine(claim_emb, headline_emb)
                        cos3 = jd_sim(claim_ent, headline_ent)

                        if np.isnan(cos1):
                            cos1 = 0
                        if np.isnan(cos2):
                            cos2 = 0
                        if np.isnan(cos3):
                            cos3 = 0

                        exp_total = self.Extractor['tfidf'] * cos1 + self.Extractor['glove'] * cos2 + self.Extractor[
                                'entity'] * cos3 + self.Extractor['cfd'] * float(escore) + self.Extractor['const'] * float(1)

                        exp_score = expit(exp_total)

                        for l in self.item_val_dis[j].keys():
                            if l == k:
                                if self.item_val_dis[j][l] == -1:
                                    self.item_val_dis[j][l] = cweights * np.log(self.Recall[i] * exp_score + smooth_para)
                                else:
                                    self.item_val_dis[j][l] = self.item_val_dis[j][l] + cweights * np.log(self.Recall[i] * exp_score + smooth_para)
                            else:
                                if self.item_val_dis[j][l] == -1:
                                    self.item_val_dis[j][l] = cweights * np.log(float(1 - Acc_Extractor[i] * exp_score) / float(self.total_claim) + smooth_para)
                                else:
                                    self.item_val_dis[j][l] = self.item_val_dis[j][l] + cweights * np.log(float(1 - Acc_Extractor[i] * exp_score)/ \
                                                                                               float(self.total_claim) + smooth_para)
        # Normalization
        for i in self.item_val_dis.keys():
            temp_rank = sorted(self.item_val_dis[i].items(), key=lambda d: d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]

        #print(self.precision(self.item_val))

        for i in self.source_claim.keys():
            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    for l in self.item_val_dis[k].keys():
                        if l == j:
                            if self.item_val_dis[k][l] == -1:
                                self.item_val_dis[k][l] = float(1 - cweights) * np.log(self.Accuracy[i] + smooth_para)
                            else:
                                self.item_val_dis[k][l] = self.item_val_dis[k][l] + float(1 - cweights) * np.log(self.Accuracy[i] + smooth_para)
                        else:
                            if self.item_val_dis[k][l] == -1:
                                self.item_val_dis[k][l] = float(1 - cweights) * np.log(float(float(1) - self.Accuracy[i]) + smooth_para) #/float(self.total_claim) + smooth_para)
                            else:
                                self.item_val_dis[k][l] = self.item_val_dis[k][l] + float(1 - cweights) * np.log(float(float(1) - self.Accuracy[i]) + smooth_para) #/float(self.total_claim) + smooth_para)


        for i in self.item_val_dis.keys():  # item_id
            sum_total = 0
            for j in self.item_val_dis[i].keys():
                if self.item_val_dis[i][j] != -1:
                    sum_total = sum_total + np.exp(self.item_val_dis[i][j])

            if sum_total != 0:
                for j in self.item_val_dis[i].keys():
                    if self.item_val_dis[i][j] != -1:
                        self.item_val_dis[i][j] = np.exp(float(self.item_val_dis[i][j])) / float(sum_total)
            self.item_val_dis[i][0] = 0

        for i in self.item_val_dis.keys():
            temp_rank = sorted(self.item_val_dis[i].items(), key=lambda d: d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]

        #print(self.precision(self.item_val))
        #print("\n")

    def LCA(self):
        self.__initialization__()
        #print(self.source_item_claim_evi)
        #exit()
        pre_accuracy = {}
        for i in self.Accuracy.keys():
            pre_accuracy[i] = 0.6

        lr1 = 0.00001
        lr1_b = 0.001
        lr2 = 0.0001
        lr2_b = 0.001
        lr3 = 0.00001
        lr3_b = 0.001

        for iter in range(self.iteration_num):
            self.ini_item_val_dis()
            self.ini_item_val()
            self.EstepLayer_Grad()
            self.ini_source()
            for i in self.Extractor.keys():
                self.Extractor[i] = 0.5
            self.MstepLayer_Grad(lr1, lr2, lr1_b, lr2_b, lr3, lr3_b)  # (0.0001, 0.1)
            for i in self.Recall.keys():
                if self.Recall[i] < 0:
                    self.Recall[i] = 0
                if self.Recall[i] > 1:
                    self.Recall[i] = 1
            for i in self.Qf.keys():
                if self.Qf[i] < 0:
                    self.Qf[i] = 0
                if self.Qf[i] > 1:
                    self.Qf[i] = 1
            for i in self.Accuracy.keys():
                if self.Accuracy[i] < 0:
                    self.Accuracy[i] = 0
                if self.Accuracy[i] > 1:
                    self.Accuracy[i] = 1
            #print(self.pearson_source())
            #print(self.spearsman_source())
        # acc_temp = {}
        # total_temp = {}
        # for i in self.source_claim.keys():
        #     if i not in acc_temp:
        #         acc_temp[i] = 0
        #     if i not in total_temp:
        #         total_temp[i] = 0
        #
        #     for j in self.source_claim[i].keys():
        #         for k in self.source_claim[i][j]:
        #             if j == self.item_val[k]:
        #                 acc_temp[i] = acc_temp[i] + 1
        #
        #             total_temp[i] = total_temp[i] + 1
        # for i in acc_temp.keys():
        #     if total_temp[i] == 0:
        #         self.Accuracy[i] = 0
        #     else:
        #         self.Accuracy[i] = float(acc_temp[i])/float(total_temp[i])




            #print(data.precision(self.item_val))
            #print(data.spearsman_source())
            #print(data.pearson_source())


        rank_data = []
        # for i in self.source_item_claim_evi.keys():
        #      for j in self.source_item_claim_evi[i].keys():
        #          claim_vector = tfidf_rep[content_feature2[j]]
        #          claim_ent = ent_rep[content_feature2[j]]
        #          claim_emb = emb_rep[content_feature2[j]]
        #
        #          for k in self.source_item_claim_evi[i][j].keys():
        #              if k != self.item_val[j] and k!=0:
        #                  continue
        #              for evi, escore in self.source_item_claim_evi[i][j][k]:
        #                  if evi not in content_feature2:
        #                      continue
        #                  headline_vector = tfidf_rep[content_feature2[evi]]
        #                  headline_emb = emb_rep[content_feature2[evi]]
        #                  headline_ent = ent_rep[content_feature2[evi]]
        #
        #                  cos1 = 1 - cosine(claim_vector, headline_vector)
        #                  cos2 = 1 - cosine(claim_emb, headline_emb)
        #                  cos3 = jd_sim(claim_ent, headline_ent)
        #
        #                  if np.isnan(cos1):
        #                      cos1 = 0
        #                  if np.isnan(cos2):
        #                      cos2 = 0
        #                  if np.isnan(cos3):
        #                      cos3 = 0
        # #
        #                  exp_score = expit(
        #                      self.Extractor['tfidf'] * cos1 + self.Extractor['glove'] * cos2 + self.Extractor[
        #                          'entity'] * cos3 + \
        #                      self.Extractor['cfd'] * float(escore) + self.Extractor['const'] * float(1))
        #                  rank_data.append((evi, j, k, exp_score))
        # rank_data = sorted(rank_data, key = lambda d:d[1], reverse=True)
        # if len(rank_data) > 1000:
        #      rank_data_1000 = rank_data[:1000]
        # else:
        #      rank_data_1000 = rank_data

        return self.item_val, rank_data

class EvidenceLCA(SimpleLCA):

    def __init__(self, data, itr, gt):
        SimpleLCA.__init__(self, data, itr, gt)
        self.Extractor = {}
        for i in self.source_item_claim_evi.keys():
            for j in self.source_item_claim_evi[i].keys():
                if j not in content_feature2:
                    continue
                self.Extractor[j] = {}
                self.Extractor[j]['tfidf'] = 0
                self.Extractor[j]['glove'] = 0

    def __initialization__(self):
        SimpleLCA.__initialization__(self)
        for i in self.Extractor.keys():
            for j in self.Extractor[i].keys():
                self.Extractor[i][j] = 1

    def precision(self, result):
        return SimpleLCA.precision(self, result)

    def ini_item_val(self):
        SimpleLCA.ini_item_val(self)

    def ini_item_val_dis(self):
        SimpleLCA.ini_item_val_dis(self)

    def ini_source(self):
        SimpleLCA.ini_source(self)

    def compute_gradient_Hs(self):
        gradient_H = {}
        for i in self.source_item_claim_evi.keys():
            gradient_iter = 0
            for j in self.source_item_claim_evi[i].keys():
                claim_vector = vectorize.transform([j]).toarray().flatten()
                claim_emb = sent2emb(j)
                for k in self.source_item_claim_evi[i][j].keys():
                    headline_vector = vectorize.transform(
                        [self.source_item_claim_evi[i][j][k][0][0]]).toarray().flatten()
                    headline_emb = sent2emb(self.source_item_claim_evi[i][j][k][0][0])
                    if k == self.item_val[j]:
                        bsyd = 1
                    else:
                        bsyd = 0

                    in_sigma = {}
                    in_sigma['tfidf'] = 1 - cosine(claim_vector, headline_vector)
                    in_sigma['glove'] = 1 - cosine(claim_emb, headline_emb)

                    for l in in_sigma.keys():
                        if np.isnan(in_sigma[l]):
                            in_sigma[l] = 0

                    sigma_sim = 0
                    for l in self.Extractor[j].keys():
                        sigma_sim = sigma_sim + self.Extractor[j][l] * in_sigma[l]

                    sigma_sim = expit(sigma_sim)

                    temp_item = bsyd - self.Accuracy[i] * sigma_sim
                    # if self.Accuracy[i] != 0:
                    if float(self.Accuracy[i] * (1 - self.Accuracy[i] * sigma_sim)) == 0:
                        temp_item = float(temp_item) / (
                        float((self.Accuracy[i] + 0.00001) * (1 - (self.Accuracy[i] + 0.00001) * sigma_sim)))
                    else:
                        temp_item = float(temp_item) / float(self.Accuracy[i] * (1 - self.Accuracy[i] * sigma_sim))
                    # else:
                    #    temp_item = 0

                    # if self.item_val[j] == 1:
                    #    temp_item = temp_item
                    # else:
                    #    temp_item = 0
                    temp_item = temp_item * float(self.item_val_dis[j][k])

                    if np.isnan(temp_item):
                        temp_item = 0
                        #print "Here nan"
                        #print self.Accuracy[i], in_sigma, self.Extractor[j]
                        print(claim_vector.shape)
                        print(headline_vector.shape)
                        print(sum(claim_vector))
                        print(sum(headline_vector))
                        print('\n')
                    gradient_iter = gradient_iter + temp_item
            # print gradient_iter
            gradient_H[i] = gradient_iter
        return gradient_H

    def compute_gradient_alpha(self):

        item_source = {}
        for i in self.source_item_claim_evi.keys():
            for j in self.source_item_claim_evi[i].keys():
                if j not in item_source:
                    item_source[j] = {}
                item_source[j][i] = 0

        gradient_Alpha = {}
        for i in item_source.keys():
            gradient_Alpha[i] = {}
            gradient_Alpha[i]['glove'] = 0
            gradient_Alpha[i]['tfidf'] = 0

            claim_vector = vectorize.transform([i]).toarray().flatten()
            claim_emb = sent2emb(i)
            for j in item_source[i].keys():
                for k in self.source_item_claim_evi[j][i].keys():
                    headline_vector = vectorize.transform(
                        [self.source_item_claim_evi[j][i][k][0][0]]).toarray().flatten()
                    headline_emb = sent2emb(self.source_item_claim_evi[j][i][k][0][0])

                    sim = {}
                    sim['tfidf'] = 1 - cosine(claim_vector, headline_vector)
                    sim['glove'] = 1 - cosine(claim_emb, headline_emb)

                    for l in sim.keys():
                        if np.isnan(sim[l]):
                            sim[l] = 0

                    sigma_sim = 0
                    for l in self.Extractor[i].keys():
                        sigma_sim = sigma_sim + self.Extractor[i][l] * sim[l]
                    sigma_sim = expit(sigma_sim)

                    if k == self.item_val[i]:
                        bsyd = 1
                    else:
                        bsyd = 0

                    for l in self.Extractor[i].keys():
                        temp_grad = (1 - sigma_sim) * sim[l] * (bsyd - self.Accuracy[j] * sigma_sim)
                        temp_grad = float(temp_grad) / float(1 - self.Accuracy[j] * sigma_sim)
                        temp_grad = temp_grad * self.item_val_dis[i][k]
                        gradient_Alpha[i][l] = gradient_Alpha[i][l] + temp_grad
        return gradient_Alpha

    def EstepLayer(self):
        for i in self.source_item_claim_evi.keys():  # source
            for j in self.source_item_claim_evi[i].keys():  # claim_id
                claim_vector = vectorize.transform([j]).toarray().flatten()
                claim_emb = sent2emb(j)
                for k in self.source_item_claim_evi[i][j].keys():
                    for l in self.item_val_dis[j].keys():
                        headline_vector = vectorize.transform(
                            [self.source_item_claim_evi[i][j][k][0][0]]).toarray().flatten()
                        # print self.source_item_claim_evi[i][j][k][0][0]
                        headline_emb = sent2emb(self.source_item_claim_evi[i][j][k][0][0])
                        cos1 = 1 - cosine(claim_vector, headline_vector)
                        cos2 = 1 - cosine(claim_emb, headline_emb)

                        if np.isnan(cos1):
                            cos1 = 0
                        if np.isnan(cos2):
                            cos2 = 0

                        if l == k:
                            if self.item_val_dis[j][l] == -1:
                                self.item_val_dis[j][l] = self.Accuracy[i] * expit(self.Extractor[j]['tfidf'] * cos1 + self.Extractor[j]['glove'] * cos2)
                            else:
                                self.item_val_dis[j][l] = self.item_val_dis[j][l] * self.Accuracy[i] * expit(self.Extractor[j]['tfidf'] * cos1 + self.Extractor[j]['glove'] * cos2)
                        else:
                            if self.item_val_dis[j][l] == -1:
                                self.item_val_dis[j][l] = float(1 - self.Accuracy[i] * expit(self.Extractor[j]['tfidf'] * cos1 + self.Extractor[j]['glove'] * cos2))
                            else:
                                self.item_val_dis[j][l] = self.item_val_dis[j][l] * float(1 - self.Accuracy[i] * expit(self.Extractor[j]['tfidf'] * cos1 + self.Extractor[j]['glove'] * cos2))
        # Normalization
        for i in self.item_val_dis.keys():  # item_id
            sum_total = 0
            for j in self.item_val_dis[i].keys():
                sum_total = sum_total + self.item_val_dis[i][j]
            if sum_total != 0:
                for j in self.item_val_dis[i].keys():
                    self.item_val_dis[i][j] = float(self.item_val_dis[i][j]) / float(sum_total)

        for i in self.item_val_dis.keys():
            temp_rank = sorted(self.item_val_dis[i].items(), key=lambda d: d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]

    def MstepLayer(self, lr1, lr2, pre, ext_pre):  # Update Accuracy
        for i in range(100):
            pre_accuracy = {}
            for m in self.Accuracy.keys():
                pre_accuracy[m] = self.Accuracy[m]
            gradient_H = self.compute_gradient_Hs()
            lr = lr1 / float(i + 1)
            for m in gradient_H.keys():
                self.Accuracy[m] = self.Accuracy[m] - float(lr) * gradient_H[m]
            delta_accuracy = {}
            for m in self.Accuracy.keys():
                delta_accuracy[m] = abs(pre_accuracy[m] - self.Accuracy[m])
            delta = sum(delta_accuracy.values())
            print("iteration source", i)
            if delta < pre:
                break
        for m in self.Accuracy.keys():
            if self.Accuracy[m] < 0:
                self.Accuracy[m] = 0
            if self.Accuracy[m] > 1:
                self.Accuracy[m] = 1

        for i in range(100):
            pre_alpha = {}
            for m in self.Extractor.keys():
                pre_alpha[m] = {}
                for n in self.Extractor[m].keys():
                    pre_alpha[m][n] = self.Extractor[m][n]

            gradient_alpha = self.compute_gradient_alpha()

            lr = lr2 / float(i + 1)
            for m in gradient_alpha.keys():
                for n in gradient_alpha[m].keys():
                    self.Extractor[m][n] = self.Extractor[m][n] - float(lr) * gradient_alpha[m][n]

            delta_extractor = {}
            for m in self.Extractor.keys():
                for n in self.Extractor[m].keys():
                    delta_extractor[m + ':' + n] = abs(self.Extractor[m][n] - pre_alpha[m][n])
            delta = sum(delta_extractor.values())

            if delta < ext_pre:
                break
            print("iteration extractor", i)

        # acc_sum = sum(self.Accuracy.values())
        # for i in self.Accuracy.keys():
        #    self.Accuracy[i] = expit(self.Accuracy[i])
        # for i in self.Extractor.keys():
        #    self.Extractor[i] = expit(self.Extractor[i])
        #print(self.Accuracy)
        #print(self.Extractor)

    def LCA(self):
        self.__initialization__()
        pre_accuracy = {}
        for i in self.Accuracy.keys():
            pre_accuracy[i] = 0.5
        lr1 = 0.001
        lr1_b = 0.1
        lr2 = 0.1
        lr2_b = 1
        for iter in range(self.iteration_num):
            self.ini_item_val_dis()
            self.ini_item_val()
            self.EstepLayer()
            self.ini_source()

            for i in self.Extractor.keys():
                for j in self.Extractor[i].keys():
                    self.Extractor[i][j] = 0
            self.MstepLayer(lr1, lr2, lr1_b, lr2_b)  # (0.0001, 0.1)
            #print(data.precision(self.item_val))
        return self.item_val

#data = SimpleLCA_nosiy(source_claim_evidence_noisy, 100, claim_label) #data, iteration_number, ground_truth
data = SimpleLCA(source_claim, source_claim_evidence_noisy, 100, claim_label, source_truthworthiness)
result = data.SimLCA()
print('Simple LCA: ')
print('precision: ', data.precision(result))
print('spearsman: ', data.spearsman_source())
print('pearson: ', data.pearson_source())

#print(data.one_label_accuracy())
#print(data.precision(result))
#print(data.label_analysis(result))
#print(data.error_analysis(result))
#print(data.label_analysis(claim_label))

#data2 = EvidenceLCA_Combine(source_claim, source_claim_evidence_noisy, 10, claim_label, source_truthworthiness)
#data2.LCA()

#
data = EvidenceLCA_noisy_recall(source_claim, source_claim_evidence_noisy, 19, claim_label, source_truthworthiness)
#data = EvidenceLCA_FullConnect(source_evidence, 100, claim_label)
result, rank_data = data.LCA()
print("evidence LCA: ")
print("precision: ", data.precision(result))
print("spearsman: ", data.spearsman_source())
print("pearson: ", data.pearson_source())


#print(data.error_analysis(result))
#print(data.Accuracy)
#exit()

#source_rank = sorted(data.Accuracy.items(), key = lambda d:d[1], reverse=True)
#source_top = []
#for i in range(100):
#    source_top.append(source_rank[i][0])


#with codecs.open("new_training_data_headline_with_n_2.txt", "w") as outFp:
#    for i in rank_data:
#        outFp.write(i[0] + "\t" + i[1] + "\t" + str(i[2]) + "\n")
#    outFp.close()







