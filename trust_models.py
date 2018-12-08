from scipy.stats import spearmanr
from scipy.stats import pearsonr
import copy
import numpy as np
from utility import jd_sim
from scipy.special import expit
from scipy.spatial.distance import cosine

cweights1 = 1
cweights2 = 0#.00000000000000001 #.0000000000000000000001
smooth_para = 0.0000001
const_para = 0.1

class Baseline:
    def __init__(self, data1, data2, gt, sgt):
        self.source_claim = data1
        self.source_claim_evidence = data2
        self.claim_result = {}
        self.source_gt = gt
        self.source_acc_gt = sgt

    def majority_vote_claim_only(self):
        claim_result = {}
        claim_vote = {}
        for i in self.source_claim.keys():
            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    if k not in claim_vote:
                        claim_vote[k] = {}
                        claim_vote[k][j] = 0
                    if j not in claim_vote[k]:
                        claim_vote[k][j] = 0
                    claim_vote[k][j] = claim_vote[k][j] + 1
        for i in claim_vote.keys():
            rank_temp = sorted(claim_vote[i].items(), key = lambda d:d[1], reverse=True)
            claim_result[i] = rank_temp[0][0]
        source_acc = {}
        source_total = {}
        for i in self.source_claim.keys():
            if i not in source_acc:
                source_acc[i] = 0
            if i not in source_total:
                source_total[i] = 0

            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    if j == claim_result[k]:
                        source_acc[i] = source_acc[i] + 1
                    source_total[i] = source_total[i] + 1
        for i in source_acc.keys():
            if source_total[i] != 0:
                source_acc[i] = float(source_acc[i])/float(source_total[i])
            else:
                source_acc[i] = 0
        self.claim_result = claim_result
        return claim_result, source_acc

    def majority_vote_claim_evidence(self):
        claim_result = {}
        for i in self.source_claim_evidence.keys():
            for j in self.source_claim_evidence[i].keys():
                if j not in claim_result:
                    claim_result[j] = {}
#                print(self.source_claim_evidence[i][j])
                for k in self.source_claim_evidence[i][j].keys():
                    if k not in claim_result[j]:
                        claim_result[j][k] = 0
                    claim_result[j][k] = claim_result[j][k] + len(self.source_claim_evidence[i][j][k])

        mj_claim = {}
        for i in claim_result.keys():
            claim_result[i][0] = 0
            rank_temp = sorted(claim_result[i].items(), key=lambda d: d[1], reverse=True)
            mj_claim[i] = rank_temp[0][0]

        source_acc = {}
        source_total = {}

        for i in self.source_claim_evidence.keys():
            source_acc[i] = 0
            total = 0
            acc = 0
            # for j in self.source_claim_evidence[i].keys():
            #     for k in self.source_claim_evidence[i][j].keys():
            #         if k == 0:
            #             total = total + len(self.source_claim_evidence[i][j][k])
            #             continue
            #         else:
            #             total = total + len(self.source_claim_evidence[i][j][k])
            #             if k == mj_claim[j]:
            #                 acc = acc + len(self.source_claim_evidence[i][j][k])
            # if total == 0:
            #     source_acc[i] = 0
            # else:
            #     source_acc[i] = float(acc)/float(total)
            for j in self.source_claim_evidence[i].keys():
                for k in self.source_claim_evidence[i][j].keys():
                    if k == 0:
                        continue
                    else:
                        total = total + len(self.source_claim_evidence[i][j][k])
                        if k == mj_claim[j] and k != 0:
                            acc = acc + len(self.source_claim_evidence[i][j][k])
                if total == 0:
                    continue
                else:
                    source_acc[i] = source_acc[i] + float(acc)/float(total)
            if len(self.source_claim_evidence[i].keys()) != 0:
                source_acc[i] = source_acc[i]/float(len(self.source_claim_evidence[i].keys()))
            else:
                source_acc[i] = 0

        source_acc_claim_only = {}
        source_acc_claim_only_total = {}
        for i in self.source_claim.keys():
            if i not in source_acc_claim_only:
                source_acc_claim_only[i] = 0
            if i not in source_acc_claim_only_total:
                source_acc_claim_only_total[i] = 0

            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    if k not in mj_claim:
                        continue
                    if j == mj_claim[k]:
                        source_acc_claim_only[i] = source_acc_claim_only[i] + 1
                    source_acc_claim_only_total[i] = source_acc_claim_only_total[i] + 1
        for i in source_acc_claim_only.keys():
            if source_acc_claim_only_total[i] != 0:
                source_acc_claim_only[i] = float(source_acc_claim_only[i])/float(source_acc_claim_only_total[i])
            else:
                source_acc_claim_only[i] = 0

        return mj_claim, source_acc, source_acc_claim_only

    def precision(self, result):
        acc = 0
        acc_total = 0
        for i in self.source_gt.keys():
            if i not in result:
                acc_total = acc_total + 1
            else:
                if result[i] == self.source_gt[i]:
                    acc = acc + 1
                acc_total = acc_total + 1
        return float(acc) / float(acc_total)

    def spearsman(self, source_res):
        vec_gt = []
        vec_res = []
        for i in self.source_acc_gt.keys():
            vec_gt.append(self.source_acc_gt[i])
            vec_res.append(source_res[i])
        return spearmanr(vec_gt, vec_res)

    def pearson(self, source_res):
        normalize_source_res = {}
        temp_total = float(sum(source_res.values()))
        for i in source_res.keys():
            normalize_source_res[i] = float(source_res[i])/float(temp_total)
        vec_gt = []
        vec_res = []
        for i in self.source_acc_gt.keys():
            vec_res.append(normalize_source_res[i])
            vec_gt.append(self.source_acc_gt[i])
        return pearsonr(vec_gt, vec_res)

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
                        self.item_val_dis[k][1] = ""
                        self.item_val_dis[k][-1] = ""
                        self.item_val_dis[k][0] = ""
                    if k not in self.item_val:
                        self.item_val[k] = ""
        for i in self.source_item_claim_evi.keys():
            for j in self.source_item_claim_evi[i].keys():
                if j not in self.item_val_dis:
                    self.item_val_dis[j] = {}
                    self.item_val_dis[j][1] = ""
                    self.item_val_dis[j][-1] = ""
                    self.item_val_dis[j][0] = ""
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
        for i in result.keys():
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
                self.item_val_dis[i][j] = ""

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
                            if self.item_val_dis[k][l] == "":
                                self.item_val_dis[k][l] = self.Accuracy[i]
                            else:
                                self.item_val_dis[k][l] = self.item_val_dis[k][l] * self.Accuracy[i]
                        else:
                            if self.item_val_dis[k][l] == "":
                                self.item_val_dis[k][l] = float(float(1) - self.Accuracy[i])/float(self.total_claim)
                            else:
                                self.item_val_dis[k][l] = self.item_val_dis[k][l] * float(float(1) - self.Accuracy[i])/float(self.total_claim)
        # Normalization
        #print(self.item_val_dis)
        for i in self.item_val_dis.keys(): # item_id
            sum_total = 0
            for j in self.item_val_dis[i].keys():
                if self.item_val_dis[i][j] == "":
                    self.item_val_dis[i][j] = 0
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

class EvidenceLCA_noisy_recall(SimpleLCA):

    def __init__(self, data1, data2, itr, gt, sgt, ent_rep, emb_rep, tfidf_rep, content_id_map):
        SimpleLCA.__init__(self, data1, data2, itr, gt, sgt)
        self.content_id = content_id_map
        self.ent_rep = ent_rep
        self.emb_rep = emb_rep
        self.tfidf_rep = tfidf_rep
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
                    #print(constraints)
                    gradient_iter = gradient_iter + float(gd) + const_para * float(constraints)
            # print gradient_iter
            gradient_H[i] = gradient_iter
        return gradient_H

    def compute_gradient_Ps(self):
        gradient_H = {}
        for i in self.source_item_claim_evi.keys():
            gradient_iter = 0
            for j in self.source_item_claim_evi[i].keys():
                claim_vector = self.tfidf_rep[self.content_id[j]]
                claim_ent = self.ent_rep[self.content_id[j]]
                claim_emb = self.emb_rep[self.content_id[j]]

                for k in self.source_item_claim_evi[i][j].keys():
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        headline_vector = self.tfidf_rep[self.content_id[evi]]
                        headline_ent = self.ent_rep[self.content_id[evi]]
                        headline_emb = self.emb_rep[self.content_id[evi]]

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
                        temp_item = float(temp_item) / (float((self.Accuracy[i] + 0.0000001) * (1 - (self.Accuracy[i] + 0.0000001) * sigma_sim)))
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
                    claim_vector = self.tfidf_rep[self.content_id[j]]
                    claim_ent = self.ent_rep[self.content_id[j]]
                    claim_emb = self.emb_rep[self.content_id[j]]
                    for k in self.source_item_claim_evi[i][j].keys():
                        if k == self.item_val[j]:
                            bsyd = 1
                        else:
                            bsyd = 0
                        for evi, escore in self.source_item_claim_evi[i][j][k]:
                            if evi not in self.content_id:
                                continue

                            headline_vector = self.tfidf_rep[self.content_id[evi]]
                            headline_emb = self.emb_rep[self.content_id[evi]]
                            headline_ent = self.ent_rep[self.content_id[evi]]

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

    def compute_gradient_QRs(self):
        gradient_Q = {}
        gradient_R = {}
        for i in self.source_item_claim_evi.keys():
            gradient_iterQ = 0
            gradient_iterR = 0
            for j in self.source_item_claim_evi[i].keys():
                claim_vector = self.tfidf_rep[self.content_id[j]]
                claim_ent = self.ent_rep[self.content_id[j]]
                claim_emb = self.emb_rep[self.content_id[j]]

                for k in self.source_item_claim_evi[i][j].keys():
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        headline_vector = self.tfidf_rep[self.content_id[evi]]
                        headline_ent = self.ent_rep[self.content_id[evi]]
                        headline_emb = self.emb_rep[self.content_id[evi]]

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
                        item1R = float(bsyd) / float(self.Recall[i])
                        item2Ru = float(1 - bsyd) * self.Qf[i] * sigma_sim
                        item2Rb = self.Recall[i] * sigma_sim * QR - np.power(QR, 2)

                        gradient_iterR = gradient_iterR + self.item_val_dis[j][k] * (item1R + float(item2Ru) / float(item2Rb))


                        item1Q = -float(self.Recall[i] * sigma_sim)
                        item2Q = self.Recall[i] * sigma_sim * QR - np.power(QR, 2)
                        gradient_iterQ = gradient_iterQ + self.item_val_dis[j][k] * float(1 - bsyd) * float(item1Q)/float(item2Q)

            constraintsQ =  - self.Accuracy[i] / float(self.Qf[i] + self.Recall[i])
            constraintsR = self.Accuracy[i] * self.Qf[i] / (float(self.Recall[i]) * float(self.Recall[i] + self.Qf[i]))
            gradient_Q[i] = gradient_iterQ + float(const_para) * constraintsQ
            gradient_R[i] = gradient_iterR + float(const_para) * constraintsR

        return gradient_Q, gradient_R

    def compute_gradient_Rs(self):
        gradient_R = {}
        for i in self.source_item_claim_evi.keys():
            gradient_iter = 0
            for j in self.source_item_claim_evi[i].keys():
                claim_vector = self.tfidf_rep[self.content_id[j]]
                claim_ent = self.ent_rep[self.content_id[j]]
                claim_emb = self.emb_rep[self.content_id[j]]

                for k in self.source_item_claim_evi[i][j].keys():
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        headline_vector = self.tfidf_rep[self.content_id[evi]]
                        headline_ent = self.ent_rep[self.content_id[evi]]
                        headline_emb = self.emb_rep[self.content_id[evi]]

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

        # for i in self.Accuracy.keys():
        #     acc = 0
        #     temp = 0
        #     if i not in self.source_claim:
        #         self.Accuracy[i] = 0
        #         continue
        #
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

            pre_q = {}
            for m in self.Qf.keys():
                pre_q[m] = self.Qf[m]

            gradient_Q, gradient_R = self.compute_gradient_QRs()

            lr = lr3
            for m in gradient_R.keys():
                self.Recall[m] = self.Recall[m] + float(lr) * gradient_R[m]

            lr = lr3
            for m in gradient_Q.keys():
                self.Qf[m] = self.Qf[m] + float(lr) * gradient_Q[m]

            #delta_extractor = {}
            #for m in self.Recall.keys():
            #    delta_extractor[m] = abs(self.Recall[m] - pre_recall[m])
            #delta = sum(delta_extractor.values())

            #if delta < lr1_b:
            #    break

        # for i in range(50):
        #     pre_q = {}
        #     for m in self.Qf.keys():
        #         pre_q[m] = self.Qf[m]
        #     gradient_Q = self.compute_gradient_Qs()
        #     lr = lr3
        #     for m in gradient_Q.keys():
        #         self.Qf[m] = self.Qf[m] + float(lr) * gradient_Q[m]

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
            print("iteration extractor", i)
        print(self.Accuracy)
        print(self.Recall)
        print(self.Qf)
        print(self.Extractor)

    def EstepLayer_Grad(self):

        for i in self.source_claim.keys():
            for j in self.source_claim[i].keys():
                for k in self.source_claim[i][j]:
                    for l in self.item_val_dis[k].keys():
                        if l == j:
                            if self.item_val_dis[k][l] == "":
                                self.item_val_dis[k][l] = float(cweights2) * np.log(self.Accuracy[i] + smooth_para)
                            else:
                                self.item_val_dis[k][l] = self.item_val_dis[k][l] + float(cweights2) * np.log(self.Accuracy[i] + smooth_para)
                        else:
                            if self.item_val_dis[k][l] == "":
                                self.item_val_dis[k][l] = float(cweights2) * np.log(float(float(1) - self.Accuracy[i]) /float(self.total_claim) + smooth_para)
                            else:
                                self.item_val_dis[k][l] = self.item_val_dis[k][l] + float(cweights2) * np.log(float(float(1) - self.Accuracy[i])/float(self.total_claim) + smooth_para)

        check = copy.deepcopy(self.item_val_dis)
        #print(check)
        for i in check.keys():
            check[i][0] = -1000000000
            for j in check[i].keys():
                if check[i][j] == "":
                    check[i][j] = -1000000000
            temp_rank = sorted(check[i].items(), key=lambda d: d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]
        print(self.precision(self.item_val))

        Acc_Extractor = {}
        for i in self.Recall.keys():
            if self.Qf[i] + self.Recall[i] == 0:
                Acc_Extractor[i] = 0
            else:
                Acc_Extractor[i] = float(self.Recall[i])/float(self.Recall[i] + self.Qf[i])

        for i in self.source_item_claim_evi.keys():  # source
            for j in self.source_item_claim_evi[i].keys():  # claim_id

                claim_vector = self.tfidf_rep[self.content_id[j]]
                claim_ent = self.ent_rep[self.content_id[j]]
                claim_emb = self.emb_rep[self.content_id[j]]

                for k in self.source_item_claim_evi[i][j].keys():
                    for evi, escore in self.source_item_claim_evi[i][j][k]:
                        if evi not in self.content_id:
                            continue
                        headline_vector = self.tfidf_rep[self.content_id[evi]]
                        headline_emb = self.emb_rep[self.content_id[evi]]
                        headline_ent = self.ent_rep[self.content_id[evi]]

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
                                if self.item_val_dis[j][l] == "":
                                    self.item_val_dis[j][l] = float(cweights1) * np.log(self.Recall[i] * exp_score + smooth_para)
                                else:
                                    self.item_val_dis[j][l] = self.item_val_dis[j][l] + float(cweights1) * np.log(self.Recall[i] * exp_score + smooth_para)
                            else:
                                if self.item_val_dis[j][l] == "":
                                    self.item_val_dis[j][l] = float(cweights1) * np.log(float(1 - Acc_Extractor[i] * exp_score) / float(self.total_claim) + smooth_para)
                                else:
                                    self.item_val_dis[j][l] = self.item_val_dis[j][l] + float(cweights1) * np.log(float(1 - Acc_Extractor[i] * exp_score)/ \
                                                                                               float(self.total_claim) + smooth_para)
        # Normalization



        #print(self.item_val_dis)

        for i in self.item_val_dis.keys():  # item_id
            sum_total = 0
            for j in self.item_val_dis[i].keys():
                if self.item_val_dis[i][j] != "":
                    sum_total = sum_total + np.exp(self.item_val_dis[i][j])

            if sum_total != 0:
                for j in self.item_val_dis[i].keys():
                    if self.item_val_dis[i][j] != "":
                        self.item_val_dis[i][j] = np.exp(float(self.item_val_dis[i][j])) / float(sum_total)
                    else:
                        self.item_val_dis[i][j] = 0
            else:
                for j in self.item_val_dis[i].keys():
                    self.item_val_dis[i][j] = 0


            self.item_val_dis[i][0] = 0

        for i in self.item_val_dis.keys():
            temp_rank = sorted(self.item_val_dis[i].items(), key=lambda d: d[1], reverse=True)
            self.item_val[i] = temp_rank[0][0]

        print(self.precision(self.item_val))
        print("\n")

    def LCA(self):
        self.__initialization__()
        #print(self.source_item_claim_evi)
        #exit()
        pre_accuracy = {}
        for i in self.Accuracy.keys():
            pre_accuracy[i] = 0.6

        lr1 = 0.0001
        lr1_b = 0.001
        lr2 = 0.0001
        lr2_b = 0.0001
        lr3 = 0.0001
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

            print("spearsman: ", self.spearsman_source())
            print("pearson: ", self.pearson_source())
            #for i in self.Accuracy.keys():
            #    if self.Accuracy[i] < 0:
            #        self.Accuracy[i] = 0
            #    if self.Accuracy[i] > 1:
            #        self.Accuracy[i] = 1
        # acc_temp = {}
        # total_temp = {}
        # for i in self.Accuracy.keys():
        #     if i not in acc_temp:
        #         acc_temp[i] = 0
        #     if i not in total_temp:
        #         total_temp[i] = 0
        #     if i not in self.source_claim:
        #         continue
        #     for j in self.source_claim[i].keys():
        #         if j == 0:
        #             continue
        #         for k in self.source_claim[i][j]:
        #             if self.item_val[k] == 0:
        #                 continue
        #             if j == self.item_val[k]:
        #                 acc_temp[i] = acc_temp[i] + 1
        #             total_temp[i] = total_temp[i] + 1
        # for i in acc_temp.keys():
        #     if total_temp[i] == 0:
        #         self.Accuracy[i] = 0
        #     else:
        #         self.Accuracy[i] = float(acc_temp[i])/float(total_temp[i])

        return self.item_val