from data_process import source_evidence_claim
from data_process import claim_label
from data_process import vectorize
from scipy.special import expit
from sklearn.metrics.pairwise import  cosine_similarity
from scipy.spatial.distance import cosine
from bayes_opt import BayesianOptimization
from decimal import *
getcontext().prec = 6
import numpy as np
#from utility import sent2emb



class TrustEvidence:

    def __init__(self, data, itr, gt):
        # Data Part
        self.source_item_claim_evi = data
        self.ground_truth = gt
        self.source_num = 0

        # Hyper Parameter
        self.iteration_num = itr
        self.total_claim = 100

        # Model Parameter
        self.Accuracy = {}
        self.item_val = {}
        self.item_val_dis = {}
#        self.Extract = 0.6

        # Run Initialization
        #self.__initialization__()

    def __initialization__(self):

        # Intialization for source trustworthiness
        self.source_num = len(self.source_item_claim_evi.keys())
        for i in self.source_item_claim_evi.keys():
            self.Accuracy[i] = 0.5

        # Intialization for Item Value Distribution
        for i in self.source_item_claim_evi.keys():
            for j in self.source_item_claim_evi[i].keys():
                if j not in self.item_val_dis:
                    self.item_val_dis[j] = {}
                for k in self.source_item_claim_evi[i][j].keys():
                    self.item_val_dis[j][k] = -1
                if j not in self.item_val:
                    self.item_val[j] = ""

    def precision(self, result):
        acc = 0
        acc_total = 0
        for i in result.keys():
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

class SimpleLCA(TrustEvidence):

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
            self.item_val[i] = 0

    def ini_source(self):
        for i in self.Accuracy.keys():
            self.Accuracy[i] = 0.5

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
                                self.item_val_dis[j][l] = float(1 - self.Accuracy[i])/float(self.total_claim)
                            else:
                                self.item_val_dis[j][l] = self.item_val_dis[j][l] * float(1 - self.Accuracy[i]) / float(self.total_claim)
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

data = SimpleLCA(source_evidence_claim, 100, claim_label)
result = data.SimLCA()
print(data.one_label_accuracy())
print(data.precision(result))
print(data.label_analysis(result))
print(data.error_analysis(result))


class EvidenceLCA(SimpleLCA):

    def __init__(self, data, itr, gt):
        SimpleLCA.__init__(self, data, itr, gt)
        self.Extractor = {}
        for i in self.source_item_claim_evi.keys():
            for j in self.source_item_claim_evi[i].keys():
                self.Extractor[j] = 0

    def __initialization__(self):
        SimpleLCA.__initialization__(self)
        for i in self.Extractor.keys():
            self.Extractor[i] = 0

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
                for k in self.source_item_claim_evi[i][j].keys():
                    headline_vector = vectorize.transform([self.source_item_claim_evi[i][j][k][0][0]]).toarray().flatten()
                    if k == self.item_val[j]:
                        bsyd = 1
                    else:
                        bsyd = 0

                    in_sigma = cosine(claim_vector, headline_vector)
                    if np.isnan(in_sigma):
                        in_sigma = 0
                    sigma_sim = expit(self.Extractor[j] * float(float(1) - in_sigma))

                    temp_item = bsyd - self.Accuracy[i] * sigma_sim
                    #if self.Accuracy[i] != 0:
                    if float(self.Accuracy[i] * (1-self.Accuracy[i]*sigma_sim)) == 0:
                        temp_item = float(temp_item)/(float((self.Accuracy[i] + 0.00001) * (1-(self.Accuracy[i] + 0.00001)*sigma_sim)))
                    else:
                        temp_item = float(temp_item)/float(self.Accuracy[i] * (1-self.Accuracy[i]*sigma_sim))
                    #else:
                    #    temp_item = 0

                    #if self.item_val[j] == 1:
                    #    temp_item = temp_item
                    #else:
                    #    temp_item = 0
                    temp_item = temp_item * float(self.item_val_dis[j][k])

                    if np.isnan(temp_item):
                        temp_item = 0
                        print "Here nan"
                        print self.Accuracy[i], in_sigma, self.Extractor[j]
                        print(claim_vector.shape)
                        print(headline_vector.shape)
                        print(sum(claim_vector))
                        print(sum(headline_vector))
                        print('\n')
                    gradient_iter = gradient_iter + temp_item
            #print gradient_iter
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
            gradient_Alpha[i] = 0
            claim_vector = vectorize.transform([i]).toarray().flatten()
            for j in item_source[i].keys():
                for k in self.source_item_claim_evi[j][i].keys():
                    headline_vector = vectorize.transform([self.source_item_claim_evi[j][i][k][0][0]]).toarray().flatten()
                    sim = 1 - cosine(claim_vector, headline_vector)
                    if np.isnan(sim):
                        sim = 0

                    sigma_sim = expit(self.Extractor[i] * sim)

                    if k == self.item_val[i]:
                        bsyd = 1
                    else:
                        bsyd = 0

                    temp_grad = (1 - sigma_sim) * sim * (bsyd - self.Accuracy[j] * sigma_sim)
                    temp_grad = float(temp_grad) / float(1 - self.Accuracy[j] * sigma_sim)
                    temp_grad = temp_grad * self.item_val_dis[i][k]
                    gradient_Alpha[i] = gradient_Alpha[i] + temp_grad
        return gradient_Alpha

    def EstepLayer(self):
        for i in self.source_item_claim_evi.keys(): # source
            for j in self.source_item_claim_evi[i].keys(): # claim_id
                claim_vector = vectorize.transform([j]).toarray()
                for k in self.source_item_claim_evi[i][j].keys():
                    for l in self.item_val_dis[j].keys():
                        headline_vector = vectorize.transform([self.source_item_claim_evi[i][j][k][0][0]])
                        if l == k:
                            if self.item_val_dis[j][l] == -1:
                                self.item_val_dis[j][l] = self.Accuracy[i] * expit(self.Extractor[j] * cosine_similarity(claim_vector, headline_vector))
                            else:
                                self.item_val_dis[j][l] = self.item_val_dis[j][l] * self.Accuracy[i] * expit(self.Extractor[j] * cosine_similarity(claim_vector, headline_vector))
                        else:
                            if self.item_val_dis[j][l] == -1:
                                self.item_val_dis[j][l] = float(1 - self.Accuracy[i] * expit(self.Extractor[j] * cosine_similarity(claim_vector, headline_vector)))/float(self.total_claim)
                            else:
                                self.item_val_dis[j][l] = self.item_val_dis[j][l] * float(1 - self.Accuracy[i] * expit(self.Extractor[j] * cosine_similarity(claim_vector, headline_vector))) / float(self.total_claim)
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
                pre_alpha[m] = self.Extractor[m]

            gradient_alpha = self.compute_gradient_alpha()
            lr = lr2 / float(i + 1)
            for m in gradient_alpha.keys():
                self.Extractor[m] = self.Extractor[m] - float(lr) * gradient_alpha[m]
            delta_extractor = {}
            for m in self.Extractor.keys():
                delta_extractor[m] = abs(self.Extractor[m] - pre_alpha[m])
            delta = sum(delta_extractor.values())
            if delta < ext_pre:
                break
            print("iteration extractor", i)

        #acc_sum = sum(self.Accuracy.values())
        #for i in self.Accuracy.keys():
        #    self.Accuracy[i] = expit(self.Accuracy[i])
        #for i in self.Extractor.keys():
        #    self.Extractor[i] = expit(self.Extractor[i])
        print self.Accuracy
        print self.Extractor

    def LCA(self):
        self.__initialization__()
        pre_accuracy = {}
        for i in self.Accuracy.keys():
            pre_accuracy[i] = 0.5
        lr1 = 0.0001
        lr1_b = 0.01
        lr2 = 1
        lr2_b = 10
        for iter in range(self.iteration_num):
            self.ini_item_val_dis()
            self.ini_item_val()
            self.EstepLayer()
            self.ini_source()
            for i in self.Extractor.keys():
                self.Extractor[i] = 0
            self.MstepLayer(lr1, lr2, lr1_b, lr2_b) #(0.0001, 0.1)
            print(data.precision(self.item_val))
        return self.item_val


data = EvidenceLCA(source_evidence_claim, 100, claim_label)
result = data.LCA()
print(data.precision(result))
#print(data.error_analysis(result))






