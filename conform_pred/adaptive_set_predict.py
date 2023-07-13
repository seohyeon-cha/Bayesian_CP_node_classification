import torch
import torch.nn as nn
import ipdb
import numpy as np
import math
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from utils import accuracy_np

def ACP(model, features, idx_train, idx_cal, idx_test, labels, nz_idx, adj, adj_normt, alpha):

    test_acc, coverage, inefficiency, bin_dict, qhat = validation_ACP(model, features, idx_train, idx_cal, idx_test, labels, nz_idx, adj, adj_normt, alpha)

    return test_acc, coverage, inefficiency, bin_dict, qhat

 
def validation_ACP(model, features, idx_train, idx_cal, idx_test, labels, nz_idx, adj, adj_normt, alpha):
    
    with torch.no_grad():
        out, _, = model(x=features
                        , labels=labels
                        , adj=adj
                        , nz_idx=nz_idx
                        , obs_idx=idx_train
                        , adj_normt=adj_normt
                        , training=False)

        
        out = out.detach().cpu() # log-softmax output 
        cal_smx_np = np.exp(out[idx_cal].numpy())
        cal_labels = labels[idx_cal].detach().cpu().numpy()
        n_cal = len(idx_cal)

        order = np.argsort(-cal_smx_np, axis=1)
        ranks = np.empty_like(order)
        for i in range(n_cal):
            ranks[i, order[i]] = np.arange(len(order[i]))

        # sorted scores
        
        prob_sort = - np.sort(-cal_smx_np, axis=1)
        Z = prob_sort.cumsum(axis=1)  

        L = np.array([ranks[i, cal_labels[i]] for i in range(n_cal)]) 
        E = np.array([Z[i, L[i]] for i in range(n_cal)])
        prob = np.array([prob_sort[i, L[i]] for i in range(n_cal)])
        U = np.random.rand(n_cal)
        E_rand = np.maximum(E - np.multiply(prob, U), 0)
        qhat = np.quantile(E_rand, 1-alpha, interpolation='higher')
        # qhat 구하고 using i,dx_cal
        
    
    with torch.no_grad():
        out, _, = model(x=features
                        , labels=labels
                        , adj=adj
                        , nz_idx=nz_idx
                        , obs_idx=idx_train
                        , adj_normt=adj_normt
                        , training=False)

        out_np = out.cpu().data.numpy()
        acc_test = accuracy_np(out_np, labels.cpu().data.numpy(), idx_test)
        
        bin_dict = {}

        test_smx_np = np.exp(out[idx_test].numpy())
        test_labels = labels[idx_test].detach().cpu().numpy()
        n_test = len(idx_test)

        # sorted indices 
        order = np.argsort(- test_smx_np, axis=1)
        ranks = np.empty_like(order)
        for i in range(n_test):
            ranks[i, order[i]] = np.arange(len(order[i]))
        ipdb.set_trace()
        # sorted scores
        prob_sort = - np.sort(- test_smx_np, axis=1)
        Z = prob_sort.cumsum(axis=1)
        L = np.argmax(Z >= qhat, axis=1).flatten()
        Z_excess = np.array([Z[i, L[i]] for i in range(n_test)]) - qhat
        p_remove = Z_excess / np.array([prob_sort[i, L[i]] for i in range(n_test)])
        U = np.random.rand(n_test)
        remove = U <= p_remove
        for i in np.where(remove)[0]:
                L[i] = L[i] - 1
        
        S = [order[i,np.arange(0, L[i]+1)] for i in range(n_test)]
        coverage = np.mean([test_labels[i] in S[i] for i in range(n_test)])
        inefficiency = np.mean([len(S[i]) for i in range(n_test)])

        
    
    
    return acc_test, coverage, inefficiency, bin_dict, qhat


def reliability_diagram(nlog_sx_scores, test_labels, qhat):

    num_test = nlog_sx_scores.size(0)
    sx_scores = torch.exp(-nlog_sx_scores)
    confidence, prediction = torch.max(sx_scores, dim=1)
    
    # initialize bins
    bin_dict = {}
    num_bins = 10
    for bin_idx in range(num_bins):
        bin_dict[bin_idx] = {}

    for bin_idx in range(num_bins):
        bin_dict[bin_idx]['count'] = 0
        bin_dict[bin_idx]['conf'] = 0
        bin_dict[bin_idx]['acc'] = 0
        bin_dict[bin_idx]['cov'] = 0
        bin_dict[bin_idx]['ineff'] = 0

        bin_dict[bin_idx]['bin_acc'] = 0
        bin_dict[bin_idx]['bin_conf'] = 0
        bin_dict[bin_idx]['bin_cov'] = 0
        bin_dict[bin_idx]['bin_ineff'] = 0


    # 여기 코드 체크해보기
    for i in range(num_test):


        conf = confidence[i]
        pred = prediction[i]
        label = test_labels[i]
        bin_idx = int(math.ceil((num_bins * conf)-1))
        if bin_idx == -1:
            bin_idx = 0

        # assing to bin with bin_idx
        bin_dict[bin_idx]['count'] += 1
        bin_dict[bin_idx]['conf'] += conf
        bin_dict[bin_idx]['acc'] += (1 if (label==pred) else 0)
        
        # CP
        nc_score = nlog_sx_scores[i]
        set_prediction = nc_score <= qhat
        coverage = (1 if set_prediction[label]==1 else 0)
        ineff = sum(set_prediction).item()
        bin_dict[bin_idx]['cov'] += coverage
        bin_dict[bin_idx]['ineff'] += ineff

    for bin_idx in range(num_bins):
        count = bin_dict[bin_idx]['count']
        if count == 0:
            bin_dict[bin_idx]['bin_acc'] = 0
            bin_dict[bin_idx]['bin_conf'] = 0
            bin_dict[bin_idx]['bin_cov'] = 0
            bin_dict[bin_idx]['bin_ineff'] = 0
        else:    
            bin_dict[bin_idx]['bin_acc'] = float(bin_dict[bin_idx]['acc']/count)
            bin_dict[bin_idx]['bin_conf'] = float(bin_dict[bin_idx]['conf']/count)
            bin_dict[bin_idx]['bin_cov'] = float(bin_dict[bin_idx]['cov']/count)
            bin_dict[bin_idx]['bin_ineff'] = float(bin_dict[bin_idx]['ineff']/count)

    return bin_dict

