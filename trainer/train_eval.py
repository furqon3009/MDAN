import torch
from utils import *
import numpy as np
import random
import math
from itertools import cycle
from scipy.stats import wasserstein_distance
denorm=130


def train_source(model, train_dl, optimizer, criterion,config,device, alpha1, alpha2):
    model.train()
    epoch_loss = 0
    epoch_score = 0
    for inputs, labels in train_dl:
        src = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        #forward
        pred, feat = model(src)
        
        # #denormalization
        # pred  = pred * denorm
        # labels = labels * denorm
        
        # Mix at source
        lbd = np.random.beta(2,2) #lambda
        mt = random.uniform(0,1) #mask

        s0,s1,s2 = inputs.shape
        randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
        mt = torch.bernoulli(randuniform).to(device)
        m_ones = torch.ones(s0,s1,s2).to(device)

        sum_mt = mt.flatten().sum()
        c_mask = sum_mt/(s0*s1*s2)
        c_unmask = ((s0*s1*s2) - sum_mt)/(s0*s1*s2)
       
        #src if mt=1 -> x=0
        src2 = torch.clone(src)
        src2 = src2 * (m_ones-mt)
        
        gamma = 0.5
        idx = torch.randperm(feat.shape[0])
        
        feat_j = feat[idx]
        label_j = labels[idx]
        feat_mix = lbd*feat + (1-lbd)*feat_j
        label_mix = lbd*labels + (1-lbd)*label_j

        #loss ori and score
        rul_loss = criterion(pred.squeeze(), labels)
        score = scoring_func(pred.squeeze() - labels)

        #loss mix
        pred_mix = model.forward_regressor_only(feat_mix)
        rul_loss_mix = criterion(pred_mix.squeeze(), label_mix)

        # pred reconstruct
        pred_reconstruct = model.forward_regressor2(src2)
        #loss reconstruct
        rul_loss_reconstruct = gamma * c_mask * criterion(pred_reconstruct, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct, src2)

        #loss_source
        rul_loss = rul_loss + alpha1 * rul_loss_mix + alpha2 * rul_loss_reconstruct

        rul_loss.backward()
        if (config['model_name']=='LSTM'):
            clip=config['CLIP']
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # only for LSTM models
        optimizer.step()

        epoch_loss += rul_loss.item()
        epoch_score += score
    return epoch_loss / len(train_dl), epoch_score, pred, labels



def train_mix_inter(model, src_train_dl, tgt_train_dl, optimizer, criterion, config, device, alpha3, lbd_mix):
    model.train()
    epoch_loss = 0
    epoch_score = 0
    temp = 0.05
    batch_iterator = zip(src_train_dl, cycle(tgt_train_dl))
    num_iter = min(len(src_train_dl),len(tgt_train_dl))
    

    i = 0
    #train source
    for source, target in batch_iterator :
         
        inputs, labels = source
        inputs_tgt, labels_tgt = target

        #Train Source Domain
        src = inputs.to(device)

        i += 1
        
        labels = labels.to(device)
        optimizer.zero_grad()
         
        #forward source
        pred, feat = model(src)
        
        #Train Intermediate Mixup Domain
        src_tgt = inputs_tgt.to(device)
        labels_tgt = labels_tgt.to(device)
        pred_tgt, feat_tgt = model(src_tgt) 

        pseudo_labels = pred_tgt.squeeze()
        

        inputs_mixup = lbd_mix * inputs + (1-lbd_mix) * inputs_tgt
        labels_mixup = lbd_mix * labels + (1-lbd_mix) * pseudo_labels

        src_mixup = inputs_mixup.to(device)
        pred_input_mixup, _ = model(src_mixup)

        feat_mixup = lbd_mix * feat + (1-lbd_mix) * feat_tgt
        pred_mixup = model.forward_regressor_only(feat_mixup)

        feat = feat.cpu().detach().numpy().flatten()
        feat_mixup = feat_mixup.cpu().detach().numpy().flatten()
        feat_tgt = feat_tgt.cpu().detach().numpy().flatten()

        q = math.exp(-1*wasserstein_distance(feat,feat_mixup)/(wasserstein_distance(feat,feat_mixup)+wasserstein_distance(feat_tgt,feat_mixup)*temp))
        lbd_mix = (i*(1-q)/num_iter) + q*lbd_mix
        lbd_mix = torch.clamp(torch.as_tensor(random.uniform(lbd_mix-0.2, lbd_mix+0.2)), min=0.0, max=1.0) 
        
        rul_loss_cd = criterion(pred_input_mixup.squeeze(), labels_mixup) + criterion(pred_mixup.squeeze(), labels_mixup)
        score = scoring_func(pred.squeeze() - labels_mixup)

        #loss intermediate
        rul_loss = alpha3 * rul_loss_cd

        rul_loss.backward()
        if (config['model_name']=='LSTM'):
            clip=config['CLIP']
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # only for LSTM models
        optimizer.step()

        epoch_loss += rul_loss.item()
        epoch_score += score

    return epoch_loss / num_iter, epoch_score, pred, labels, lbd_mix

def train_mix_tgt(model, tgt_train_dl, optimizer, criterion, config, device, alpha4, alpha5, lbd):
    model.train()
    epoch_loss = 0
    epoch_score = 0
    temp = 0.05
    num_iter = len(tgt_train_dl)

    i = 0
    #train source
    for inputs, labels in tgt_train_dl:
        src = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        #forward
        pred, feat = model(src)

        pseudo_labels = pred.squeeze()

        idx = torch.randperm(feat.shape[0])
        
        feat_j = feat[idx]
        pseudo_labels_j = pseudo_labels[idx]
        labels_mixup = lbd * pseudo_labels + (1-lbd) * pseudo_labels_j
        feat_mixup = lbd * feat + (1-lbd) * feat_j
        pred_mixup = model.forward_regressor_only(feat_mixup)

        rul_loss_tgt_or = criterion(pred.squeeze(), pseudo_labels)
        rull_loss_tgt_mix = criterion(pred_mixup.squeeze(), labels_mixup)

        #loss target
        rul_loss = alpha4 * rul_loss_tgt_or + alpha5 * rull_loss_tgt_mix
        score = scoring_func(pred.squeeze() - labels)

        rul_loss.backward()
        if (config['model_name']=='LSTM'):
            clip=config['CLIP']
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # only for LSTM models
        optimizer.step()

        epoch_loss += rul_loss.item()
        epoch_score += score

    return epoch_loss / len(tgt_train_dl), epoch_score, pred, labels


def evaluate(model, test_dl, criterion, config,device):
    model.eval()
    total_feas=[];total_labels=[]
    epoch_loss = 0
    epoch_score = 0
    predicted_rul = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_dl:
            src = inputs.to(device)
            if config['permute'] == True:
                src = src.permute(0, 2, 1).to(device)  # permute for CNN model
            labels = labels.to(device)
            if config['model_name'] == 'seq2seq':
                pred, feat, dec_outputs = model(src)
            else:
                pred, feat = model(src)
            # denormalize predictions
            pred = pred * denorm
            if labels.max() <= 1:
                labels = labels * denorm
            rul_loss = criterion(pred.squeeze(), labels)
            score = scoring_func(pred.squeeze() - labels)
            epoch_loss += rul_loss.item()
            epoch_score += score
            total_feas.append(feat)
            total_labels.append(labels)

            predicted_rul += (pred.squeeze().tolist())
            true_labels += labels.tolist()

    model.train()
    return epoch_loss / len(test_dl), epoch_score, torch.cat(total_feas), torch.cat(total_labels),predicted_rul,true_labels
# def evaluate(model, test_dl, criterion, config):
#     model.eval()
#     epoch_loss = 0
#     epoch_score = 0
#     predicted_rul = []
#     true_labels = []
#     with torch.no_grad():
#         for inputs, labels in test_dl:
#             src = inputs.to(device)
#             if config['permute'] == True:
#                 src = src.permute(0, 2, 1).to(device)  # permute for CNN model
#             labels = labels.to(device)
#             pred, feat = model(src)
#             # denormalize predictions
#             pred = pred * denorm
#             if labels.max() <= 1:
#                 labels = labels * denorm
#             rul_loss = criterion(pred.squeeze(), labels)
#             score = scoring_func(pred.squeeze() - labels)
#             epoch_loss += rul_loss.item()
#             epoch_score += score
#
#             predicted_rul += (pred.squeeze().tolist())
#             true_labels += labels.tolist()
#     return epoch_loss / len(test_dl), epoch_score, predicted_rul,true_labels