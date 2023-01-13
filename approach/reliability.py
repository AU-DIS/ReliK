import settings as sett
import torch

import numpy as np
import math
from numpy import linalg as LA
from scipy.special import expit

def reliability_DistMult_local_normalization_as_Sum(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs):
    '''
    Getting the reliability sum as currently defined
    '''
    reliability_score = []
    for subgraph in subgraphs:
        
        max_score = 0
        counter_neg = 0
        counter_pos = 0
        sum_neg = 0
        sum_pos = 0

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            if score > max_score:
                                max_score = score
                            if -score > max_score:
                                max_score = -score

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]

                            if ((h,r,t) in all_triples):
                                sum_pos += 1/2 + ((score/max_score)/2)
                                counter_pos += 1
                            else:
                                sum_neg += 1/2 + ((score/max_score)/2)
                                counter_neg += 1
        reliability_score.append(( (sum_pos/counter_pos) + (1-(sum_neg/counter_neg)) )/2)
        
    return reliability_score

def reliability_local_normalization_as_Sum(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs, norm):
    '''
    Getting the reliability sum as currently defined
    '''
    reliability_score = []
    for subgraph in subgraphs:
        
        max_score = 0
        counter_neg = 0
        counter_pos = 0
        sum_neg = 0
        sum_pos = 0

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            score = score * (-1)
                            if score > max_score:
                                max_score = score

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            score = score * (-1)

                            if ((h,r,t) in all_triples):
                                sum_pos += 1-(score/max_score)
                                counter_pos += 1
                            else:
                                sum_neg += 1-(score/max_score)
                                counter_neg += 1
        reliability_score.append(( (sum_pos/counter_pos) + (1-(sum_neg/counter_neg)) )/2)
        
    return reliability_score

def reliability_DistMult_local_normalization_as_Sum_Rel_Freq(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs):
    '''
    Getting the reliability sum as currently defined
    '''
    reliability_score = []
    for subgraph in subgraphs:
        
        max_score = 0
        counter_neg = 0
        counter_pos = 0
        sum_neg = 0
        sum_pos = 0
        freq_rel = dict()
        for r in range(emb_train_triples.num_relations):
            freq_rel[r] = 0
        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            if score > max_score:
                                max_score = score
                            if -score > max_score:
                                max_score = -score
                            if ((h,r,t) in all_triples):
                                freq_rel[r] += 1

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]

                            if ((h,r,t) in all_triples):
                                sum_pos += (1/2 + ((score/max_score)/2))*(freq_rel[r]/len(all_triples))
                                counter_pos += 1
                            else:
                                sum_neg += (1/2 + ((score/max_score)/2))*(freq_rel[r]/len(all_triples))
                                counter_neg += 1
        reliability_score.append(( (sum_pos/counter_pos) + (1-(sum_neg/counter_neg)) )/2)
        
    return reliability_score

def reliability_local_normalization_as_Sum_Rel_Freq(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs, norm):
    '''
    Getting the reliability sum as currently defined
    '''
    reliability_score = []
    for subgraph in subgraphs:
        
        max_score = 0
        counter_neg = 0
        counter_pos = 0
        sum_neg = 0
        sum_pos = 0
        freq_rel = dict()
        for r in range(emb_train_triples.num_relations):
            freq_rel[r] = 0
        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            score = score * (-1)
                            if score > max_score:
                                max_score = score
                            if ((h,r,t) in all_triples):
                                freq_rel[r] += 1

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            score = score * (-1)

                            if ((h,r,t) in all_triples):
                                sum_pos += (1-(score/max_score))*(freq_rel[r]/len(all_triples))
                                counter_pos += 1
                            else:
                                sum_neg += (1-(score/max_score))*(freq_rel[r]/len(all_triples))
                                counter_neg += 1
        reliability_score.append(( (sum_pos/counter_pos) + (1-(sum_neg/counter_neg)) )/2)
        
    return reliability_score

def reliability_local_normalization_as_Difference_in_Average(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs):
    '''
    Getting the reliability sum as currently defined
    '''
    reliability_score = []
    max_score = 0
    min_score = 0
    flag = True
    for subgraph in subgraphs:
        
       
        counter_neg = 0
        counter_pos = 0
        sum_neg = 0
        sum_pos = 0

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            if flag:
                                max_score = score
                                min_score = score
                                flag = False
                            
                            if ((h,r,t) in all_triples):
                                sum_pos += score
                                counter_pos += 1
                            else:
                                sum_neg += score
                                counter_neg += 1
                            if score > max_score:
                                max_score = score
                            if score < min_score:
                                min_score = score
        sum_pos = sum_pos/counter_pos
        sum_neg = sum_neg/counter_neg
        if sum_pos > sum_neg:
            reliability_score.append(sum_pos - sum_neg)
        else:
            reliability_score.append(sum_neg - sum_pos)
        
    return reliability_score, max_score, min_score

def reliability_local_normalization_Ratio(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs):
    '''
    Getting the reliability sum as currently defined
    '''
    reliability_score = []
    for subgraph in subgraphs:
        
        max_score = 0
        counter_neg = 0
        counter_pos = 0
        sum_neg = 0
        sum_pos = 0

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            score = score * (-1)
                            if score > max_score:
                                max_score = score

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            score = score * (-1)

                            if ((h,r,t) in all_triples):
                                sum_pos += 1-(score/max_score)
                                counter_pos += 1
                            else:
                                sum_neg += 1-(score/max_score)
                                counter_neg += 1
        reliability_score.append( (sum_pos/counter_pos) / (1-(sum_neg/counter_neg)) )
        
    return reliability_score

def reliability_DistMult_local_normalization_Ratio(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs):
    '''
    Getting the reliability sum as currently defined
    '''
    reliability_score = []
    for subgraph in subgraphs:
        
        max_score = 0
        counter_neg = 0
        counter_pos = 0
        sum_neg = 0
        sum_pos = 0

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            if score > max_score:
                                max_score = score
                            if -score > max_score:
                                max_score = -score

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]

                            if ((h,r,t) in all_triples):
                                sum_pos += 1/2 + ((score/max_score)/2)
                                counter_pos += 1
                            else:
                                sum_neg += 1/2 + ((score/max_score)/2)
                                counter_neg += 1
        reliability_score.append( (sum_pos/counter_pos) / (1-(sum_neg/counter_neg)) )
        
    return reliability_score

def reliability_local_normalization_as_Difference_MaxMin(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs):
    '''
    Getting the reliability sum as currently defined
    '''
    reliability_score = []
    max_pos = []
    min_pos = []
    max_neg = []
    min_neg = []

    for subgraph in subgraphs:
        max_score_pos = 0
        min_score_pos = 0
        max_score_neg = 0
        min_score_neg = 0
        flag = True

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            if flag:
                                max_score_pos = score
                                min_score_pos = score
                                max_score_neg = score
                                min_score_neg = score
                                flag = False
                            
                            if ((h,r,t) in all_triples):
                                if score > max_score_pos:
                                    max_score_pos = score
                                if score < min_score_pos:
                                    min_score_pos = score
                            else:
                                if score > max_score_neg:
                                    max_score_neg = score
                                if score < min_score_neg:
                                    min_score_neg = score

        
        delta=max(max_score_pos,max_score_neg)-min(min_score_pos,min_score_neg)
        rel = ((min_score_pos - max_score_neg)-min(min_score_pos,min_score_neg)) / delta
        reliability_score.append(rel)
        max_pos.append(max_score_pos)
        min_pos.append(min_score_pos)
        max_neg.append(max_score_neg)
        min_neg.append(min_score_neg)

        
    return reliability_score, max_pos, min_pos, max_neg, min_neg

def reliability_transformation(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs):
    reliability_score_orignial = []
    reliability_score_average = []
    reliability_score_ratio = []
    reliability_score_maxmin = []
    max_pos = []
    min_pos = []
    max_neg = []
    min_neg = []
    for subgraph in subgraphs:
        max_score_pos = 0
        min_score_pos = 0
        max_score_neg = 0
        min_score_neg = 0

        max_score = 0
        min_score = 0
        counter_neg = 0
        counter_pos = 0
        sum_neg = 0
        sum_pos = 0
        avg_neg = 0
        avg_pos = 0

        flag = True

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            score = -math.log(expit(score + sett.GAMMA))
                            if flag:
                                max_score_pos = score
                                min_score_pos = score
                                max_score_neg = score
                                min_score_neg = score
                                max_score = score
                                min_score = score
                                flag = False
                            if score > max_score:
                                max_score = score
                            if score < min_score:
                                min_score = score

        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            score = -math.log(expit(score + sett.GAMMA))
                            if ((h,r,t) in all_triples):
                                sum_pos += 1-(score/max_score)
                                avg_pos += score
                                counter_pos += 1
                                if score > max_score_pos:
                                    max_score_pos = score
                                if score < min_score_pos:
                                    min_score_pos = score
                            else:
                                sum_neg += 1-(score/max_score)
                                avg_neg += score
                                counter_neg += 1
                                if score > max_score_neg:
                                    max_score_neg = score
                                if score < min_score_neg:
                                    min_score_neg = score
        avg_pos = avg_pos/counter_pos
        avg_neg = avg_neg/counter_neg
        if avg_pos > avg_neg:
            reliability_score_average.append(avg_pos - avg_neg)
        else:
            reliability_score_average.append(avg_neg - avg_pos)
        reliability_score_ratio.append( (sum_pos/counter_pos) / (1-(sum_neg/counter_neg)) )
        reliability_score_orignial.append(( (sum_pos/counter_pos) + (1-(sum_neg/counter_neg)) )/2)
        delta=max(max_score_pos,max_score_neg)-min(min_score_pos,min_score_neg)
        rel = ((min_score_pos - max_score_neg)-min(min_score_pos,min_score_neg)) / delta
        reliability_score_maxmin.append(rel)

        max_pos.append(max_score_pos)
        min_pos.append(min_score_pos)
        max_neg.append(max_score_neg)
        min_neg.append(min_score_neg)
        
    return reliability_score_orignial, reliability_score_average, reliability_score_ratio, reliability_score_maxmin, max_pos, min_pos, max_neg, min_neg

#############################################################

# OLD not used reliability scores!!!

#############################################################

def checkTransEScore(score, h,r,t, emb_train_triples, entity2embedding, relation2embedding):
    head_vec = np.array(entity2embedding[emb_train_triples.entity_id_to_label[h]])
    relation_vec = np.array(relation2embedding[emb_train_triples.relation_id_to_label[r]])
    tail_vec = np.array(entity2embedding[emb_train_triples.entity_id_to_label[t]])
    norm = -LA.norm(head_vec + relation_vec - tail_vec, ord=1)
    decision = math.isclose(norm, score, abs_tol=10**-sett.SIGFIGS)
    assert decision, f"score is not close (sigfigs {sett.SIGFIGS}) to value "

def reliability_local_normalization_three_part(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs, related):
    '''
    Getting the reliability sum as currently defined
    '''
    reliability_score = []
    for subgraph in subgraphs:
        
        max_score = 0
        counter_some_neg = 0
        counter_neg = 0
        counter_pos = 0
        sum_some_neg = 0
        sum_neg = 0
        sum_pos = 0
        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            score = score * (-1)
                            if score > max_score:
                                max_score = score

                            if ((h,r,t) in all_triples):
                                sum_pos += score
                                counter_pos += 1
                            elif((h,t) in related):
                                sum_some_neg += (score * score)
                                counter_some_neg += 1
                            else:
                                sum_neg += score
                                counter_neg += 1

        sum_pos = sum_pos/(max_score * counter_pos)
        sum_neg = 1-(sum_neg/(max_score * counter_neg))
        sum_some_neg = 1-(sum_some_neg/(max_score * counter_some_neg * max_score))

        reliability_score.append((sum_pos + sum_neg + sum_some_neg)/3)
        
    return reliability_score

def reliability_local_normalization(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs):
    '''
    Getting the reliability sum as currently defined
    '''
    reliability_score = []
    for subgraph in subgraphs:
        
        max_score = 0
        counter_neg = 0
        counter_pos = 0
        sum_neg = 0
        sum_pos = 0
        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            score = score * (-1)
                            if score > max_score:
                                max_score = score

                            if ((h,r,t) in all_triples):
                                sum_pos += score
                                counter_pos += 1
                            else:
                                sum_neg += score
                                counter_neg += 1

        sum_pos = sum_pos/(max_score * counter_pos)
        sum_neg = 1-(sum_neg/(max_score * counter_neg))

        reliability_score.append((sum_pos + sum_neg)/2)
        
    return reliability_score

def reliability_global_normalization(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs):
    '''
    Getting the reliability sum as currently defined
    '''
    reliability_score = []
    neg_sum_list = []
    pos_sum_list = []
    neg_counter_list = []
    pos_counter_list = []

    max_score = 0

    for subgraph in subgraphs:
        
        counter_neg = 0
        counter_pos = 0
        sum_neg = 0
        sum_pos = 0
        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            score = score * (-1)
                            if score > max_score:
                                max_score = score

                            if ((h,r,t) in all_triples):
                                sum_pos += score
                                counter_pos += 1
                            else:
                                sum_neg += score
                                counter_neg += 1
        neg_sum_list.append(sum_neg)
        pos_sum_list.append(sum_pos)
        neg_counter_list.append(counter_neg)
        pos_counter_list.append(counter_pos)
    
    for i in range(len(subgraphs)):
        sum_pos_final = pos_sum_list[i]/(max_score * pos_counter_list[i])
        sum_neg_final = 1-(neg_sum_list[i]/(max_score * neg_counter_list[i]))
        reliability_score.append((sum_pos_final + sum_neg_final)/2)
    return reliability_score

def reliability(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs, checkScore=False):
    '''
    Getting the reliability sum as currently defined
    '''
    
    reliability_score = []
    for subgraph in subgraphs:
        
        max_score = 0
        first = True
        min_score = 0
        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            if first:
                                first = False
                                min_score = score
                                max_score = score
                            if min_score > score:
                                min_score = score
                            if max_score < score:
                                max_score = score
                            if checkScore:
                                checkTransEScore(score, h,r,t, emb_train_triples, entity2embedding, relation2embedding)
                            
        # Getting the reliability sum as currently defined
        # But with already using sigmoid function to make values closer to each other
        sum = 0
        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            if checkScore:
                                checkTransEScore(score, h,r,t, emb_train_triples, entity2embedding, relation2embedding)
                            score = score / min_score
                            score = expit(score)
                            if ((h,r,t) in all_triples):
                                sum += 1-score
                            else:
                                sum += score
        reliability_score.append(sum)
        
    return reliability_score

def reliabilityNonSig(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs, checkScore=False):
    '''
    Getting the reliability sum without using sigmoid on the score function, for labels
    '''
    
    reliability_score = []
    for subgraph in subgraphs:                            
        sum = 0
        for h in range(emb_train_triples.num_entities):
            if emb_train_triples.entity_id_to_label[h] in subgraph:
                for t in range(emb_train_triples.num_entities):
                    if emb_train_triples.entity_id_to_label[t] in subgraph:
                        for r in range(emb_train_triples.num_relations):
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            if checkScore:
                                checkTransEScore(score, h,r,t, emb_train_triples, entity2embedding, relation2embedding)
                            if ((h,r,t) in all_triples):
                                sum += -score
                            else:
                                sum += score
        reliability_score.append(sum)
        
    return reliability_score

def reliabilityLabelNonSig(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs, checkScore=False):
    reliability_score = []
    for subgraph in subgraphs:
        reliability_score_label = dict()                            
        for r in range(emb_train_triples.num_relations):
            sum = 0
            for h in range(emb_train_triples.num_entities):
                if emb_train_triples.entity_id_to_label[h] in subgraph:
                    for t in range(emb_train_triples.num_entities):
                        if emb_train_triples.entity_id_to_label[t] in subgraph:
                            
                                ten = torch.tensor([[h,r,t]])
                                score = model.score_hrt(ten)
                                score = score.detach().numpy()[0][0]
                                if checkScore:
                                    checkTransEScore(score, h,r,t, emb_train_triples, entity2embedding, relation2embedding)
                                if ((h,r,t) in all_triples):
                                    sum += -score
                                else:
                                    sum += score
            reliability_score_label[emb_train_triples.relation_id_to_label[r]] = sum
        reliability_score.append(reliability_score_label)
        
    return reliability_score

def reliabilityLabel(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs, checkScore=False):
    '''
    Getting the reliability sum as currently defined for labels
    '''
    reliability_score = []
    for subgraph in subgraphs:
        reliability_score_label = dict()
        for r in range(emb_train_triples.num_relations):
            max_score = 0
            first = True
            min_score = 0
            for h in range(emb_train_triples.num_entities):
                if emb_train_triples.entity_id_to_label[h] in subgraph:
                    for t in range(emb_train_triples.num_entities):
                        if emb_train_triples.entity_id_to_label[t] in subgraph:
                        
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            if first:
                                first = False
                                min_score = score
                                max_score = score
                            if min_score > score:
                                min_score = score
                            if max_score < score:
                                max_score = score
                            if checkScore:
                                checkTransEScore(score, h,r,t, emb_train_triples, entity2embedding, relation2embedding)
        for r in range(emb_train_triples.num_relations):
            sum = 0
            for h in range(emb_train_triples.num_entities):
                if emb_train_triples.entity_id_to_label[h] in subgraph:
                    for t in range(emb_train_triples.num_entities):
                        if emb_train_triples.entity_id_to_label[t] in subgraph:
                            ten = torch.tensor([[h,r,t]])
                            score = model.score_hrt(ten)
                            score = score.detach().numpy()[0][0]
                            if checkScore:
                                checkTransEScore(score, h,r,t, emb_train_triples, entity2embedding, relation2embedding)
                            score = score / min_score
                            score = expit(score)
                            if ((h,r,t) in all_triples):
                                sum += 1-score
                            else:
                                sum += score
            reliability_score_label[emb_train_triples.relation_id_to_label[r]] = sum
        reliability_score.append(reliability_score_label)
        
    return reliability_score