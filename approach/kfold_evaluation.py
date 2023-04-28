import torch
import numpy as np
import csv
import pandas as pd
import os
import timeit

import matplotlib.pyplot as plt
from collections import defaultdict
from pykeen.triples import TriplesFactory

import embedding as emb
import settings as sett
import datahandler as dh
import classifier as cla

import networkx as nx
import dsdm as dsd
from typing import Callable, cast
import itertools
from collections import Counter
import random

LOWEST_RANK = 1000000

def getKFoldEmbeddings():
    models = []
    LP_triples_pos = []
    LP_triples_neg = []
    emb_train = []
    emb_test = []
    print(sett.DATASETNAME)
    all_triples, all_triples_set, entity_to_id_map, relation_to_id_map, test_triples, validation_triples = emb.getDataFromPykeen(datasetname=sett.DATASETNAME)
    full_dataset = torch.cat((all_triples, test_triples.mapped_triples, validation_triples.mapped_triples))

    for i in range(sett.N_SPLITS):
        save = f"KFold/{sett.EMBEDDING_TYPE}_{sett.DATASETNAME}_{sett.N_SPLITS}_fold/{sett.DATASETNAME}_{i}th"
        models.append(emb.loadModel(save))
        emb_triples_id, LP_triples_id = dh.loadKFoldSplit(i, n_split=sett.N_SPLITS)
        emb_triples = full_dataset[emb_triples_id]
        LP_triples = full_dataset[LP_triples_id]
        emb_train_triples = TriplesFactory(emb_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
        emb_train.append(emb_train_triples)
        emb_test_triples = TriplesFactory(LP_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
        emb_test.append(emb_test_triples)
        LP_triples_pos.append(LP_triples.tolist())

        neg_triples = dh.loadTriples(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold/{i}th_neg")
        LP_triples_neg.append(neg_triples)
    
    if sett.EMBEDDING_TYPE == 'TransE':
        entity2embedding, relation2embedding = emb.createEmbeddingMaps_TransE(models[0], emb_train_triples)
    elif sett.EMBEDDING_TYPE == 'DistMult':
        entity2embedding, relation2embedding = emb.createEmbeddingMaps_DistMult(models[0], emb_train_triples)
    else:
        entity2embedding, relation2embedding = emb.createEmbeddingMaps_DistMult(models[0], emb_train_triples)
    
    return models, all_triples, all_triples_set, LP_triples_pos, LP_triples_neg, emb_train, entity2embedding, relation2embedding, entity_to_id_map, relation_to_id_map

def makeTCPart(LP_triples_pos, LP_triples_neg, entity2embedding, relation2embedding, subgraphs, emb_train_triples):
    X_train, X_test, y_train, y_test = cla.prepareTrainTestData(LP_triples_pos, LP_triples_neg, emb_train_triples)
    clf = cla.trainClassifier(X_train, y_train, entity2embedding, relation2embedding)
    LP_test_score = cla.testClassifierSubgraphs(clf, X_test, y_test, entity2embedding, relation2embedding, subgraphs)

    return LP_test_score

def findingRank(orderedList, key):
    counter = 1
    for ele in orderedList:
        if key[0] == ele[0] and key[1] == ele[1]:
            return counter
        counter += 1
    return None

def findingRankNegHead(orderedList, key, all_triples_set, fix):
    counter = 1
    for ele in orderedList:
        if key[0] == ele[0] and key[1] == ele[1]:
            return counter
        tup = (fix,ele[0],ele[1])
        if tup in all_triples_set:
            continue
        counter += 1
    return None

def findingRankNegTail(orderedList, key, all_triples_set, fix):
    counter = 1
    for ele in orderedList:
        if key[0] == ele[0] and key[1] == ele[1]:
            return counter
        tup = (ele[0],ele[1],fix)
        if tup in all_triples_set:
            continue
        counter += 1
    return None

def overlapHead(all_triples_set, ranking, head, degree):
    count = 0
    deg = degree
    for ele in ranking:
        tup = (head,ele[0],ele[1])
        if tup in all_triples_set:
            count += 1
        deg -= 1
        if deg == 0:
            return count
    return count

def overlapRelation(all_triples_set, ranking, relation, degree):
    count = 0
    deg = degree
    for ele in ranking:
        if (ele[0],relation,ele[1]) in all_triples_set:
            count += 1
        deg -= 1
        if deg == 0:
            return count
    return count

def overlapTail(all_triples_set, ranking, tail, degree):
    count = 0
    deg = degree
    for ele in ranking:
        if (ele[0],ele[1],tail) in all_triples_set:
            count += 1
        deg -= 1
        if deg == 0:
            return count
    return count

def KFoldNegGen():
    models, all_triples, all_triples_set, LP_triples_pos, LP_triples_neg, emb_train, entity2embedding, relation2embedding, entity_to_id_map, relation_to_id_map = getKFoldEmbeddings()

    for i in range(sett.N_SPLITS):
        neg_triples, throw = dh.createNegTripleHT(all_triples_set, LP_triples_pos[i], emb_train[i])
        dh.storeTriples(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold/{i}th_neg", neg_triples)

def run_eval(source: str = 'None'):
    models, all_triples, all_triples_set, LP_triples_pos, LP_triples_neg, emb_train, entity2embedding, relation2embedding, entity_to_id_map, relation_to_id_map = getKFoldEmbeddings()
    
    full_graph = TriplesFactory(all_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
    df = pd.DataFrame(full_graph.triples, columns=['subject', 'predicate', 'object'])

    
    start = timeit.default_timer()
    #subgraphs = dh.loadSubGraphs(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold")
    if source == 'None':
        subgraphs = dh.loadSubGraphsEmb(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold")
    else:
        subgraphs = dh.loadSubGraphsEmbSel(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold", source)

    used_ent = set()

    for subgraph in subgraphs:
        used_ent = used_ent.union(subgraph)
    
    end = timeit.default_timer()
    print(f'Subgraphs are done in time: {end-start}')

    score_tail = []
    score_rel = []
    score_cla = []

    for i in range(sett.N_SPLITS):
        '''
        start = timeit.default_timer()
        LP_test_score_tail, start_time_clf_training, end_time_clf_training, start_time_LP_score, end_time_LP_score = emb.baselineLP_tail(models[i], subgraphs, emb_train[i], LP_triples_pos[i], all_triples_set)
        score_tail.append(LP_test_score_tail)
        end = timeit.default_timer()
        print(f'Tail prediction for fold {i} has taken {end-start}')

        start = timeit.default_timer()
        LP_test_score_rel, start_time_clf_training, end_time_clf_training, start_time_LP_score, end_time_LP_score = emb.baselineLP_relation(models[i], subgraphs, emb_train[i], LP_triples_pos[i], all_triples_set)
        score_rel.append(LP_test_score_rel)
        end = timeit.default_timer()
        print(f'Relation prediction for fold {i} has taken {end-start}')
        '''
        start = timeit.default_timer()
        LP_test_score = makeTCPart(LP_triples_pos[i],  LP_triples_neg[i], entity2embedding, relation2embedding, subgraphs, emb_train[i])
        score_cla.append(LP_test_score)
        end = timeit.default_timer()
        print(f'Triple Classification for fold {i} has taken {end-start}')


    print(f"finished scores of {sett.EMBEDDING_TYPE}")

    model_deg_rank_score = []
    model_deg_rank_score_log = []
    model_deg_score = []
    model_siblings_degree_score = []
    model_siblings_score = []
    model_siblings_log_score = []

    model_siblings_degree_score_ent = []
    model_siblings_score_ent = []
    model_siblings_log_score_ent = []

    model_relation_sum_at_1 = []
    model_relation_sum_at_5 = []
    model_relation_sum_at_10 = []
    model_relation_sum_for_MRR = []

    model_tail_sum_at_1 = []
    model_tail_sum_at_5 = []
    model_tail_sum_at_10 = []
    model_tail_sum_for_MRR = []

    for i in range(sett.N_SPLITS):
        model_deg_rank_score.append([])
        model_deg_rank_score_log.append([])
        model_deg_score.append([])
        model_siblings_degree_score.append([])
        model_siblings_score.append([])
        model_siblings_log_score.append([])

        model_siblings_degree_score_ent.append([])
        model_siblings_score_ent.append([])
        model_siblings_log_score_ent.append([])

        model_relation_sum_at_1.append([])
        model_relation_sum_at_5.append([])
        model_relation_sum_at_10.append([])
        model_relation_sum_for_MRR.append([])

        model_tail_sum_at_1.append([])
        model_tail_sum_at_5.append([])
        model_tail_sum_at_10.append([])
        model_tail_sum_for_MRR.append([])


    first_relation = True
    first_tail = True
    sbg = -1
    print()
    print(f'Iterations per subgraph start now!')
    for subgraph in subgraphs:
        print()
        iteration_start = timeit.default_timer()
        subgraph_list = list(subgraph)
        sbg += 1
        outDegree = dict()
        occurences = defaultdict(int)
        inDegree = dict()

        start = timeit.default_timer()
        for t in df.values:
            
            if (t[0] in subgraph) and (t[2] in subgraph):
                if entity_to_id_map[t[0]] in outDegree.keys():
                    outDegree[entity_to_id_map[t[0]]] = 1 + outDegree[entity_to_id_map[t[0]]]
                else:
                    outDegree[entity_to_id_map[t[0]]] = 1

                if entity_to_id_map[t[2]] in inDegree.keys():
                    inDegree[entity_to_id_map[t[2]]] = 1 + inDegree[entity_to_id_map[t[2]]]
                else:
                    inDegree[entity_to_id_map[t[2]]] = 1

                if entity_to_id_map[t[2]] not in outDegree.keys():
                    outDegree[entity_to_id_map[t[2]]] = 0

                if entity_to_id_map[t[0]] not in inDegree.keys():
                    inDegree[entity_to_id_map[t[0]]] = 1
                
                occurences[relation_to_id_map[t[1]]] = 1 + occurences[relation_to_id_map[t[1]]]
        end = timeit.default_timer()
        print(f'Degree Mapping complete in {end-start}')

        HeadModelsRanking = []
        RelationModelsRanking = []
        TailModelsRanking = []

        HeadModelRank = []
        RelationModelRank = []
        TailModelRank = []
        for i in range(sett.N_SPLITS):
            HeadModelsRanking.append(dict())
            RelationModelsRanking.append(defaultdict(dict))
            TailModelsRanking.append(defaultdict(dict))

            HeadModelRank.append(dict())
            RelationModelRank.append(dict())
            TailModelRank.append(dict())

        start = timeit.default_timer()
        for he in list(subgraph_list):
                head = entity_to_id_map[he]
                for i in range(sett.N_SPLITS):
                    HeadModelsRanking[i][head] = dict()
                for relation in occurences.keys():
                    if first_relation:
                        for i in range(sett.N_SPLITS):
                            RelationModelsRanking[i][relation] = dict()

                    for ta in list(subgraph_list):
                            tail = entity_to_id_map[ta]
                            if first_tail:
                                for i in range(sett.N_SPLITS):
                                    TailModelsRanking[i][tail] = dict()
                            
                            ten = torch.tensor([[head,relation,tail]])
                            for i in range(sett.N_SPLITS):
                                score = models[i].score_hrt(ten)
                                score = score.detach().numpy()[0][0]

                                HeadModelsRanking[i][head][(relation,tail)] = score
                                RelationModelsRanking[i][relation][(head,tail)] = score
                                TailModelsRanking[i][tail][(head,relation)] = score

                    first_tail = False
                first_relation = False
        end = timeit.default_timer()
        print(f'Ranking Mapping complete {end-start}')

        start = timeit.default_timer()
        for he in list(subgraph_list):
                head = entity_to_id_map[he]
                for i in range(sett.N_SPLITS):
                    #print(HeadModelRank[i])
                    #print(HeadModelsRanking[i])
                    HeadModelRank[i][head] = list(dict(sorted(HeadModelsRanking[i][head].items(), key=lambda item: item[1], reverse=True)).keys())

                    TailModelRank[i][head] = list(dict(sorted(TailModelsRanking[i][head].items(), key=lambda item: item[1], reverse=True)).keys())
        
        for relation in occurences.keys():
            for i in range(sett.N_SPLITS):
                RelationModelRank[i][relation] = list(dict(sorted(RelationModelsRanking[i][relation].items(), key=lambda item: item[1], reverse=True)).keys())
        end = timeit.default_timer()
        print(f'Ranking Sorting complete {end-start}')

        #print(f"finished ranking dictionaries of {sett.EMBEDDING_TYPE} and sbg {sbg}")

        counter = 0
        counterEnt = 0
        counterSib = 0
        sums_deg_rank = []
        sums_deg_rank_log = []
        sums_deg = []
        sums_siblings_degree = []
        sums_siblings = []
        sums_siblings_log = []

        for i in range(sett.N_SPLITS):
            sums_deg_rank.append(0)
            sums_deg_rank_log.append(0)
            sums_deg.append(0)
            sums_siblings_degree.append(0)
            sums_siblings.append(0)
            sums_siblings_log.append(0)

        start = timeit.default_timer()
        for he in list(subgraph_list):
            h = entity_to_id_map[he]
            if h in outDegree.keys():
                for ta in list(subgraph_list):
                    t = entity_to_id_map[ta]
                    if t in inDegree.keys():
                        for r in occurences.keys():
                            counter += 1
                            if (h,r,t) in all_triples_set:
                                counterSib += 1
                            for i in range(sett.N_SPLITS):
                                hRank = findingRank(HeadModelRank[i][h],(r,t))
                                rRank = findingRank(RelationModelRank[i][r],(h,t))
                                tRank = findingRank(TailModelRank[i][t],(h,r))

                                hRankNeg = findingRankNegHead(HeadModelRank[i][h],(r,t),all_triples_set,h)
                                tRankNeg = findingRankNegTail(TailModelRank[i][t],(h,r),all_triples_set,t)

                                if (h,r,t) in all_triples_set:
                                    sums_deg_rank[i] += ( (1/(max(hRank - outDegree[h] + 1,1))) + (1/(max(rRank - occurences[r] + 1,1))) + (1/(max(tRank - inDegree[t] + 1,1))) )/3
                                    sums_deg_rank_log[i] += np.log( 1+ 1/(max(hRank - outDegree[h] + 1,1)) + 1/(max(rRank - occurences[r] + 1,1)) + 1/(max(tRank - inDegree[t] + 1,1)) )
                                    sums_siblings_degree[i] += ( (1/(max(hRank - outDegree[h] + 1,1))) + (1/(max(tRank - inDegree[t] + 1,1))) )/2
                                    sums_siblings[i] += ( (1/hRankNeg) + (1/tRankNeg) )/2
                                    sums_siblings_log[i] += np.log( 1+ (1/hRankNeg) + (1/tRankNeg) )
                                else:
                                    sums_deg_rank[i] += ( 1-(1/(max(hRank - outDegree[h] + 1,1))) + 1-(1/(max(rRank - occurences[r] + 1,1))) + 1-(1/(max(tRank - inDegree[t] + 1,1))) )/3
                                    sums_deg_rank_log[i] += np.log( 1+ 1-(1/(max(hRank - outDegree[h] + 1,1)) ) + 1-(1/(max(rRank - occurences[r] + 1,1)) ) + 1-(1/(max(tRank - inDegree[t] + 1,1)) ) )

            
                counterEnt += 1
                for i in range(sett.N_SPLITS):
                    if inDegree[h] == 0 and outDegree[h] == 0:
                        counterEnt -= 1
                    elif inDegree[h] == 0:
                        sums_deg[i] += (overlapTail(all_triples_set, TailModelRank[i][h], h, outDegree[h])/outDegree[h])
                    elif outDegree[h] == 0:
                        sums_deg[i] += (overlapHead(all_triples_set, HeadModelRank[i][h], h, inDegree[h])/inDegree[h])
                    else:
                        sums_deg[i] += ( (overlapHead(all_triples_set, HeadModelRank[i][h], h, inDegree[h])/inDegree[h]) + (overlapTail(all_triples_set, TailModelRank[i][h], h, outDegree[h])/outDegree[h]) )/2
        end = timeit.default_timer()
        print(f'Getting Score Sums is done in {end-start}')

        start = timeit.default_timer()
        for i in range(sett.N_SPLITS):
            if counter == 0:
                model_deg_rank_score[i].append(-100)
                model_deg_rank_score_log[i].append(-100)
            else:
                model_deg_rank_score[i].append(sums_deg_rank[i]/counter)
                model_deg_rank_score_log[i].append(sums_deg_rank_log[i]/counter)
            
            if counterEnt == 0:
                model_deg_score[i].append(-100)
            else:
                model_deg_score[i].append(sums_deg[i]/counterEnt)
            
            
            if counterSib == 0:
                model_siblings_degree_score[i].append(-100)
                model_siblings_score[i].append(-100)
                model_siblings_log_score[i].append(-100)
            else:
                model_siblings_degree_score[i].append(sums_siblings_degree[i]/counterSib)
                model_siblings_score[i].append(sums_siblings[i]/counterSib)
                model_siblings_log_score[i].append(sums_siblings_log[i]/counterSib)

                model_siblings_degree_score_ent[i].append(sums_siblings_degree[i]/len(subgraph))
                model_siblings_score_ent[i].append(sums_siblings[i]/len(subgraph))
                model_siblings_log_score_ent[i].append(sums_siblings_log[i]/len(subgraph))
        end = timeit.default_timer()
        print(f'Dividing Scores is done in {end-start}')
        
        start = timeit.default_timer()
        counter_of_test_tp = []
        for i in range(sett.N_SPLITS):
            relation_sum_at_1 = 0
            relation_sum_at_5 = 0
            relation_sum_at_10 = 0
            relation_sum_for_MRR = 0

            tail_sum_at_1 = 0
            tail_sum_at_5 = 0
            tail_sum_at_10 = 0
            tail_sum_for_MRR = 0
            counter_of_test_tp = 0
            for tp in LP_triples_pos[i]:
                counter = 0
                if (emb_train[i].entity_id_to_label[tp[0]] in subgraph) and (emb_train[0].entity_id_to_label[tp[2]] in subgraph):
                    counter_of_test_tp += 1

                    tmp_scores = dict()
                    for relation in range(emb_train[0].num_relations):
                        tup = (tp[0],relation,tp[2])
                        if tup in all_triples_set and relation != tp[1]:
                            continue
                        ten = torch.tensor([[tp[0],relation,tp[2]]])
                        score = models[i].score_hrt(ten)
                        score = score.detach().numpy()[0][0]
                        tmp_scores[relation] = score
                    sl = sorted(tmp_scores.items(), key=lambda x:x[1], reverse=True)
                    
                    for pair in sl:
                            counter += 1
                            if pair[0] == tp[1]:
                                if counter <= 1:
                                    relation_sum_at_1 += 1
                                    relation_sum_at_5 += 1
                                    relation_sum_at_10 += 1
                                elif counter <= 5:
                                    relation_sum_at_5 += 1
                                    relation_sum_at_10 += 1
                                elif counter <= 10:
                                    relation_sum_at_10 += 1
                                relation_sum_for_MRR += 1/counter
                                break

                    tmp_scores = dict()
                    for tail in range(emb_train[0].num_entities):
                        tup = (tp[0],tp[1],tail)
                        if tup in all_triples_set and tail != tp[2]:
                            continue
                        ten = torch.tensor([[tp[0],tp[1],tail]])
                        score = models[i].score_hrt(ten)
                        score = score.detach().numpy()[0][0]
                        tmp_scores[tail] = score

                    sl = sorted(tmp_scores.items(), key=lambda x:x[1], reverse=True)
                        
                    for pair in sl:
                            counter += 1
                            if pair[0] == tp[2]:
                                if counter <= 1:
                                    tail_sum_at_1 += 1
                                    tail_sum_at_5 += 1
                                    tail_sum_at_10 += 1
                                elif counter <= 5:
                                    tail_sum_at_5 += 1
                                    tail_sum_at_10 += 1
                                elif counter <= 10:
                                    tail_sum_at_10 += 1
                                tail_sum_for_MRR += 1/counter
                                break
            if counter_of_test_tp > 0:
                model_tail_sum_at_1[i].append(tail_sum_at_1/counter_of_test_tp)
                model_tail_sum_at_5[i].append(tail_sum_at_5/counter_of_test_tp)
                model_tail_sum_at_10[i].append(tail_sum_at_10/counter_of_test_tp)
                model_tail_sum_for_MRR[i].append(tail_sum_for_MRR/counter_of_test_tp)

                model_relation_sum_at_1[i].append(relation_sum_at_1/counter_of_test_tp)
                model_relation_sum_at_5[i].append(relation_sum_at_5/counter_of_test_tp)
                model_relation_sum_at_10[i].append(relation_sum_at_10/counter_of_test_tp)
                model_relation_sum_for_MRR[i].append(relation_sum_for_MRR/counter_of_test_tp)
            else:
                model_relation_sum_at_1[i].append(-100)
                model_relation_sum_at_5[i].append(-100)
                model_relation_sum_at_10[i].append(-100)
                model_relation_sum_for_MRR[i].append(-100)

                model_tail_sum_at_1[i].append(-100)
                model_tail_sum_at_5[i].append(-100)
                model_tail_sum_at_10[i].append(-100)
                model_tail_sum_for_MRR[i].append(-100)
        end = timeit.default_timer()
        print(f'Hit@ and MRR are done {end-start}')

        iteration_end = timeit.default_timer()
        print(f'Iteration for subgraph {sbg} took {iteration_end-iteration_start}')
        print()



    path = f'approach/scoreData/KFold/{sett.DATASETNAME}_{sett.EMBEDDING_TYPE}'
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)
    if source == 'None':
        for i in range(sett.N_SPLITS):
            c = open(f'{path}/kfold_relative_reliability_of_{i}th_fold_sbg_approx{sett.SIZE_OF_SUBGRAPHS}.csv', "w")
            writer = csv.writer(c)
            data = ['subgraph', 'triple classification score', 'degree + rank', 'degree + rank + log', 'degree', 'siblings degree', 'siblings', 'siblings log','Tail Hit @ 1','Tail Hit @ 5','Tail Hit @ 10','Tail MRR','Relation Hit @ 1','Relation Hit @ 5','Relation Hit @ 10','Relation MRR', 'siblings degree Ent', 'siblings Ent', 'siblings log Ent']
            writer.writerow(data)
            for j in range(len(score_cla[i])):
                data = [j, score_cla[i][j], model_deg_rank_score[i][j], model_deg_rank_score_log[i][j], model_deg_score[i][j], model_siblings_degree_score[i][j], model_siblings_score[i][j], model_siblings_log_score[i][j]]
                tmp = [model_tail_sum_at_1[i][j],model_tail_sum_at_5[i][j],model_tail_sum_at_10[i][j],model_tail_sum_for_MRR[i][j],model_relation_sum_at_1[i][j],model_relation_sum_at_5[i][j],model_relation_sum_at_10[i][j],model_relation_sum_for_MRR[i][j], model_siblings_degree_score_ent[i][j], model_siblings_score_ent[i][j], model_siblings_log_score_ent[i][j]]
                data.extend(tmp)
                writer.writerow(data)
            c.close()
    else:
        for i in range(sett.N_SPLITS):
            c = open(f'{path}/kfold_relative_reliability_in_{sett.EMBEDDING_TYPE}_of_{i}th_fold_dense_sbg_of_{sett.SIZE_OF_SUBGRAPHS}_of_{source}.csv', "w")
            writer = csv.writer(c)
            data = ['subgraph', 'triple classification score', 'degree + rank', 'degree + rank + log', 'degree', 'siblings degree', 'siblings', 'siblings log','Tail Hit @ 1','Tail Hit @ 5','Tail Hit @ 10','Tail MRR','Relation Hit @ 1','Relation Hit @ 5','Relation Hit @ 10','Relation MRR', 'siblings degree Ent', 'siblings Ent', 'siblings log Ent']
            writer.writerow(data)
            for j in range(len(score_cla[i])):
                data = [j, score_cla[i][j], model_deg_rank_score[i][j], model_deg_rank_score_log[i][j], model_deg_score[i][j], model_siblings_degree_score[i][j], model_siblings_score[i][j], model_siblings_log_score[i][j]]
                tmp = [model_tail_sum_at_1[i][j],model_tail_sum_at_5[i][j],model_tail_sum_at_10[i][j],model_tail_sum_for_MRR[i][j],model_relation_sum_at_1[i][j],model_relation_sum_at_5[i][j],model_relation_sum_at_10[i][j],model_relation_sum_for_MRR[i][j], model_siblings_degree_score_ent[i][j], model_siblings_score_ent[i][j], model_siblings_log_score_ent[i][j]]
                data.extend(tmp)
                writer.writerow(data)
            c.close()
    
def run_mod(source: str) -> None:
    run_eval(source)

def getkHopSiblingScore(u: str, v: str, M: nx.MultiDiGraph, models: list[object], entity_to_id_map: object, relation_to_id_map: object, all_triples_set: set[tuple[int,int,int]], alltriples: TriplesFactory) -> float:

    subgraph_list, labels, existing, count, ex_triples  = dh.getkHopneighbors(u,v,M)

    allset_u = set(itertools.product([u],labels,subgraph_list))
    allset_v = set(itertools.product(subgraph_list,labels,[v]))
    allset = allset_v.union(allset_u)
    #alllist = list(allset)
    possible = len(allset)
    print(f'We have {count} existing, {possible} possible, worst rank is {possible-count+1}')
    selectedComparators = set()
    if (possible-count) <= 100:
        selectedComparators = set(itertools.product([u],labels,subgraph_list))
    else:
        while (len(selectedComparators) < sett.SAMPLE * (possible-count) or len(selectedComparators) < 100):# and len(selectedComparators) < 1000:
            tp = random.choice(list(allset))
            if tp not in ex_triples:
                selectedComparators.add(tp)
            allset.remove(tp)


    HeadModelRank = []
    TailModelRank = []

    getScoreList = list(selectedComparators.union(ex_triples))

    for i in range(sett.N_SPLITS):
        HeadModelRank.append(dict())
        TailModelRank.append(dict())

    head = entity_to_id_map[u]
    tail = entity_to_id_map[v]
    for tp in getScoreList:
        h = entity_to_id_map[tp[0]]
        rel = relation_to_id_map[tp[1]]
        t = entity_to_id_map[tp[2]]
        ten = torch.tensor([[h,rel,t]])
        if h == head:
            for i in range(sett.N_SPLITS):
                score = models[i].score_hrt(ten)
                score = score.detach().numpy()[0][0]
                HeadModelRank[i][(rel,t)] = score
        if t == tail:
            for i in range(sett.N_SPLITS):
                score = models[i].score_hrt(ten)
                score = score.detach().numpy()[0][0]
                TailModelRank[i][(h,rel)] = score
    
    hRankNeg = 0
    tRankNeg = 0

    for label in existing:
        relation = relation_to_id_map[label]
        
        for i in range(sett.N_SPLITS):
            part1 = list(dict(sorted(HeadModelRank[i].items(), key=lambda item: item[1], reverse=True)).keys())
                
            part2 = list(dict(sorted(TailModelRank[i].items(), key=lambda item: item[1], reverse=True)).keys())

            hRankNeg += findingRankNegHead(part1,(relation,tail),all_triples_set,head) / sett.N_SPLITS
            tRankNeg += findingRankNegTail(part2,(head,relation),all_triples_set,tail) / sett.N_SPLITS

    hRankNeg = hRankNeg/len(existing)
    tRankNeg = tRankNeg/len(existing)

    possible = alltriples.num_entities * alltriples.num_relations * 2
    c1 = Counter(elem[0] for elem in list(all_triples_set))
    c2 = Counter(elem[2] for elem in list(all_triples_set))

    all_c = c1[head] + c2[tail]
    
    return ( (1/(hRankNeg +( (possible-all_c+1)-len(selectedComparators) ) )) + (1/(tRankNeg + ( (possible-all_c+1)-len(selectedComparators)) )) )/2

def binomial(u: str, v: str, M: nx.MultiDiGraph, models: list[object], entity_to_id_map: object, relation_to_id_map: object, all_triples_set: set[tuple[int,int,int]], alltriples: TriplesFactory) -> float:
    
    subgraph_list, labels, existing, count, ex_triples  = dh.getkHopneighbors(u,v,M)

    subgraphs = dh.loadSubGraphs(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold")
    m = []
    for el in list(subgraphs[0]):
        m.append(entity_to_id_map[el])
    allset_u = set(itertools.product([entity_to_id_map[u]],range(alltriples.num_relations),m))
    allset_v = set(itertools.product(m,range(alltriples.num_relations),[entity_to_id_map[v]]))

    #allset_u = set(itertools.product([entity_to_id_map[u]],range(alltriples.num_relations),range(alltriples.num_entities)))
    #allset_v = set(itertools.product(range(alltriples.num_entities),range(alltriples.num_relations),[entity_to_id_map[v]]))
    allset = allset_v.union(allset_u)
    allset = allset.difference(all_triples_set)
    
    #alllist = list(allset)
    possible = len(allset)
    print(f'We have {count} existing, {possible} possible, worst rank is {possible-count+1}')
    selectedComparators = set(random.choices(list(allset), k=max( min(100,len(allset)), int((sett.SAMPLE*len(allset))//1) )))

    HeadModelRank = []
    TailModelRank = []

    ex_triples_new = set()
    for tp in list(ex_triples):
        ex_triples_new.add( (entity_to_id_map[tp[0]], relation_to_id_map[tp[1]], entity_to_id_map[tp[2]]) )
    
    getScoreList = list(selectedComparators.union(ex_triples_new))

    u_comp = allset_u.intersection(selectedComparators)
    v_comp = allset_v.intersection(selectedComparators)

    for i in range(sett.N_SPLITS):
        HeadModelRank.append(dict())
        TailModelRank.append(dict())

    head = entity_to_id_map[u]
    tail = entity_to_id_map[v]
    for tp in getScoreList:
        h = tp[0]
        rel = tp[1]
        t = tp[2]
        ten = torch.tensor([[h,rel,t]])
        if h == head:
            for i in range(sett.N_SPLITS):
                score = models[i].score_hrt(ten)
                score = score.detach().numpy()[0][0]
                HeadModelRank[i][(rel,t)] = score
        if t == tail:
            for i in range(sett.N_SPLITS):
                score = models[i].score_hrt(ten)
                score = score.detach().numpy()[0][0]
                TailModelRank[i][(h,rel)] = score
    
    hRankNeg = 0
    tRankNeg = 0

    for label in existing:
        relation = relation_to_id_map[label]
        
        for i in range(sett.N_SPLITS):
            part1 = list(dict(sorted(HeadModelRank[i].items(), key=lambda item: item[1], reverse=True)).keys())
                
            part2 = list(dict(sorted(TailModelRank[i].items(), key=lambda item: item[1], reverse=True)).keys())

            
            pos = findingRankNegHead(part1,(relation,tail),all_triples_set,head) / sett.N_SPLITS
            hRankNeg += (pos / len(u_comp)) * len(allset_u)
            neg = findingRankNegTail(part2,(head,relation),all_triples_set,tail) / sett.N_SPLITS
            tRankNeg += (neg / len(v_comp)) * len(allset_v)

    hRankNeg = hRankNeg/len(existing)
    tRankNeg = tRankNeg/len(existing)
    
    return ( 1/hRankNeg + 1/tRankNeg )/2

def triangle(u: str, v: str, M: nx.MultiDiGraph, models: list[object], entity_to_id_map: object, relation_to_id_map: object, all_triples_set: set[tuple[int,int,int]], alltriples: TriplesFactory) -> float:

    subgraph_list, labels, existing, count, ex_triples  = dh.getTriangle(u,v,M)

    subgraph_list_new = []
    for e in subgraph_list:
        subgraph_list_new.append(entity_to_id_map[e])
    
    allset_u = set(itertools.product([entity_to_id_map[u]],range(alltriples.num_relations),subgraph_list_new))
    allset_v = set(itertools.product(subgraph_list_new,range(alltriples.num_relations),[entity_to_id_map[v]]))
    allset = allset_v.union(allset_u)
    allset = allset.difference(all_triples_set)
    
    #alllist = list(allset)
    possible = len(allset)
    print(f'We have {count} existing, {possible} possible, worst rank is {possible-count+1}')
    selectedComparators = set(random.choices(list(allset), k=max(min(100,len(allset)), int((sett.SAMPLE*len(allset))//1) )))

    HeadModelRank = []
    TailModelRank = []

    ex_triples_new = set()
    for tp in list(ex_triples):
        ex_triples_new.add( (entity_to_id_map[tp[0]], relation_to_id_map[tp[1]], entity_to_id_map[tp[2]]) )
    
    getScoreList = list(selectedComparators.union(ex_triples_new))

    u_comp = allset_u.intersection(selectedComparators)
    v_comp = allset_v.intersection(selectedComparators)

    for i in range(sett.N_SPLITS):
        HeadModelRank.append(dict())
        TailModelRank.append(dict())

    head = entity_to_id_map[u]
    tail = entity_to_id_map[v]
    for tp in getScoreList:
        h = tp[0]
        rel = tp[1]
        t = tp[2]
        ten = torch.tensor([[h,rel,t]])
        if h == head:
            for i in range(sett.N_SPLITS):
                score = models[i].score_hrt(ten)
                score = score.detach().numpy()[0][0]
                HeadModelRank[i][(rel,t)] = score
        if t == tail:
            for i in range(sett.N_SPLITS):
                score = models[i].score_hrt(ten)
                score = score.detach().numpy()[0][0]
                TailModelRank[i][(h,rel)] = score
    
    hRankNeg = 0
    tRankNeg = 0

    for label in existing:
        relation = relation_to_id_map[label]
        
        for i in range(sett.N_SPLITS):
            part1 = list(dict(sorted(HeadModelRank[i].items(), key=lambda item: item[1], reverse=True)).keys())
                
            part2 = list(dict(sorted(TailModelRank[i].items(), key=lambda item: item[1], reverse=True)).keys())

            
            pos = findingRankNegHead(part1,(relation,tail),all_triples_set,head) / sett.N_SPLITS
            hRankNeg += (pos / (len(u_comp)+1)) * len(allset_u)
            neg = findingRankNegTail(part2,(head,relation),all_triples_set,tail) / sett.N_SPLITS
            tRankNeg += (neg / (len(v_comp)+1)) * len(allset_v)

    hRankNeg = hRankNeg/len(existing)
    tRankNeg = tRankNeg/len(existing)
    
    return ( 1/hRankNeg + 1/tRankNeg )/2

def getSiblingScore(u: str, v: str, M: nx.MultiDiGraph, models: list[object], entity_to_id_map: object, relation_to_id_map: object, all_triples_set: set[tuple[int,int,int]], alltriples: TriplesFactory) -> float:
    #subgraphs = dh.loadSubGraphs(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold")

    head = entity_to_id_map[u]
    tail = entity_to_id_map[v]

    HeadModelRank: list[dict[tuple[int,int],float]] = []
    TailModelRank: list[dict[tuple[int,int],float]] = []

    for _ in range(sett.N_SPLITS):
        HeadModelRank.append(dict())
        TailModelRank.append(dict())

    #for entstr in list(subgraphs[0]):
    for ent in range(alltriples.num_entities):
        for rel in range(alltriples.num_relations):
            #ent = entity_to_id_map[entstr]
            ten_h = torch.tensor([[head,rel,ent]])
            ten_t = torch.tensor([[ent,rel,tail]])

            for i in range(sett.N_SPLITS):
                score = models[i].score_hrt(ten_h)
                score = score.detach().numpy()[0][0]
                HeadModelRank[i][(rel,ent)] = score

                score = models[i].score_hrt(ten_t)
                score = score.detach().numpy()[0][0]
                TailModelRank[i][(ent,rel)] = score

    hRankNeg = 0
    tRankNeg = 0

    between_labels: list[str] = []



    for el in M.get_edge_data(u,v).items():
        between_labels.append(el[1]['label'])

    for label in between_labels:
        relation = relation_to_id_map[label]
        
        for i in range(sett.N_SPLITS):
            part1 = list(dict(sorted(HeadModelRank[i].items(), key=lambda item: item[1], reverse=True)).keys())
                
            part2 = list(dict(sorted(TailModelRank[i].items(), key=lambda item: item[1], reverse=True)).keys())

            hRankNeg += dh.findingRankNegHead(part1,(relation,tail),all_triples_set,head) / sett.N_SPLITS
            tRankNeg += dh.findingRankNegTail(part2,(head,relation),all_triples_set,tail) / sett.N_SPLITS

    hRankNeg = hRankNeg/len(between_labels)
    tRankNeg = tRankNeg/len(between_labels)
    global LOWEST_RANK
    if LOWEST_RANK > hRankNeg:
        LOWEST_RANK = hRankNeg
    if LOWEST_RANK > tRankNeg:
        LOWEST_RANK = tRankNeg
    return ( (1/hRankNeg) + (1/tRankNeg) ) /2

def groundTruthWeights(score_calculation: Callable[[str, str, nx.MultiDiGraph, list[object], object, object, set[tuple[int,int,int]], TriplesFactory], float] ) -> None:
    all_triples, all_triples_set, entity_to_id_map, relation_to_id_map, test_triples, validation_triples = emb.getDataFromPykeen(datasetname=sett.DATASETNAME)
    full_dataset = torch.cat((all_triples, test_triples.mapped_triples, validation_triples.mapped_triples))
    full_graph = TriplesFactory(full_dataset,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
    df = pd.DataFrame(full_graph.triples, columns=['subject', 'predicate', 'object'])
    M = nx.MultiDiGraph()

    subgraphs = dh.loadSubGraphs(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold")

    models, all_triples, all_triples_set, LP_triples_pos, LP_triples_neg, emb_train, entity2embedding, relation2embedding, entity_to_id_map, relation_to_id_map = getKFoldEmbeddings()
    for t in df.values:
        if t[0] in subgraphs[0] and t[2] in subgraphs[0]:
            M.add_edge(t[0], t[2], label = t[1])

    G = nx.Graph()
    count = 0
    pct = 0
    start = timeit.default_timer()
    length: int = len(nx.DiGraph(M).edges())
    print(f'Starting with {length}')
    for u,v in nx.DiGraph(M).edges():
        print(f'{u} and {v}')
        w = score_calculation(u, v, M, models, entity_to_id_map, relation_to_id_map, all_triples_set, full_graph)
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
            print(w)
        count += 1
        now = timeit.default_timer()
        print(now-start)
        if count % ((length // 100)+1) == 0:
            pct += 1
            now = timeit.default_timer()
            print(f'Finished with {pct}% for {sett.EMBEDDING_TYPE} in time {now-start}, took avg of {(now-start)/pct} per point')

    weighted_graph: list[tuple[str,str,float]] = []
    for u,v,data in G.edges(data=True):
        weighted_graph.append((u,v,data['weight']))

    with open(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold/{sett.EMBEDDING_TYPE}_weightedGraph_{score_calculation.__name__}_{sett.SAMPLE}_samples.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(weighted_graph)
    '''
    H = nx.Graph()
    with open(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold/{sett.EMBEDDING_TYPE}_weightedGraph_{score_calculation.__name__}.csv", "r") as f:
        plots = csv.reader(f, delimiter=',')
        for row in plots:
            H.add_edge(str(row[0]),str(row[1]),weight=float(row[2]))
    global LOWEST_RANK
    print(LOWEST_RANK)

    flowless_R = dsd.flowless(H, 5, weight='weight')
    greedy_R = dsd.greedy_charikar(H, weight='weight')

    print(len(flowless_R[0]))
    flow = dh.createSubGraphs(full_dataset, entity_to_id_map, relation_to_id_map, number_of_graphs=1000, size_of_graphs=len(flowless_R[0]))
    flow_den = []
    flow_den.append(set(flowless_R[0]))

    print(len(greedy_R[0]))
    if len(greedy_R[0]) != len(flowless_R[0]):
        greedy = dh.createSubGraphs(full_dataset, entity_to_id_map, relation_to_id_map, number_of_graphs=1000, size_of_graphs=len(greedy_R[0]))
    greedy_den = []
    greedy_den.append(set(greedy_R[0]))
    path = f"approach/Subgraphs/"

    if score_calculation.__name__ == 'getSiblingScore':
        path += 'Exact/'
    if score_calculation.__name__ == 'getkHopSiblingScore':
        path += 'Pessimistic/'
    if score_calculation.__name__ == 'triangle':
        path += 'Triangle/'
    if score_calculation.__name__ == 'binomial':
        path += 'Binomial/'

    dh.storeDenSubGraphs(path, flow)
    if len(greedy_R[0]) != len(flowless_R[0]):
        dh.storeDenSubGraphs(path, greedy)

    path = f'{path}Dense_'
    dh.storeDenSubGraphs(path, flow_den)
    dh.storeDenSubGraphs(path, greedy_den)
    '''

def calculateMSE() -> None:

    MSE_RotatE_bin = list[float]()
    MSE_TransE_bin = list[float]()
    MSE_PairRE_bin = list[float]()
    MSE_DistMult_bin = list[float]()

    MSE_RotatE_ksib = list[float]()
    MSE_TransE_ksib = list[float]()
    MSE_PairRE_ksib = list[float]()
    MSE_DistMult_ksib = list[float]()

    MSE_RotatE_tria = list[float]()
    MSE_TransE_tria = list[float]()
    MSE_PairRE_tria = list[float]()
    MSE_DistMult_tria = list[float]()

    

    

    for eb in ['RotatE', 'TransE', 'PairRE', 'DistMult']:
        sib = list[float]()
        with open(f'approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold/{eb}_weightedGraph_getSiblingScore_0.0_samples.csv','r') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            for row in rows:
                sib.append(float(row[2]))

        for smp in [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0]:
            bin = list[float]()
            ksib = list[float]()
            tria = list[float]()

            for sc in ['binomial','getkHopSiblingScore','triangle']:
                with open(f'approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold/{eb}_weightedGraph_{sc}_{smp}_samples.csv','r') as csvfile:
                    rows = csv.reader(csvfile, delimiter=',')
                    for row in rows:
                        if sc == 'binomial':
                            bin.append(float(row[2]))
                        if sc == 'getkHopSiblingScore':
                            ksib.append(float(row[2]))
                        if sc == 'triangle':
                            tria.append(float(row[2]))

            bin_sum = 0
            ksib_sum = 0
            tria_sum = 0

            for i in range(len(bin)):
                bin_sum += abs(sib[i]-bin[i])**2
                ksib_sum += abs(sib[i]-ksib[i])**2
                tria_sum += abs(sib[i]-tria[i])**2

            if eb == 'RotatE':
                MSE_RotatE_bin.append(bin_sum/len(sib))
                MSE_RotatE_ksib.append(ksib_sum/len(sib))
                MSE_RotatE_tria.append(tria_sum/len(sib))
            if eb == 'TransE':
                MSE_TransE_bin.append(bin_sum/len(sib))
                MSE_TransE_ksib.append(ksib_sum/len(sib))
                MSE_TransE_tria.append(tria_sum/len(sib))
            if eb == 'PairRE':
                MSE_PairRE_bin.append(bin_sum/len(sib))
                MSE_PairRE_ksib.append(ksib_sum/len(sib))
                MSE_PairRE_tria.append(tria_sum/len(sib))
            if eb == 'DistMult':
                MSE_DistMult_bin.append(bin_sum/len(sib))
                MSE_DistMult_ksib.append(ksib_sum/len(sib))
                MSE_DistMult_tria.append(tria_sum/len(sib))

    steps = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0]
    legend_names = ['RotatE', 'TransE', 'PairRE', 'DistMult']
    symbol_map = {'RotatE': '--o', 'TransE': '--^', 'PairRE': '--*', 'DistMult': '-->'}
    for ydata, name in zip([MSE_RotatE_bin, MSE_TransE_bin, MSE_PairRE_bin, MSE_DistMult_bin], legend_names):
        plt.plot(steps, ydata,symbol_map[name],label = name)
    plt.legend(loc = 'best')
    plt.savefig(f'approach/MSE_{sett.DATASETNAME}_binomial_comparison.png', dpi=500, bbox_inches='tight')
    plt.clf()

    for ydata, name in zip([MSE_RotatE_ksib, MSE_TransE_ksib, MSE_PairRE_ksib, MSE_DistMult_ksib], legend_names):
        plt.plot(steps, ydata, symbol_map[name], label = name)
    plt.legend(loc = 'best')
    plt.savefig(f'approach/MSE_{sett.DATASETNAME}_ksib_comparison.png', dpi=500, bbox_inches='tight')
    plt.clf()
    '''
    for ydata, name in zip([MSE_RotatE_tria, MSE_TransE_tria, MSE_PairRE_tria, MSE_DistMult_tria], legend_names):
        plt.plot(steps, ydata,'--o', label = name)
    plt.legend(loc = 'best')
    plt.savefig(f'approach/MSE_triangle_comparison.png', dpi=500, bbox_inches='tight')
    plt.clf()'''


    legend_names = ['binomial', '1-hop-siblings']
    for ydata, name in zip([MSE_RotatE_bin, MSE_RotatE_ksib], legend_names):
        plt.plot(steps, ydata,'--o', label = name)
    plt.legend(loc = 'best')
    plt.savefig(f'approach/MSE_{sett.DATASETNAME}_RotatE_comparison.png', dpi=500, bbox_inches='tight')
    plt.clf()

    for ydata, name in zip([MSE_TransE_bin, MSE_TransE_ksib], legend_names):
        plt.plot(steps, ydata,'--o', label = name)
    plt.legend(loc = 'best')
    plt.savefig(f'approach/MSE_{sett.DATASETNAME}_TransE_comparison.png', dpi=500, bbox_inches='tight')
    plt.clf()

    for ydata, name in zip([MSE_PairRE_bin, MSE_PairRE_ksib], legend_names):
        plt.plot(steps, ydata,'--o', label = name)
    plt.legend(loc = 'best')
    plt.savefig(f'approach/MSE_{sett.DATASETNAME}_PairRE_comparison.png', dpi=500, bbox_inches='tight')
    plt.clf()

    for ydata, name in zip([MSE_DistMult_bin, MSE_DistMult_ksib], legend_names):
        plt.plot(steps, ydata,'--o', label = name)
    plt.legend(loc = 'best')
    plt.savefig(f'approach/MSE_{sett.DATASETNAME}_DistMult_comparison.png', dpi=500, bbox_inches='tight')
    plt.clf()
        

def runDense(size: int) -> None:

    models, all_triples, all_triples_set, LP_triples_pos, LP_triples_neg, emb_train, entity2embedding, relation2embedding, entity_to_id_map, relation_to_id_map = getKFoldEmbeddings()
    
    full_graph = TriplesFactory(all_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
    df = pd.DataFrame(full_graph.triples, columns=['subject', 'predicate', 'object'])

    
    start = timeit.default_timer()
    #subgraphs = dh.loadSubGraphs(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold")
    subgraphs = []

    with open(f"approach/Subgraphs/Exact/Dense_subgraphs_{sett.SIZE_OF_SUBGRAPHS}_{sett.EMBEDDING_TYPE}.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        subgraphs = list[set[str]]()
        for row in rows:
            subgraph = set[str]()
            for ele in row:
                subgraph.add(ele)
            subgraphs.append(subgraph)

    for e in dh.loadSubGraphsEmbSel(f"approach/Subgraphs/Exact", sett.EMBEDDING_TYPE): subgraphs.append(e)

    used_ent = set()

    for subgraph in subgraphs:
        used_ent = used_ent.union(subgraph)
    
    end = timeit.default_timer()
    print(f'Subgraphs are done in time: {end-start}')

    score_tail = []
    score_rel = []
    score_cla = []

    for i in range(sett.N_SPLITS):
        '''
        start = timeit.default_timer()
        LP_test_score_tail, start_time_clf_training, end_time_clf_training, start_time_LP_score, end_time_LP_score = emb.baselineLP_tail(models[i], subgraphs, emb_train[i], LP_triples_pos[i], all_triples_set)
        score_tail.append(LP_test_score_tail)
        end = timeit.default_timer()
        print(f'Tail prediction for fold {i} has taken {end-start}')

        start = timeit.default_timer()
        LP_test_score_rel, start_time_clf_training, end_time_clf_training, start_time_LP_score, end_time_LP_score = emb.baselineLP_relation(models[i], subgraphs, emb_train[i], LP_triples_pos[i], all_triples_set)
        score_rel.append(LP_test_score_rel)
        end = timeit.default_timer()
        print(f'Relation prediction for fold {i} has taken {end-start}')
        '''
        start = timeit.default_timer()
        LP_test_score = makeTCPart(LP_triples_pos[i],  LP_triples_neg[i], entity2embedding, relation2embedding, subgraphs, emb_train[i])
        score_cla.append(LP_test_score)
        end = timeit.default_timer()
        print(f'Triple Classification for fold {i} has taken {end-start}')


    print(f"finished scores of {sett.EMBEDDING_TYPE}")

    model_deg_rank_score = []
    model_deg_rank_score_log = []
    model_deg_score = []
    model_siblings_degree_score = []
    model_siblings_score = []
    model_siblings_log_score = []

    model_siblings_degree_score_ent = []
    model_siblings_score_ent = []
    model_siblings_log_score_ent = []

    model_relation_sum_at_1 = []
    model_relation_sum_at_5 = []
    model_relation_sum_at_10 = []
    model_relation_sum_for_MRR = []

    model_tail_sum_at_1 = []
    model_tail_sum_at_5 = []
    model_tail_sum_at_10 = []
    model_tail_sum_for_MRR = []

    for i in range(sett.N_SPLITS):
        model_deg_rank_score.append([])
        model_deg_rank_score_log.append([])
        model_deg_score.append([])
        model_siblings_degree_score.append([])
        model_siblings_score.append([])
        model_siblings_log_score.append([])

        model_siblings_degree_score_ent.append([])
        model_siblings_score_ent.append([])
        model_siblings_log_score_ent.append([])

        model_relation_sum_at_1.append([])
        model_relation_sum_at_5.append([])
        model_relation_sum_at_10.append([])
        model_relation_sum_for_MRR.append([])

        model_tail_sum_at_1.append([])
        model_tail_sum_at_5.append([])
        model_tail_sum_at_10.append([])
        model_tail_sum_for_MRR.append([])


    first_relation = True
    first_tail = True
    sbg = -1
    print()
    print(f'Iterations per subgraph start now!')
    for subgraph in subgraphs:
        print()
        iteration_start = timeit.default_timer()
        subgraph_list = list(subgraph)
        sbg += 1
        outDegree = dict()
        occurences = defaultdict(int)
        inDegree = dict()

        start = timeit.default_timer()
        for t in df.values:
            
            if (t[0] in subgraph) and (t[2] in subgraph):
                if entity_to_id_map[t[0]] in outDegree.keys():
                    outDegree[entity_to_id_map[t[0]]] = 1 + outDegree[entity_to_id_map[t[0]]]
                else:
                    outDegree[entity_to_id_map[t[0]]] = 1

                if entity_to_id_map[t[2]] in inDegree.keys():
                    inDegree[entity_to_id_map[t[2]]] = 1 + inDegree[entity_to_id_map[t[2]]]
                else:
                    inDegree[entity_to_id_map[t[2]]] = 1

                if entity_to_id_map[t[2]] not in outDegree.keys():
                    outDegree[entity_to_id_map[t[2]]] = 0

                if entity_to_id_map[t[0]] not in inDegree.keys():
                    inDegree[entity_to_id_map[t[0]]] = 1
                
                occurences[relation_to_id_map[t[1]]] = 1 + occurences[relation_to_id_map[t[1]]]
        end = timeit.default_timer()
        print(f'Degree Mapping complete in {end-start}')

        HeadModelsRanking = []
        RelationModelsRanking = []
        TailModelsRanking = []

        HeadModelRank = []
        RelationModelRank = []
        TailModelRank = []
        for i in range(sett.N_SPLITS):
            HeadModelsRanking.append(dict())
            RelationModelsRanking.append(defaultdict(dict))
            TailModelsRanking.append(defaultdict(dict))

            HeadModelRank.append(dict())
            RelationModelRank.append(dict())
            TailModelRank.append(dict())

        start = timeit.default_timer()
        for he in list(subgraph_list):
                head = entity_to_id_map[he]
                for i in range(sett.N_SPLITS):
                    HeadModelsRanking[i][head] = dict()
                for relation in occurences.keys():
                    if first_relation:
                        for i in range(sett.N_SPLITS):
                            RelationModelsRanking[i][relation] = dict()

                    for ta in list(subgraph_list):
                            tail = entity_to_id_map[ta]
                            if first_tail:
                                for i in range(sett.N_SPLITS):
                                    TailModelsRanking[i][tail] = dict()
                            
                            ten = torch.tensor([[head,relation,tail]])
                            for i in range(sett.N_SPLITS):
                                score = models[i].score_hrt(ten)
                                score = score.detach().numpy()[0][0]

                                HeadModelsRanking[i][head][(relation,tail)] = score
                                RelationModelsRanking[i][relation][(head,tail)] = score
                                TailModelsRanking[i][tail][(head,relation)] = score

                    first_tail = False
                first_relation = False
        end = timeit.default_timer()
        print(f'Ranking Mapping complete {end-start}')

        start = timeit.default_timer()
        for he in list(subgraph_list):
                head = entity_to_id_map[he]
                for i in range(sett.N_SPLITS):
                    #print(HeadModelRank[i])
                    #print(HeadModelsRanking[i])
                    HeadModelRank[i][head] = list(dict(sorted(HeadModelsRanking[i][head].items(), key=lambda item: item[1], reverse=True)).keys())

                    TailModelRank[i][head] = list(dict(sorted(TailModelsRanking[i][head].items(), key=lambda item: item[1], reverse=True)).keys())
        
        for relation in occurences.keys():
            for i in range(sett.N_SPLITS):
                RelationModelRank[i][relation] = list(dict(sorted(RelationModelsRanking[i][relation].items(), key=lambda item: item[1], reverse=True)).keys())
        end = timeit.default_timer()
        print(f'Ranking Sorting complete {end-start}')

        #print(f"finished ranking dictionaries of {sett.EMBEDDING_TYPE} and sbg {sbg}")

        counter = 0
        counterEnt = 0
        counterSib = 0
        sums_deg_rank = []
        sums_deg_rank_log = []
        sums_deg = []
        sums_siblings_degree = []
        sums_siblings = []
        sums_siblings_log = []

        for i in range(sett.N_SPLITS):
            sums_deg_rank.append(0)
            sums_deg_rank_log.append(0)
            sums_deg.append(0)
            sums_siblings_degree.append(0)
            sums_siblings.append(0)
            sums_siblings_log.append(0)

        start = timeit.default_timer()
        for he in list(subgraph_list):
            h = entity_to_id_map[he]
            if h in outDegree.keys():
                for ta in list(subgraph_list):
                    t = entity_to_id_map[ta]
                    if t in inDegree.keys():
                        for r in occurences.keys():
                            counter += 1
                            if (h,r,t) in all_triples_set:
                                counterSib += 1
                            for i in range(sett.N_SPLITS):
                                hRank = findingRank(HeadModelRank[i][h],(r,t))
                                rRank = findingRank(RelationModelRank[i][r],(h,t))
                                tRank = findingRank(TailModelRank[i][t],(h,r))

                                hRankNeg = findingRankNegHead(HeadModelRank[i][h],(r,t),all_triples_set,h)
                                tRankNeg = findingRankNegTail(TailModelRank[i][t],(h,r),all_triples_set,t)

                                if (h,r,t) in all_triples_set:
                                    sums_deg_rank[i] += ( (1/(max(hRank - outDegree[h] + 1,1))) + (1/(max(rRank - occurences[r] + 1,1))) + (1/(max(tRank - inDegree[t] + 1,1))) )/3
                                    sums_deg_rank_log[i] += np.log( 1+ 1/(max(hRank - outDegree[h] + 1,1)) + 1/(max(rRank - occurences[r] + 1,1)) + 1/(max(tRank - inDegree[t] + 1,1)) )
                                    sums_siblings_degree[i] += ( (1/(max(hRank - outDegree[h] + 1,1))) + (1/(max(tRank - inDegree[t] + 1,1))) )/2
                                    sums_siblings[i] += ( (1/hRankNeg) + (1/tRankNeg) )/2
                                    sums_siblings_log[i] += np.log( 1+ (1/hRankNeg) + (1/tRankNeg) )
                                else:
                                    sums_deg_rank[i] += ( 1-(1/(max(hRank - outDegree[h] + 1,1))) + 1-(1/(max(rRank - occurences[r] + 1,1))) + 1-(1/(max(tRank - inDegree[t] + 1,1))) )/3
                                    sums_deg_rank_log[i] += np.log( 1+ 1-(1/(max(hRank - outDegree[h] + 1,1)) ) + 1-(1/(max(rRank - occurences[r] + 1,1)) ) + 1-(1/(max(tRank - inDegree[t] + 1,1)) ) )

            
                counterEnt += 1
                for i in range(sett.N_SPLITS):
                    if inDegree[h] == 0 and outDegree[h] == 0:
                        counterEnt -= 1
                    elif inDegree[h] == 0:
                        sums_deg[i] += (overlapTail(all_triples_set, TailModelRank[i][h], h, outDegree[h])/outDegree[h])
                    elif outDegree[h] == 0:
                        sums_deg[i] += (overlapHead(all_triples_set, HeadModelRank[i][h], h, inDegree[h])/inDegree[h])
                    else:
                        sums_deg[i] += ( (overlapHead(all_triples_set, HeadModelRank[i][h], h, inDegree[h])/inDegree[h]) + (overlapTail(all_triples_set, TailModelRank[i][h], h, outDegree[h])/outDegree[h]) )/2
        end = timeit.default_timer()
        print(f'Getting Score Sums is done in {end-start}')

        start = timeit.default_timer()
        for i in range(sett.N_SPLITS):
            if counter == 0:
                model_deg_rank_score[i].append(-100)
                model_deg_rank_score_log[i].append(-100)
            else:
                model_deg_rank_score[i].append(sums_deg_rank[i]/counter)
                model_deg_rank_score_log[i].append(sums_deg_rank_log[i]/counter)
            
            if counterEnt == 0:
                model_deg_score[i].append(-100)
            else:
                model_deg_score[i].append(sums_deg[i]/counterEnt)
            
            
            if counterSib == 0:
                model_siblings_degree_score[i].append(-100)
                model_siblings_score[i].append(-100)
                model_siblings_log_score[i].append(-100)
            else:
                model_siblings_degree_score[i].append(sums_siblings_degree[i]/counterSib)
                model_siblings_score[i].append(sums_siblings[i]/counterSib)
                model_siblings_log_score[i].append(sums_siblings_log[i]/counterSib)

                model_siblings_degree_score_ent[i].append(sums_siblings_degree[i]/len(subgraph))
                model_siblings_score_ent[i].append(sums_siblings[i]/len(subgraph))
                model_siblings_log_score_ent[i].append(sums_siblings_log[i]/len(subgraph))
        end = timeit.default_timer()
        print(f'Dividing Scores is done in {end-start}')
        
        start = timeit.default_timer()
        counter_of_test_tp = []
        for i in range(sett.N_SPLITS):
            relation_sum_at_1 = 0
            relation_sum_at_5 = 0
            relation_sum_at_10 = 0
            relation_sum_for_MRR = 0

            tail_sum_at_1 = 0
            tail_sum_at_5 = 0
            tail_sum_at_10 = 0
            tail_sum_for_MRR = 0
            counter_of_test_tp = 0
            for tp in LP_triples_pos[i]:
                counter = 0
                if (emb_train[i].entity_id_to_label[tp[0]] in subgraph) and (emb_train[0].entity_id_to_label[tp[2]] in subgraph):
                    counter_of_test_tp += 1

                    tmp_scores = dict()
                    for relation in range(emb_train[0].num_relations):
                        tup = (tp[0],relation,tp[2])
                        if tup in all_triples_set and relation != tp[1]:
                            continue
                        ten = torch.tensor([[tp[0],relation,tp[2]]])
                        score = models[i].score_hrt(ten)
                        score = score.detach().numpy()[0][0]
                        tmp_scores[relation] = score
                    sl = sorted(tmp_scores.items(), key=lambda x:x[1], reverse=True)
                    
                    for pair in sl:
                            counter += 1
                            if pair[0] == tp[1]:
                                if counter <= 1:
                                    relation_sum_at_1 += 1
                                    relation_sum_at_5 += 1
                                    relation_sum_at_10 += 1
                                elif counter <= 5:
                                    relation_sum_at_5 += 1
                                    relation_sum_at_10 += 1
                                elif counter <= 10:
                                    relation_sum_at_10 += 1
                                relation_sum_for_MRR += 1/counter
                                break

                    tmp_scores = dict()
                    for tail in range(emb_train[0].num_entities):
                        tup = (tp[0],tp[1],tail)
                        if tup in all_triples_set and tail != tp[2]:
                            continue
                        ten = torch.tensor([[tp[0],tp[1],tail]])
                        score = models[i].score_hrt(ten)
                        score = score.detach().numpy()[0][0]
                        tmp_scores[tail] = score

                    sl = sorted(tmp_scores.items(), key=lambda x:x[1], reverse=True)
                        
                    for pair in sl:
                            counter += 1
                            if pair[0] == tp[2]:
                                if counter <= 1:
                                    tail_sum_at_1 += 1
                                    tail_sum_at_5 += 1
                                    tail_sum_at_10 += 1
                                elif counter <= 5:
                                    tail_sum_at_5 += 1
                                    tail_sum_at_10 += 1
                                elif counter <= 10:
                                    tail_sum_at_10 += 1
                                tail_sum_for_MRR += 1/counter
                                break
            if counter_of_test_tp > 0:
                model_tail_sum_at_1[i].append(tail_sum_at_1/counter_of_test_tp)
                model_tail_sum_at_5[i].append(tail_sum_at_5/counter_of_test_tp)
                model_tail_sum_at_10[i].append(tail_sum_at_10/counter_of_test_tp)
                model_tail_sum_for_MRR[i].append(tail_sum_for_MRR/counter_of_test_tp)

                model_relation_sum_at_1[i].append(relation_sum_at_1/counter_of_test_tp)
                model_relation_sum_at_5[i].append(relation_sum_at_5/counter_of_test_tp)
                model_relation_sum_at_10[i].append(relation_sum_at_10/counter_of_test_tp)
                model_relation_sum_for_MRR[i].append(relation_sum_for_MRR/counter_of_test_tp)
            else:
                model_relation_sum_at_1[i].append(-100)
                model_relation_sum_at_5[i].append(-100)
                model_relation_sum_at_10[i].append(-100)
                model_relation_sum_for_MRR[i].append(-100)

                model_tail_sum_at_1[i].append(-100)
                model_tail_sum_at_5[i].append(-100)
                model_tail_sum_at_10[i].append(-100)
                model_tail_sum_for_MRR[i].append(-100)
        end = timeit.default_timer()
        print(f'Hit@ and MRR are done {end-start}')

        iteration_end = timeit.default_timer()
        print(f'Iteration for subgraph {sbg} took {iteration_end-iteration_start}')
        print()



    path = f'approach/scoreData/KFold/{sett.DATASETNAME}_{sett.EMBEDDING_TYPE}'
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)

    for i in range(sett.N_SPLITS):
            c = open(f'{path}/kfold_reliability_of_{i}th_fold_sbg_Exact_{sett.SIZE_OF_SUBGRAPHS}.csv', "w")
            writer = csv.writer(c)
            data = ['subgraph', 'triple classification score', 'degree + rank', 'degree + rank + log', 'degree', 'siblings degree', 'siblings', 'siblings log','Tail Hit @ 1','Tail Hit @ 5','Tail Hit @ 10','Tail MRR','Relation Hit @ 1','Relation Hit @ 5','Relation Hit @ 10','Relation MRR', 'siblings degree Ent', 'siblings Ent', 'siblings log Ent']
            writer.writerow(data)
            for j in range(len(score_cla[i])):
                data = [j, score_cla[i][j], model_deg_rank_score[i][j], model_deg_rank_score_log[i][j], model_deg_score[i][j], model_siblings_degree_score[i][j], model_siblings_score[i][j], model_siblings_log_score[i][j]]
                tmp = [model_tail_sum_at_1[i][j],model_tail_sum_at_5[i][j],model_tail_sum_at_10[i][j],model_tail_sum_for_MRR[i][j],model_relation_sum_at_1[i][j],model_relation_sum_at_5[i][j],model_relation_sum_at_10[i][j],model_relation_sum_for_MRR[i][j], model_siblings_degree_score_ent[i][j], model_siblings_score_ent[i][j], model_siblings_log_score_ent[i][j]]
                data.extend(tmp)
                writer.writerow(data)
            c.close()

def runDenseGlobal(size: int, score_calculation: Callable[[str, str, nx.MultiDiGraph, list[object], object, object, set[tuple[int,int,int]], TriplesFactory], float]) -> None:    
    all_triples, all_triples_set, entity_to_id_map, relation_to_id_map, test_triples, validation_triples = emb.getDataFromPykeen(datasetname=sett.DATASETNAME)
    full_dataset = torch.cat((all_triples, test_triples.mapped_triples, validation_triples.mapped_triples))
    full_graph = TriplesFactory(full_dataset,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
    df = pd.DataFrame(full_graph.triples, columns=['subject', 'predicate', 'object'])
    M = nx.MultiDiGraph()

    subgraphs = list[set[str]]()
    
    with open(f"approach/Subgraphs/Exact/Dense_subgraphs_{sett.SIZE_OF_SUBGRAPHS}_{sett.EMBEDDING_TYPE}.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        for row in rows:
            subgraph = set[str]()
            for ele in row:
                subgraph.add(ele)
            subgraphs.append(subgraph)

    for e in dh.loadSubGraphsEmbSel(f"approach/Subgraphs/Exact", sett.EMBEDDING_TYPE): subgraphs.append(e)

    models, all_triples, all_triples_set, LP_triples_pos, LP_triples_neg, emb_train, entity2embedding, relation2embedding, entity_to_id_map, relation_to_id_map = getKFoldEmbeddings()
    for t in df.values:
        M.add_edge(t[0], t[2], label = t[1])

    model_siblings_score = []
    for subgraph in subgraphs:
        start = timeit.default_timer()
        count = 0
        sib_sum = 0
        for u,v in nx.DiGraph(M).subgraph(subgraph).edges():
            print(f'{u} and {v}')
            w = score_calculation(u, v, M, models, entity_to_id_map, relation_to_id_map, all_triples_set, full_graph)
            count += 1
            sib_sum += w

        sib_sum = sib_sum/count
        model_siblings_score.append(sib_sum)


    full_graph = TriplesFactory(all_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
    
    start = timeit.default_timer()
    #subgraphs = dh.loadSubGraphs(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold")
    

    used_ent = set()

    for subgraph in subgraphs:
        used_ent = used_ent.union(subgraph)
    
    end = timeit.default_timer()
    print(f'Subgraphs are done in time: {end-start}')

    score_tail = []
    score_rel = []
    score_cla = []

    for i in range(sett.N_SPLITS):
        start = timeit.default_timer()
        LP_test_score = makeTCPart(LP_triples_pos[i],  LP_triples_neg[i], entity2embedding, relation2embedding, subgraphs, emb_train[i])
        score_cla.append(LP_test_score)
        end = timeit.default_timer()
        print(f'Triple Classification for fold {i} has taken {end-start}')


    print(f"finished scores of {sett.EMBEDDING_TYPE}")

    model_relation_sum_at_1 = []
    model_relation_sum_at_5 = []
    model_relation_sum_at_10 = []
    model_relation_sum_for_MRR = []

    model_tail_sum_at_1 = []
    model_tail_sum_at_5 = []
    model_tail_sum_at_10 = []
    model_tail_sum_for_MRR = []

    for i in range(sett.N_SPLITS):

        model_relation_sum_at_1.append([])
        model_relation_sum_at_5.append([])
        model_relation_sum_at_10.append([])
        model_relation_sum_for_MRR.append([])

        model_tail_sum_at_1.append([])
        model_tail_sum_at_5.append([])
        model_tail_sum_at_10.append([])
        model_tail_sum_for_MRR.append([])


    first_relation = True
    first_tail = True
    sbg = -1
    print()
    print(f'Iterations per subgraph start now!')
    for subgraph in subgraphs:
        
        start = timeit.default_timer()
        counter_of_test_tp = []
        for i in range(sett.N_SPLITS):
            relation_sum_at_1 = 0
            relation_sum_at_5 = 0
            relation_sum_at_10 = 0
            relation_sum_for_MRR = 0

            tail_sum_at_1 = 0
            tail_sum_at_5 = 0
            tail_sum_at_10 = 0
            tail_sum_for_MRR = 0
            counter_of_test_tp = 0
            for tp in LP_triples_pos[i]:
                counter = 0
                if (emb_train[i].entity_id_to_label[tp[0]] in subgraph) and (emb_train[0].entity_id_to_label[tp[2]] in subgraph):
                    counter_of_test_tp += 1

                    tmp_scores = dict()
                    for relation in range(emb_train[0].num_relations):
                        tup = (tp[0],relation,tp[2])
                        if tup in all_triples_set and relation != tp[1]:
                            continue
                        ten = torch.tensor([[tp[0],relation,tp[2]]])
                        score = models[i].score_hrt(ten)
                        score = score.detach().numpy()[0][0]
                        tmp_scores[relation] = score
                    sl = sorted(tmp_scores.items(), key=lambda x:x[1], reverse=True)
                    
                    for pair in sl:
                            counter += 1
                            if pair[0] == tp[1]:
                                if counter <= 1:
                                    relation_sum_at_1 += 1
                                    relation_sum_at_5 += 1
                                    relation_sum_at_10 += 1
                                elif counter <= 5:
                                    relation_sum_at_5 += 1
                                    relation_sum_at_10 += 1
                                elif counter <= 10:
                                    relation_sum_at_10 += 1
                                relation_sum_for_MRR += 1/counter
                                break

                    tmp_scores = dict()
                    for tail in range(emb_train[0].num_entities):
                        tup = (tp[0],tp[1],tail)
                        if tup in all_triples_set and tail != tp[2]:
                            continue
                        ten = torch.tensor([[tp[0],tp[1],tail]])
                        score = models[i].score_hrt(ten)
                        score = score.detach().numpy()[0][0]
                        tmp_scores[tail] = score

                    sl = sorted(tmp_scores.items(), key=lambda x:x[1], reverse=True)
                        
                    for pair in sl:
                            counter += 1
                            if pair[0] == tp[2]:
                                if counter <= 1:
                                    tail_sum_at_1 += 1
                                    tail_sum_at_5 += 1
                                    tail_sum_at_10 += 1
                                elif counter <= 5:
                                    tail_sum_at_5 += 1
                                    tail_sum_at_10 += 1
                                elif counter <= 10:
                                    tail_sum_at_10 += 1
                                tail_sum_for_MRR += 1/counter
                                break
            if counter_of_test_tp > 0:
                model_tail_sum_at_1[i].append(tail_sum_at_1/counter_of_test_tp)
                model_tail_sum_at_5[i].append(tail_sum_at_5/counter_of_test_tp)
                model_tail_sum_at_10[i].append(tail_sum_at_10/counter_of_test_tp)
                model_tail_sum_for_MRR[i].append(tail_sum_for_MRR/counter_of_test_tp)

                model_relation_sum_at_1[i].append(relation_sum_at_1/counter_of_test_tp)
                model_relation_sum_at_5[i].append(relation_sum_at_5/counter_of_test_tp)
                model_relation_sum_at_10[i].append(relation_sum_at_10/counter_of_test_tp)
                model_relation_sum_for_MRR[i].append(relation_sum_for_MRR/counter_of_test_tp)
            else:
                model_relation_sum_at_1[i].append(-100)
                model_relation_sum_at_5[i].append(-100)
                model_relation_sum_at_10[i].append(-100)
                model_relation_sum_for_MRR[i].append(-100)

                model_tail_sum_at_1[i].append(-100)
                model_tail_sum_at_5[i].append(-100)
                model_tail_sum_at_10[i].append(-100)
                model_tail_sum_for_MRR[i].append(-100)
        end = timeit.default_timer()
        print(f'Hit@ and MRR are done {end-start}')
        print()



    path = f'approach/scoreData/KFold/{sett.DATASETNAME}_{sett.EMBEDDING_TYPE}'
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)

    for i in range(sett.N_SPLITS):
            c = open(f'{path}/kfold_reliability_of_{i}th_fold_sbg_ExactGlobal_{sett.SIZE_OF_SUBGRAPHS}.csv', "w")
            writer = csv.writer(c)
            data = ['subgraph', 'triple classification score', 'siblings','Tail Hit @ 1','Tail Hit @ 5','Tail Hit @ 10','Tail MRR','Relation Hit @ 1','Relation Hit @ 5','Relation Hit @ 10','Relation MRR']
            writer.writerow(data)
            for j in range(len(score_cla[i])):
                data = [j, score_cla[i][j], model_siblings_score[j]]
                tmp = [model_tail_sum_at_1[i][j],model_tail_sum_at_5[i][j],model_tail_sum_at_10[i][j],model_tail_sum_for_MRR[i][j],model_relation_sum_at_1[i][j],model_relation_sum_at_5[i][j],model_relation_sum_at_10[i][j],model_relation_sum_for_MRR[i][j]]
                data.extend(tmp)
                writer.writerow(data)
            c.close()


if __name__ == "__main__":
    run_eval()


#KFoldNegGen()
#run_eval()

