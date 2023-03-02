import torch
import numpy as np
import csv
import pandas as pd
import os

from collections import defaultdict
from pykeen.triples import TriplesFactory

import embedding as emb
import settings as sett
import datahandler as dh
import classifier as cla

def getKFoldEmbeddings():
    models = []
    LP_triples_pos = []
    LP_triples_neg = []
    emb_train = []
    emb_test = []
    
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

def run_eval():
    models, all_triples, all_triples_set, LP_triples_pos, LP_triples_neg, emb_train, entity2embedding, relation2embedding, entity_to_id_map, relation_to_id_map = getKFoldEmbeddings()
    
    full_graph = TriplesFactory(all_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
    df = pd.DataFrame(full_graph.triples, columns=['subject', 'predicate', 'object'])

    

    subgraphs = dh.loadSubGraphs(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold")

    used_ent = set()

    for subgraph in subgraphs:
        used_ent = used_ent.union(subgraph)

    score_tail = []
    score_rel = []
    score_cla = []

    for i in range(sett.N_SPLITS):
        LP_test_score_tail, start_time_clf_training, end_time_clf_training, start_time_LP_score, end_time_LP_score = emb.baselineLP_tail(models[i], subgraphs, emb_train[i], LP_triples_pos[i], all_triples_set)
        score_tail.append(LP_test_score_tail)

        LP_test_score_rel, start_time_clf_training, end_time_clf_training, start_time_LP_score, end_time_LP_score = emb.baselineLP_relation(models[i], subgraphs, emb_train[i], LP_triples_pos[i], all_triples_set)
        score_rel.append(LP_test_score_rel)

        LP_test_score = makeTCPart(LP_triples_pos[i],  LP_triples_neg[i], entity2embedding, relation2embedding, subgraphs, emb_train[i])
        score_cla.append(LP_test_score)


    print(f"finished scores of {sett.EMBEDDING_TYPE}")

    model_deg_rank_score = []
    model_deg_rank_score_log = []
    model_deg_score = []
    model_siblings_degree_score = []
    model_siblings_score = []
    model_siblings_log_score = []

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
    for subgraph in subgraphs:
        sbg += 1
        outDegree = dict()
        occurences = defaultdict(int)
        inDegree = dict()

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

        for head in range(emb_train[0].num_entities):
            if emb_train[0].entity_id_to_label[head] in subgraph:
                for i in range(sett.N_SPLITS):
                    HeadModelsRanking[i][head] = dict()
                
                for relation in range(emb_train[0].num_relations):
                    if first_relation:
                        for i in range(sett.N_SPLITS):
                            RelationModelsRanking[i][relation] = dict()

                    for tail in range(emb_train[0].num_entities):
                        if emb_train[0].entity_id_to_label[tail] in subgraph:
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

        for head in range(emb_train[0].num_entities):
            if emb_train[0].entity_id_to_label[head] in subgraph:
                for i in range(sett.N_SPLITS):
                    #print(HeadModelRank[i])
                    #print(HeadModelsRanking[i])
                    HeadModelRank[i][head] = list(dict(sorted(HeadModelsRanking[i][head].items(), key=lambda item: item[1], reverse=True)).keys())

                    TailModelRank[i][head] = list(dict(sorted(TailModelsRanking[i][head].items(), key=lambda item: item[1], reverse=True)).keys())
        
        for relation in range(emb_train[0].num_relations):
            for i in range(sett.N_SPLITS):
                RelationModelRank[i][relation] = list(dict(sorted(RelationModelsRanking[i][relation].items(), key=lambda item: item[1], reverse=True)).keys())

        print(f"finished ranking dictionaries of {sett.EMBEDDING_TYPE} and sbg {sbg}")

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

        for h in range(emb_train[0].num_entities):
            if emb_train[0].entity_id_to_label[h] in subgraph and h in outDegree.keys():
                for t in range(emb_train[0].num_entities):
                    if emb_train[0].entity_id_to_label[t] in subgraph and t in inDegree.keys():
                        for r in range(emb_train[0].num_relations):
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
            


    path = f'approach/scoreData/KFold/{sett.DATASETNAME}_{sett.EMBEDDING_TYPE}'
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)
    for i in range(sett.N_SPLITS):
        c = open(f'{path}/kfold_relative_reliability_of_{i}th_fold.csv', "w")
        writer = csv.writer(c)
        data = ['subgraph', 'tail score', 'relation score', 'triple classification score', 'degree + rank', 'degree + rank + log', 'degree', 'siblings degree', 'siblings', 'siblings log','Tail Hit @ 1','Tail Hit @ 5','Tail Hit @ 10','Tail MRR','Relation Hit @ 1','Relation Hit @ 5','Relation Hit @ 10','Relation MRR']
        writer.writerow(data)
        for j in range(len(score_tail[i])):
            data = [j, score_tail[i][j], score_rel[i][j], score_cla[i][j], model_deg_rank_score[i][j], model_deg_rank_score_log[i][j], model_deg_score[i][j], model_siblings_degree_score[i][j], model_siblings_score[i][j], model_siblings_log_score[i][j]]
            tmp = [model_tail_sum_at_1[i][j],model_tail_sum_at_5[i][j],model_tail_sum_at_10[i][j],model_tail_sum_for_MRR[i][j],model_relation_sum_at_1[i][j],model_relation_sum_at_5[i][j],model_relation_sum_at_10[i][j],model_relation_sum_for_MRR[i][j]]
            data.extend(tmp)
            writer.writerow(data)
        c.close()
    
#KFoldNegGen()
run_eval()





                             



                            

    

