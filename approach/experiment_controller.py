import argparse
import os
import torch
import pandas as pd
import networkx as nx
import csv

import embedding as emb
import datahandler as dh
import classifier as cla

from pykeen.models import TransE
from pykeen.pipeline import pipeline

from pykeen.triples import TriplesFactory
from pykeen.triples import CoreTriplesFactory
from pykeen.contrib.lightning import LCWALitModule
from pykeen.datasets import Dataset
import pytorch_lightning
import gc

import os

def makeTCPart(LP_triples_pos, LP_triples_neg, entity2embedding, relation2embedding, subgraphs, emb_train_triples):
    X_train, X_test, y_train, y_test = cla.prepareTrainTestData(LP_triples_pos, LP_triples_neg, emb_train_triples)
    clf = cla.trainClassifier(X_train, y_train, entity2embedding, relation2embedding)
    LP_test_score = cla.testClassifierSubgraphs(clf, X_test, y_test, entity2embedding, relation2embedding, subgraphs)

    return LP_test_score

def getSiblingScore(u: str, v: str, M: nx.MultiDiGraph, models: list[object], entity_to_id_map: object, relation_to_id_map: object, all_triples_set: set[tuple[int,int,int]], alltriples: TriplesFactory, n_split: int) -> float:

    head = entity_to_id_map[u]
    tail = entity_to_id_map[v]

    HeadModelRank: list[dict[tuple[int,int],float]] = []
    TailModelRank: list[dict[tuple[int,int],float]] = []

    for _ in range(n_split):
        HeadModelRank.append(dict())
        TailModelRank.append(dict())

    #for entstr in list(subgraphs[0]):
    for ent in range(alltriples.num_entities):
        for rel in range(alltriples.num_relations):
            #ent = entity_to_id_map[entstr]
            ten_h = torch.tensor([[head,rel,ent]])
            ten_t = torch.tensor([[ent,rel,tail]])

            for i in range(n_split):
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
        
        for i in range(n_split):
            part1 = list(dict(sorted(HeadModelRank[i].items(), key=lambda item: item[1], reverse=True)).keys())
                
            part2 = list(dict(sorted(TailModelRank[i].items(), key=lambda item: item[1], reverse=True)).keys())

            hRankNeg += dh.findingRankNegHead(part1,(relation,tail),all_triples_set,head) / n_split
            tRankNeg += dh.findingRankNegTail(part2,(head,relation),all_triples_set,tail) / n_split

    hRankNeg = hRankNeg/len(between_labels)
    tRankNeg = tRankNeg/len(between_labels)
    return ( (1/hRankNeg) + (1/tRankNeg) ) /2

def grabAllKFold(datasetname: str, n_split: int):
    all_triples, all_triples_set, entity_to_id_map, relation_to_id_map, test_triples, validation_triples = emb.getDataFromPykeen(datasetname=datasetname)
    full_dataset = torch.cat((all_triples, test_triples.mapped_triples, validation_triples.mapped_triples))

    isExist = os.path.exists(f"approach/KFold/{datasetname}_{n_split}_fold")
    full_dataset = torch.cat((all_triples, test_triples.mapped_triples, validation_triples.mapped_triples))
    if not isExist:
        dh.generateKFoldSplit(full_dataset, datasetname, random_seed=None, n_split=nmb_KFold)
    full_dataset = torch.cat((all_triples, test_triples.mapped_triples, validation_triples.mapped_triples))
    full_graph = TriplesFactory(full_dataset,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)

    emb_train_triples = []
    emb_test_triples = []
    LP_triples_pos = []
    for i in range(n_split):
        emb_triples_id, LP_triples_id = dh.loadKFoldSplit(i, datasetname,n_split=nmb_KFold)
        emb_triples = full_dataset[emb_triples_id]
        LP_triples = full_dataset[LP_triples_id]
        emb_train_triples.append(TriplesFactory(emb_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map))
        emb_test_triples.append(TriplesFactory(LP_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map))
        LP_triples_pos.append(LP_triples.tolist())

    return all_triples, all_triples_set, entity_to_id_map, relation_to_id_map, emb_train_triples, emb_test_triples, LP_triples_pos, full_graph

def getOrTrainModels(embedding: str, dataset_name: str, n_split: int, emb_train_triples, emb_test_triples):
    models = []
    for i in range(n_split):
        isFile = os.path.isfile(f"approach/trainedEmbeddings/{dataset_name}_{embedding}_{n_split}_fold/{dataset_name}_{i}th/trained_model.pkl")
        if not isFile:
            save = f"{dataset_name}_{embedding}_{n_split}_fold/{dataset_name}_{i}th"
            emb_model, emb_triples_used = emb.trainEmbedding(emb_train_triples[i], emb_test_triples[i], random_seed=42, saveModel=True, savename = save, embedd = embedding, dimension = 50, epoch_nmb = 50)
            models.append(emb_model)
        else:
            save = f"{dataset_name}_{embedding}_{n_split}_fold/{dataset_name}_{i}th"
            models.append(emb.loadModel(save))

    return models

def KFoldNegGen(datasetname: str, n_split: int, all_triples_set, LP_triples_pos, emb_train):
    isFile = os.path.isfile(f"approach/KFold/{datasetname}_{n_split}_fold/0th_neg.csv")
    LP_triples_neg = []
    if not isFile:
        for i in range(n_split):
            neg_triples, throw = dh.createNegTripleHT(all_triples_set, LP_triples_pos[i], emb_train[i])
            dh.storeTriples(f"approach/KFold/{datasetname}_{n_split}_fold/{i}th_neg", neg_triples)
            LP_triples_neg.append(neg_triples)
    else:
        for i in range(n_split):
            neg_triples = dh.loadTriples(f"approach/KFold/{datasetname}_{n_split}_fold/{i}th_neg")
            LP_triples_neg.append(neg_triples)
    return LP_triples_neg

def DoGlobalSiblingScore(embedding, datasetname, n_split, size_subgraph, models, entity_to_id_map, relation_to_id_map, all_triples_set, full_graph):
    df = pd.DataFrame(full_graph.triples, columns=['subject', 'predicate', 'object'])
    M = nx.MultiDiGraph()

    subgraphs = list[set[str]]()
    
    with open(f"approach/KFold/{datasetname}_{n_split}_fold/subgraphs_{size_subgraph}.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        for row in rows:
            subgraph = set[str]()
            for ele in row:
                subgraph.add(ele)
            subgraphs.append(subgraph)

    #for e in dh.loadSubGraphsEmbSel(f"approach/Subgraphs/Exact", embedding): subgraphs.append(e)

    for t in df.values:
        M.add_edge(t[0], t[2], label = t[1])

    model_siblings_score = []
    tracker = 0
    for subgraph in subgraphs:
        count = 0
        sib_sum = 0
        for u,v in nx.DiGraph(M).subgraph(subgraph).edges():
            #print(f'{u} and {v}')
            w = getSiblingScore(u, v, M, models, entity_to_id_map, relation_to_id_map, all_triples_set, full_graph, n_split)
            count += 1
            sib_sum += w

        sib_sum = sib_sum/count
        model_siblings_score.append(sib_sum)
        tracker += 1
        if tracker % 100 == 0: print(f'have done {tracker} of {len(subgraphs)} in {embedding}')

    path = f"approach/scoreData/{datasetname}_{n_split}/{embedding}/siblings_score_subgraphs_{size_subgraph}.csv"
    c = open(f'{path}', "w")
    writer = csv.writer(c)
    data = ['subgraph','siblings']
    writer.writerow(data)
    for j in range(len(model_siblings_score)):
        data = [j, model_siblings_score[j]]
        writer.writerow(data)
    c.close()

def classifierExp(embedding, datasetname, size_subgraph, LP_triples_pos,  LP_triples_neg, entity2embedding, relation2embedding, emb_train, n_split):
    subgraphs = list[set[str]]()
    
    with open(f"approach/KFold/{datasetname}_{n_split}_fold/subgraphs_{size_subgraph}.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        for row in rows:
            subgraph = set[str]()
            for ele in row:
                subgraph.add(ele)
            subgraphs.append(subgraph)
    
    score_cla = []
    for i in range(n_split):
        LP_test_score = makeTCPart(LP_triples_pos[i],  LP_triples_neg[i], entity2embedding, relation2embedding, subgraphs, emb_train[i])
        score_cla.append(LP_test_score)

    fin_score_cla = []
    for i in range(len(score_cla[0])):
        sumcla = 0
        measured = 0
        for j in range(n_split):
            if score_cla[j][i] >= 0:
                sumcla += score_cla[j][i]
                measured += 1
        if measured == 0:
            fin_score_cla.append(-100)
        else:
            fin_score_cla.append(sumcla/n_split)

    path = f"approach/scoreData/{datasetname}_{n_split}/{embedding}/classifier_score_subgraphs_{size_subgraph}.csv"
    c = open(f'{path}', "w")
    writer = csv.writer(c)
    data = ['subgraph','classifier']
    writer.writerow(data)
    for j in range(len(fin_score_cla)):
        data = [j, fin_score_cla[j]]
        writer.writerow(data)
    c.close()

def prediction(embedding, datasetname, size_subgraph, emb_train, all_triples_set, n_split):
    model_relation_sum_at_1 = []
    model_relation_sum_at_5 = []
    model_relation_sum_at_10 = []
    model_relation_sum_for_MRR = []

    model_tail_sum_at_1 = []
    model_tail_sum_at_5 = []
    model_tail_sum_at_10 = []
    model_tail_sum_for_MRR = []

    for i in range(n_split):
        model_relation_sum_at_1.append([])
        model_relation_sum_at_5.append([])
        model_relation_sum_at_10.append([])
        model_relation_sum_for_MRR.append([])

        model_tail_sum_at_1.append([])
        model_tail_sum_at_5.append([])
        model_tail_sum_at_10.append([])
        model_tail_sum_for_MRR.append([])

    subgraphs = list[set[str]]()
    
    with open(f"approach/KFold/{datasetname}_{n_split}_fold/subgraphs_{size_subgraph}.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        for row in rows:
            subgraph = set[str]()
            for ele in row:
                subgraph.add(ele)
            subgraphs.append(subgraph)

    for subgraph in subgraphs:
        for i in range(n_split):
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

                model_relation_sum_at_1[i].append(-100)
                model_tail_sum_at_5[i].append(-100)
                model_tail_sum_at_10[i].append(-100)
                model_tail_sum_for_MRR[i].append(-100)

    fin_score_tail_at1 = []
    fin_score_tail_at5 = []
    fin_score_tail_at10 = []
    fin_score_tail_atMRR = []

    fin_score_relation_at1 = []
    fin_score_relation_at5 = []
    fin_score_relation_at10 = []
    fin_score_relation_atMRR = []
    for i in range(len(model_relation_sum_at_1[0])):
        tailat1 = 0
        tailat5 = 0
        tailat10 = 0
        tailatMRR = 0

        relationat1 = 0
        relationat5 = 0
        relationat10 = 0
        relationatMRR = 0
        measured = 0
        for j in range(n_split):
            if model_relation_sum_at_1[j][i] >= 0:
                tailat1 += model_relation_sum_at_1[j][i]
                measured += 1
            if model_relation_sum_at_5[j][i] >= 0:
                tailat5 += model_relation_sum_at_5[j][i]
            if model_relation_sum_at_10[j][i] >= 0:
                tailat10 += model_relation_sum_at_10[j][i]
            if model_relation_sum_for_MRR[j][i] >= 0:
                tailatMRR += model_relation_sum_for_MRR[j][i]

            if model_relation_sum_at_1[j][i] >= 0:
                relationat1 += model_relation_sum_at_1[j][i]
            if model_relation_sum_at_5[j][i] >= 0:
                relationat5 += model_relation_sum_at_5[j][i]
            if model_relation_sum_at_10[j][i] >= 0:
                relationat10 += model_relation_sum_at_10[j][i]
            if model_relation_sum_for_MRR[j][i] >= 0:
                relationatMRR += model_relation_sum_for_MRR[j][i]
        if measured == 0:
            fin_score_tail_at1.append(-100)
            fin_score_tail_at5.append(-100)
            fin_score_tail_at10.append(-100)
            fin_score_tail_atMRR.append(-100)

            fin_score_relation_at1.append(-100)
            fin_score_relation_at5.append(-100)
            fin_score_relation_at10.append(-100)
            fin_score_relation_atMRR.append(-100)
        else:
            fin_score_tail_at1.append(tailat1/measured)
            fin_score_tail_at5.append(tailat5/measured)
            fin_score_tail_at10.append(tailat10/measured)
            fin_score_tail_atMRR.append(tailatMRR/measured)

            fin_score_relation_at1.append(relationat1/measured)
            fin_score_relation_at5.append(relationat5/measured)
            fin_score_relation_at10.append(relationat10/measured)
            fin_score_relation_atMRR.append(relationatMRR/measured)

    path = f"approach/scoreData/{datasetname}_{n_split}/{embedding}/prediction_score_subgraphs_{size_subgraph}.csv"
    c = open(f'{path}', "w")
    writer = csv.writer(c)
    data = ['subgraph','Tail Hit @ 1','Tail Hit @ 5','Tail Hit @ 10','Tail MRR','Relation Hit @ 1','Relation Hit @ 5','Relation Hit @ 10','Relation MRR']
    writer.writerow(data)
    for j in range(len(fin_score_tail_at1)):
        data = [j, fin_score_tail_at1[j], fin_score_tail_at5[j], fin_score_tail_at10[j], fin_score_tail_atMRR[j], fin_score_relation_at1[j], fin_score_relation_at5[j], fin_score_relation_at10[j], fin_score_relation_atMRR[j]]
        writer.writerow(data)
    c.close()

def yago2():
    #os.environ["CUDA_VISIBLE_DEVICES"]="1"
    gc.collect()

    torch.cuda.empty_cache()
    data=pd.read_csv('approach/yago2core_facts.clean.notypes_3.tsv',sep='\t',names=['subject', 'predicate', 'object'])

    entity_to_id_map = {v: k for v, k in enumerate(pd.factorize(pd.concat([data['subject'],data['object']]))[1])}
    entity_to_id_map2 = {k: v for v, k in enumerate(pd.factorize(pd.concat([data['subject'],data['object']]))[1])}
    relation_to_id_map = {v: k for v, k in enumerate(pd.factorize(data['predicate'])[1])}
    relation_to_id_map2 = {k: v for v, k in enumerate(pd.factorize(data['predicate'])[1])}
    #print(len(entity_to_id_map))
    #print(data)
    data['subject'] = data['subject'].map(entity_to_id_map2)
    data['object'] = data['object'].map(entity_to_id_map2)  
    data['predicate'] = data['predicate'].map(relation_to_id_map2)  
    #data.replace({'subject': entity_to_id_map})
    #print(data)
    ten = torch.tensor(data.values)

    full_yago2 = CoreTriplesFactory(ten,num_entities=len(entity_to_id_map),num_relations=len(relation_to_id_map))
    h = Dataset().from_tf(full_yago2, [0.8-0.0000001,0.2,0.0000001])
    '''dh.generateKFoldSplit(ten, 'Yago2', random_seed=None, n_split=nmb_KFold)

    
    alldata = CoreTriplesFactory(ten,num_entities=len(entity_to_id_map),num_relations=len(relation_to_id_map))

    emb_triples_id, LP_triples_id = dh.loadKFoldSplit(0, 'Yago2',n_split=nmb_KFold)
    emb_triples = ten[emb_triples_id]
    LP_triples = ten[LP_triples_id]
    
    emb_train_triples = CoreTriplesFactory(emb_triples,num_entities=len(entity_to_id_map),num_relations=len(relation_to_id_map))
    emb_test_triples = CoreTriplesFactory(LP_triples,num_entities=len(entity_to_id_map),num_relations=len(relation_to_id_map))'''
    del ten
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    model = LCWALitModule(
        dataset=h,
        model='TransE',
        model_kwargs=dict(embedding_dim=50),
        batch_size=32
    )
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    trainer = pytorch_lightning.Trainer(
        accelerator="gpu",  # automatically choose accelerator
        logger=False,  # defaults to TensorBoard; explicitly disabled here
        precision=16,  # mixed precision training
        min_epochs= 50,
        max_epochs=50,
        devices=-1
    )
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    trainer.fit(model=model)

    model

    '''result = pipeline(training=emb_train_triples,testing=emb_test_triples,model=TransE,random_seed=4,training_loop='LCWA', model_kwargs=dict(embedding_dim=50),training_kwargs=dict(num_epochs=50, batch_size=32), evaluation_fallback= True)   

    model = result.model'''

    from pykeen.evaluation import RankBasedEvaluator
    evaluator = RankBasedEvaluator()

    # Get triples to test
    mapped_triples = dataset.testing.mapped_triples

    # Evaluate
    results = evaluator.evaluate(
        model=model,
        mapped_triples=mapped_triples,
        batch_size=1024,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )

    results.save_to_directory(f"approach/trainedEmbeddings/yago2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--embedding', dest='embedding', type=str, help='choose which embedding type to use')
    parser.add_argument('-d','--datasetname', dest='dataset_name', type=str, help='choose which dataset to use')
    parser.add_argument('-t','--tasks', dest='tasks', type=str, help='if set, only run respective tasks, split with \",\" could be from [siblings, prediction, triple, densest]')
    parser.add_argument('-s','--subgraphs', dest='size_subgraphs', type=int, help='choose which size of subgraphs are getting tested')
    parser.add_argument('-n','--n_subgraphs', dest='n_subgraphs', type=int, help='choose which n of subgraphs are getting tested')
    parser.add_argument('-st','--setup', dest='setup',action='store_true' ,help='if set, just setup and train embedding, subgraphs, neg-triples')
    parser.add_argument('-heur','--heuristic', dest='heuristic', type=str, help='which heuristic should be used in the case of dense subgraph task')
    args = parser.parse_args()

    nmb_KFold: int = 5

    if args.embedding == 'Yago':
        yago2()
        quit()

    # Default cases in which return message to user, that some arguments are needed
    if not args.heuristic and not args.setup and not args.size_subgraphs and not args.tasks and not args.dataset_name and not args.embedding:
        print('Please, provide at least an embedding and a dataset to perform an experiment')
        quit(code=1)

    if not args.dataset_name and not args.embedding:
        print('Please, provide at least an embedding and a dataset to perform an experiment')    
        quit(code=1)

    if not args.dataset_name:
        print('Please, provide at least a dataset to perform an experiment')    
        quit(code=1)

    if not args.embedding:
        print('Please, provide at least an embedding to perform an experiment')    
        quit(code=1)

    # If no list provided do everything
    if args.tasks:
        task_list: set[str] = set(args.tasks.split(','))
    elif args.setup:
        task_list: set[str] = set()
    else:
        task_list: set[str] = set(('siblings', 'prediction', 'triple', 'densest'))

    if args.size_subgraphs:
        size_subgraphs = args.size_subgraphs
    else:
        size_subgraphs = 50

    if args.n_subgraphs:
        n_subgraphs = args.n_subgraphs
    else:
        n_subgraphs = 100

    # collecting all information except the model from the KFold
    all_triples, all_triples_set, entity_to_id_map, relation_to_id_map, emb_train_triples, emb_test_triples, LP_triples_pos, full_graph = grabAllKFold(args.dataset_name, nmb_KFold)

    # checking if we have negative triples for
    LP_triples_neg = KFoldNegGen(args.dataset_name, nmb_KFold, all_triples_set, LP_triples_pos, emb_train_triples)

    # getting or training the models
    models = getOrTrainModels(args.embedding, args.dataset_name, nmb_KFold, emb_train_triples, emb_test_triples)

    if not os.path.isfile(f"approach/KFold/{args.dataset_name}_{nmb_KFold}_fold/subgraphs_{size_subgraphs}.csv"):
        subgraphs = dh.createSubGraphs(all_triples, entity_to_id_map, relation_to_id_map, number_of_graphs=n_subgraphs, size_of_graphs=size_subgraphs)
        dh.storeSubGraphs(f"approach/KFold/{args.dataset_name}_{nmb_KFold}_fold", subgraphs)

    if 'siblings' in task_list:
        DoGlobalSiblingScore(args.embedding, args.dataset_name, nmb_KFold, size_subgraphs, models, entity_to_id_map, relation_to_id_map, all_triples_set, full_graph)
    if 'prediction' in task_list:
        prediction(args.embedding, args.dataset_name, args.size_subgraph, emb_train_triples, all_triples_set, nmb_KFold)
    if 'triple' in task_list:
        classifierExp(args.embedding, args.dataset_name, args.size_subgraph, LP_triples_pos,  LP_triples_neg, entity_to_id_map, relation_to_id_map, emb_train_triples, nmb_KFold)
    if 'densest' in task_list:
        print('HELLO not done yet')
        

