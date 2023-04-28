import random
from more_itertools import substrings
import networkx as nx
from numpy import typename
import pandas as pd
import ast
import numpy as np
import csv
import torch
import os
import dsdm as dsd
import settings as sett
import embedding as emb
from pykeen.triples import TriplesFactory
from sklearn.model_selection import KFold
import classifier as cla
from collections import defaultdict
import timeit
import itertools

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
        emb_triples_id, LP_triples_id = loadKFoldSplit(i, n_split=sett.N_SPLITS)
        emb_triples = full_dataset[emb_triples_id]
        LP_triples = full_dataset[LP_triples_id]
        emb_train_triples = TriplesFactory(emb_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
        emb_train.append(emb_train_triples)
        emb_test_triples = TriplesFactory(LP_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
        emb_test.append(emb_test_triples)
        LP_triples_pos.append(LP_triples.tolist())

        neg_triples = loadTriples(f"approach/KFold/{sett.DATASETNAME}_{sett.N_SPLITS}_fold/{i}th_neg")
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


def generateKFoldSplit(full_dataset, datasetname, random_seed=None, n_split=5):
    kf = KFold(n_splits=n_split, random_state=random_seed, shuffle=True)
    fold_train_test_pairs = []
    isExist = os.path.exists(f"approach/KFold/{datasetname}_{n_split}_fold")

    if not isExist:
        os.makedirs(f"approach/KFold/{datasetname}_{n_split}_fold")

    for i, (train_index, test_index) in enumerate(kf.split(full_dataset)):
        c = open(f"approach/KFold/{datasetname}_{n_split}_fold/{i}_th_fold.csv", "w")
        writer = csv.writer(c)
        writer.writerows([train_index, test_index])
        c.close()
        fold_train_test_pairs.append([train_index.tolist(),test_index.tolist()])
    return fold_train_test_pairs


def loadKFoldSplit(ith_fold, datasetname, n_split=5):
    with open(f"approach/KFold/{datasetname}_{n_split}_fold/{ith_fold}_th_fold.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        i = 0
        train = []
        test = []
        for row in rows:
            if i==0:
                for ele in row:
                    train.append(int(ele))
            else:
                for ele in row:
                    test.append(int(ele))
            i += 1
    return train, test
    

def createNegTripleHT(kg_triple_set, kg_triple, triples):
    '''
    Creating negative triples
    By taking an existing triple ans swapping head and tail 
    so we get a non existing triple as neg triple
    '''
    kg_neg_triple_list = []
    related_nodes = set()
    lst_emb = list(range(triples.num_entities))
    bigcount = 0
    for pos_sample in kg_triple:
        related_nodes.add((pos_sample[0],pos_sample[2]))
        not_created = True
        relation = pos_sample[1]
        count = 0
        did_break = False
        while not_created:
            if count > (0.1*len(lst_emb)):
                did_break = True
                break
            head = random.choice(lst_emb)
            tail = random.choice(lst_emb)
            kg_neg_triple = [head,relation,tail]
            kg_neg_triple_tuple = (head,relation,tail)
            if (kg_neg_triple_tuple not in kg_triple_set):
                not_created = False
        if did_break:
            continue
        kg_neg_triple_list.append(kg_neg_triple)
        bigcount += 1
        if bigcount % 10000 == 0:
            print(f'Have created {bigcount} neg samples')

    return kg_neg_triple_list, related_nodes

def createNegTripleRelation(kg_triple_set, kg_triple, triples):
    '''
    Creating negative triples
    By taking an existing triple ans swapping head and tail 
    so we get a non existing triple as neg triple
    '''
    kg_neg_triple_list = []
    lst_emb = list(range(triples.num_relations))
    bigcount = 0
    for pos_sample in kg_triple:
        not_created = True
        head = pos_sample[0]
        tail = pos_sample[2]
        count = 0
        did_break = False
        while not_created:
            if count > (10 * len(lst_emb)):
                did_break = True
                break
            relation = random.choice(lst_emb)
            kg_neg_triple = [head,relation,tail]
            kg_neg_triple_tuple = (head,relation,tail)
            if (kg_neg_triple_tuple not in kg_triple_set):
                not_created = False
            count += 1
        if did_break:
            continue
        kg_neg_triple_list.append(kg_neg_triple)
        bigcount += 1
        if bigcount % 10000 == 0:
            print(f'Have created {bigcount} neg samples')

    return kg_neg_triple_list

def createSubGraphs(all_triples, entity_to_id, relation_to_id, number_of_graphs=10, size_of_graphs=20, restart=0.2):
    '''
    Creates subgraphs from the given KG by specific random walks with restart
    Returns all subgraphs in a list, each as a list of included nodes
    '''
    full_graph = TriplesFactory(all_triples,entity_to_id=entity_to_id,relation_to_id=relation_to_id)
    df = pd.DataFrame(full_graph.triples, columns=['subject', 'predicate', 'object'])
    G = nx.MultiDiGraph()

    for t in df.values:
        G.add_edge(t[0], t[2], label = t[1])
    subgraphs = []
    while len(subgraphs) < number_of_graphs:
        visited = set()
        node = random.choice(list(G.nodes()))
        original_node = node
        visited.add(node)
        all_neighbours = set()
        while len(visited) < size_of_graphs:
            if random.random() < restart:
                node = original_node
            else:
                neighbors = set(G.neighbors(node)) - visited
                all_neighbours = set.union(neighbors, all_neighbours) - visited
                if len(all_neighbours) == 0:
                    node = random.choice(list(G.nodes()))
                elif len(neighbors) == 0:
                    node = random.choice(list(all_neighbours))
                else:
                    node = random.choice(list(neighbors))
            visited.add(node)
        subgraphs.append(visited)
    return subgraphs

def storeSubGraphs(path, subgraphs):
    with open(f"{path}/subgraphs_{len(subgraphs[0])}.csv", "a+") as f:
        wr = csv.writer(f)
        wr.writerows(subgraphs)

def storeDenSubGraphs(path, subgraphs):
    with open(f"{path}subgraphs_{len(subgraphs[0])}_{sett.EMBEDDING_TYPE}.csv", "a+") as f:
        wr = csv.writer(f)
        wr.writerows(subgraphs)

def loadSubGraphs(path):
    with open(f"{path}/subgraphs_{sett.SIZE_OF_SUBGRAPHS}.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        subgraphs = []
        for row in rows:
            subgraph = set()
            for ele in row:
                subgraph.add(ele)
            subgraphs.append(subgraph)
    return subgraphs

def loadSubGraphsEmb(path):
    with open(f"{path}/subgraphs_{sett.SIZE_OF_SUBGRAPHS}_{sett.EMBEDDING_TYPE}.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        subgraphs = []
        for row in rows:
            subgraph = set()
            for ele in row:
                subgraph.add(ele)
            subgraphs.append(subgraph)
    return subgraphs

def loadSubGraphsEmbSel(path: str, source: str) -> list[set[str]]:
    with open(f"{path}/subgraphs_{sett.SIZE_OF_SUBGRAPHS}_{source}.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        subgraphs = list[set[str]]()
        for row in rows:
            subgraph = set[str]()
            for ele in row:
                subgraph.add(ele)
            subgraphs.append(subgraph)
    return subgraphs

def prepareDensestSubgraphs():
    path = f"approach"
    all_triples, all_triples_set, entity_to_id_map, relation_to_id_map, test_triples, validation_triples = emb.getDataFromPykeen(datasetname=sett.DATASETNAME)
    full_dataset = torch.cat((all_triples, test_triples.mapped_triples, validation_triples.mapped_triples))
    full_graph = TriplesFactory(full_dataset,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
    df = pd.DataFrame(full_graph.triples, columns=['subject', 'predicate', 'object'])
    M = nx.MultiDiGraph()

    for t in df.values:
        M.add_edge(t[0], t[2], label = t[1])

    G = nx.Graph()
    for u,v,data in M.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)

    flowless_R = dsd.flowless(G, 5, weight='weight')
    greedy_R = dsd.greedy_charikar(G, weight='weight')

    print(len(flowless_R[0]))
    flow = createSubGraphs(full_dataset, entity_to_id_map, relation_to_id_map, number_of_graphs=5, size_of_graphs=len(flowless_R[0]))
    flow.append(set(flowless_R[0]))
    print(len(greedy_R[0]))
    greedy = createSubGraphs(full_dataset, entity_to_id_map, relation_to_id_map, number_of_graphs=5, size_of_graphs=len(greedy_R[0]))
    greedy.append(set(greedy_R[0]))

    storeSubGraphs(path, flow)
    storeSubGraphs(path, greedy)

def densestEmbSbg():
    all_triples, all_triples_set, entity_to_id_map, relation_to_id_map, test_triples, validation_triples = emb.getDataFromPykeen(datasetname=sett.DATASETNAME)
    full_dataset = torch.cat((all_triples, test_triples.mapped_triples, validation_triples.mapped_triples))
    full_graph = TriplesFactory(full_dataset,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
    df = pd.DataFrame(full_graph.triples, columns=['subject', 'predicate', 'object'])
    M = nx.MultiDiGraph()

    models, all_triples, all_triples_set, LP_triples_pos, LP_triples_neg, emb_train, entity2embedding, relation2embedding, entity_to_id_map, relation_to_id_map = getKFoldEmbeddings()
    
    for t in df.values:
        M.add_edge(t[0], t[2], label = t[1])

    G = nx.Graph()
    count = 0
    pct = 0
    start = timeit.default_timer()
    length = len(nx.DiGraph(M).edges())
    print(f'Starting with {length}')
    for u,v in nx.DiGraph(M).edges():
        print(f'{u} and {v}')
        w = getkHopSiblingScore(u, v, M, models, entity_to_id_map, relation_to_id_map, all_triples_set)
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
            print('ERROR')
        else:
            G.add_edge(u, v, weight=w)
            print(w)
        count += 1
        if count % (length // 100) == 0:
            pct += 1
            now = timeit.default_timer()
            print(f'Finished with {pct}% for {sett.EMBEDDING_TYPE} in time {now-start}, took avg of {(now-start)/pct} per point')

    flowless_R = dsd.flowless(G, 5, weight='weight')
    greedy_R = dsd.greedy_charikar(G, weight='weight')

    print(len(flowless_R[0]))
    flow = list()#createSubGraphs(full_dataset, entity_to_id_map, relation_to_id_map, number_of_graphs=5, size_of_graphs=len(flowless_R[0]))
    flow.append(set(flowless_R[0]))
    print(len(greedy_R[0]))

    greedy = list()#createSubGraphs(full_dataset, entity_to_id_map, relation_to_id_map, number_of_graphs=5, size_of_graphs=len(greedy_R[0]))
    greedy.append(set(greedy_R[0]))
    path = f"approach"
    storeDenSubGraphs(path, flow)
    storeDenSubGraphs(path, greedy)

    l = dict(G.degree(weight='weight'))
    k = min(l, key=l.get)
    print(f'{k} with {l[k]}')
    print(f'density is {greedy_R[1]} or {flowless_R[1]}')
    print(greedy_R[0],greedy_R[1])
    print(flowless_R[0],flowless_R[1])
    print(f'total weigths are: {sum(l.values())}')
    print(f'total density is: {(sum(l.values())/2)/(len(G.nodes)**2)}')
    sub = G.subgraph(greedy_R[0])
    l = dict(sub.degree(weight='weight'))
    print(f'total weigths of greedy dense are: {sum(l.values())}')

    sub = G.subgraph(flowless_R[0])
    l = dict(sub.degree(weight='weight'))
    print(f'total weigths of flowless dense are: {sum(l.values())}')



    

def getkHopSiblingScore(u, v, M, models, entity_to_id_map, relation_to_id_map, all_triples_set, any):

    subgraph_list, labels, existing, count, ex_triples  = getkHopneighbors(u,v,M)

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
        while (len(selectedComparators) < 0.1 * (possible-count) or len(selectedComparators) < 100):# and len(selectedComparators) < 1000:
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
    

    '''
    first_tail = True
    for he in list(subgraph_list):
        head = entity_to_id_map[he]
        for i in range(sett.N_SPLITS):
            HeadModelsRanking[i][head] = dict()
        for rel in labels:
            relation = relation_to_id_map[rel]
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
                    TailModelsRanking[i][tail][(head,relation)] = score

                    first_tail = False
    '''
    head = entity_to_id_map[u]
    tail = entity_to_id_map[v]
    
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

    return ( (1/(hRankNeg +( (possible-count+1)-len(selectedComparators) ) )) + (1/(tRankNeg + ( (possible-count+1)-len(selectedComparators)) )) )/2

def getkHopneighbors(u,v,M):

    labels = set()
    between_labels = set()
    existing_triples = set()

    for el in M.get_edge_data(u,v).items():
        labels.add(el[1]['label'])
        between_labels.add(el[1]['label'])

    entities = set()

    entities.add(u)
    entities.add(v)

    for n in M.neighbors(u):
        entities.add(n)

    for n in M.neighbors(v):
        entities.add(n)

    entities = list(entities)
    count = 0
    for e in entities:
        #for f in entities:
            if M.get_edge_data(u,e) == None:
                continue
            for el in M.get_edge_data(u,e).items():
                labels.add(el[1]['label'])
                existing_triples.add( (u,el[1]['label'],e) )
                count += 1
    for e in entities:
        #for f in entities:
            if M.get_edge_data(e,v) == None:
                continue
            for el in M.get_edge_data(e,v).items():
                labels.add(el[1]['label'])
                existing_triples.add( (e,el[1]['label'],v) )
                count += 1
    count = len(existing_triples)
    return entities, list(labels), list(between_labels), count, existing_triples

def getTriangle(u,v,M):

    labels = set()
    between_labels = set()
    existing_triples = set()

    for el in M.get_edge_data(u,v).items():
        labels.add(el[1]['label'])
        between_labels.add(el[1]['label'])

    entities = set()
    u_entities = set()
    v_entities = set()

    entities.add(u)
    entities.add(v)

    for n in M.neighbors(u):
        u_entities.add(n)
    for n in list(u_entities):
        for m in M.neighbors(n):
            if m in u_entities:
                entities.add(m)
                break
    
    for n in M.neighbors(v):
        v_entities.add(n)
    for n in list(v_entities):
        for m in M.neighbors(n):
            if m in v_entities:
                entities.add(m)
                break

    entities = list(entities)
    count = 0
    for e in entities:
        #for f in entities:
            if M.get_edge_data(u,e) == None:
                continue
            for el in M.get_edge_data(u,e).items():
                labels.add(el[1]['label'])
                existing_triples.add( (u,el[1]['label'],e) )
                count += 1
    for e in entities:
        #for f in entities:
            if M.get_edge_data(e,v) == None:
                continue
            for el in M.get_edge_data(e,v).items():
                labels.add(el[1]['label'])
                existing_triples.add( (e,el[1]['label'],v) )
                count += 1
    count = len(existing_triples)
    return entities, list(labels), list(between_labels), count, existing_triples

def storeTriples(path, triples):
    with open(f"{path}.csv", "a+") as f:
        wr = csv.writer(f)
        wr.writerows(triples)

def storeRelated(path, related):
    related = list(related)
    with open(f"{path}.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(related)

def loadTriples(path):
    with open(f"{path}.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        triples = []
        for row in rows:
            tp = [int(row[0]),int(row[1]),int(row[2])]
            triples.append(tp)
    return triples

def loadRelated(path):
    with open(f"{path}.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        related_nodes = set()
        for row in rows:
            tp = (int(row[0]),int(row[1]))
            related_nodes.add(tp)
    return related_nodes


def convertListToData(sample_triples, triples, pos_sample=True):
    ds = []
    if pos_sample:
        for t in sample_triples:
            ds.append([triples.entity_id_to_label[t[0]], triples.relation_id_to_label[t[1]], triples.entity_id_to_label[t[2]], 1])
    else:
        for t in sample_triples:
            ds.append([triples.entity_id_to_label[t[0]], triples.relation_id_to_label[t[1]], triples.entity_id_to_label[t[2]], 0])

    dataset = np.array(ds)

    X = dataset[:, :-1]
    y = dataset[:, -1]

    return X, y

def convertListToData_Relation(sample_triples, triples, pos_sample=True):
    ds = dict()
    for i in range(triples.num_relations):
        ds[i] = []
    if pos_sample:
        for t in sample_triples:
            ds[t[1]].append([triples.entity_id_to_label[t[0]], triples.relation_id_to_label[t[1]], triples.entity_id_to_label[t[2]], 1])
    else:
        for t in sample_triples:
            ds[t[1]].append([triples.entity_id_to_label[t[0]], triples.relation_id_to_label[t[1]], triples.entity_id_to_label[t[2]], 0])

    X_dict = dict()
    y_dict = dict()
    for i in range(triples.num_relations):
        dataset = np.array(ds[i])
        X = dataset[:, :-1]
        y = dataset[:, -1]
        X_dict[i] = X
        y_dict[i] = y

    return X_dict, y_dict
