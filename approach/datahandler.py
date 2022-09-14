import random
from more_itertools import substrings
import networkx as nx
from numpy import typename
import pandas as pd
import ast
import numpy as np
import csv
import torch

import settings as sett

from pykeen.triples import TriplesFactory

def createNegTripleHT(kg_triple_set, kg_triple, triples):
    '''
    Creating negative triples
    By taking an existing triple ans swapping head and tail 
    so we get a non existing triple as neg triple
    '''
    kg_neg_triple_list = []
    lst_emb = list(range(triples.num_entities))
    bigcount = 0
    for pos_sample in kg_triple:
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

    return kg_neg_triple_list

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
    with open(f"{path}/subgraphs_{sett.SIZE_OF_SUBGRAPHS}.csv", "a+") as f:
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

def storeTriples(path, triples):
    with open(f"{path}.csv", "a+") as f:
        wr = csv.writer(f)
        wr.writerows(triples)

def loadTriples(path):
    with open(f"{path}.csv", "r") as f:
        rows = csv.reader(f, delimiter=',')
        triples = []
        for row in rows:
            tp = [int(row[0]),int(row[1]),int(row[2])]
            triples.append(tp)
    return triples


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
