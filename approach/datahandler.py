import random
import networkx as nx
from numpy import typename
import pandas as pd

from pykeen.triples import TriplesFactory

def createNegTriple(kg_triple, triples):
    '''
    Creating negative triples
    By taking an existing triple ans swapping head and tail 
    so we get a non existing triple as neg triple
    '''
    kg_neg_triple_list = []
    lst_emb = list(range(triples.num_entities))
    for pos_sample in kg_triple:
        not_created = True
        relation = pos_sample[1]
        while not_created:
            head = random.choice(lst_emb)
            tail = random.choice(lst_emb)
            kg_neg_triple = [head,relation,tail]
            # TODO ckeck containments in sets
            # done
            if (kg_neg_triple not in kg_triple):
                not_created = False
        kg_neg_triple_list.append(kg_neg_triple)

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
        while len(visited) < size_of_graphs:
            if random.random() < restart:
                node = original_node
            else:
                neighbors = set(G.neighbors(node)) - visited
                if len(neighbors) == 0:
                    neighbors = set(G.nodes()) - visited
                node = random.sample(neighbors,1)[0]
            visited.add(node)
        subgraphs.append(visited)
    return subgraphs

# TODO implement functions
def storeSubGraphs(subgraphs):
    return
def readSubGraphs():
    return