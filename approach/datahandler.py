import random
import networkx as nx
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
            if (kg_neg_triple not in kg_triple):
                not_created = False
        kg_neg_triple_list.append(kg_neg_triple)

    return kg_neg_triple_list

def createSubGraphs(all_triples, entity_to_id, relation_to_id, h=10, k=20):
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
    while len(subgraphs) < h:
        visited = []
        node = random.choice(list(G.nodes()))
        original_node = node
        visited.append(node)
        while len(visited) < k:
            if random.random() < 0.9:
                node = original_node
            else:
                neighbors = list(set(G.neighbors(node)) - set(visited))
                if len(neighbors) == 0:
                    neighbors = list(set(G.nodes()) - set(visited))
                node = random.choice(neighbors)
            visited.append(node)
        subgraphs.append(visited)
    return subgraphs