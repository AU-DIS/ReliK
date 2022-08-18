import embedding as emb
import datahandler as dh
import classifier as cla
import settings as set

import math
import torch

from sklearn.model_selection import train_test_split
from pykeen.triples import TriplesFactory
from numpy import linalg as LA
import numpy as np

def sigmoid(x, max_score, min_score):
    x_norm = 12 * (x - min_score) / (max_score - min_score) - 6

    return 1 / (1 + math.exp(-x_norm))

def checkTransEScore(score, h,r,t, emb_train_triples, entity2embedding, relation2embedding):
    head_vec = np.array(entity2embedding[emb_train_triples.entity_id_to_label[h]])
    relation_vec = np.array(relation2embedding[emb_train_triples.relation_id_to_label[r]])
    tail_vec = np.array(entity2embedding[emb_train_triples.entity_id_to_label[t]])
    norm = -LA.norm(head_vec + relation_vec - tail_vec, ord=1)
    decision = math.isclose(norm, score, abs_tol=10**-set.SIGFIGS)
    assert decision, f"score is not close (sigfigs {set.SIGFIGS}) to value "

def reliability(all_triples, emb_train_triples, model, entity2embedding, relation2embedding, subgraphs, checkScore=False):
    '''
    Getting the reliability sum as currently defined
    TODO max min finding should be improved
    '''
    
    reliability_score = []
    # Finding max and min values necessary for sigmoid
    # TODO make that part more optimized and easily readable
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
                            score = sigmoid(score, max_score, min_score)
                            if ((h,r,t) in all_triples):
                                sum += 1-score
                            else:
                                sum += score
        reliability_score.append(sum)
    return reliability_score

if __name__ == "__main__":
    # Getting Data from dataset, as tensor of triples and mappings from entity to ids
    all_triples, all_triples_set, entity_to_id_map, relation_to_id_map = emb.getDataFromPykeen(datasetname=set.DATASETNAME)
    
    # Split Data between embedding and LP classifier part
    emb_triples, LP_triples = train_test_split(all_triples, test_size=set.LP_EMB_SPLIT)

    # Split Data between embedding train and test
    # TODO CHANGE: just train on whole graph
    # Done
    emb_train_triples = TriplesFactory(emb_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
    emb_test_triples = TriplesFactory(emb_triples[:1,:],entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)

    # Sanity Check if triples have been changed in the Factory
    assert torch.equal(emb_triples,emb_train_triples.mapped_triples), "Triples changed in creation of Triples Factory"
    assert entity_to_id_map == emb_train_triples.entity_to_id, "Entity mapping changed in creation of Triples Factory"
    assert relation_to_id_map == emb_train_triples.relation_to_id, "Relation mapping changed in creation of Triples Factory"

    # Train and create embedding model
    # TODO currently not stored after creation, should be done
    emb_model, emb_triples_used = emb.trainEmbedding(emb_train_triples, emb_test_triples, random_seed=42)
    entity2embedding, relation2embedding = emb.createEmbeddingMaps(emb_model, emb_train_triples)

    # Sanity Check if triples have been changed while doing the embedding
    assert torch.equal(emb_triples,emb_triples_used.mapped_triples), "Triples changed after embedding"
    assert entity_to_id_map == emb_triples_used.entity_to_id, "Entity mapping changed after embedding"
    assert relation_to_id_map == emb_triples_used.relation_to_id, "Relation mapping changed after embedding"

    # Get negative triples
    LP_triples_pos = LP_triples.tolist()
    LP_triples_neg = dh.createNegTriple(LP_triples_pos, emb_train_triples)

    # Split LP data for training and testing
    # train classifier and test it
    # TODO only test classifier on the respective subgraphs
    # Done
    X_train, X_test, y_train, y_test = cla.prepareTrainTestData(LP_triples_pos, LP_triples_neg, emb_train_triples)
    clf = cla.trainLPClassifier(X_train, y_train, entity2embedding, relation2embedding)

    subgraphs = dh.createSubGraphs(all_triples, entity_to_id_map, relation_to_id_map, size_of_graphs=set.SIZE_OF_SUBGRAPHS, restart=set.RESET_PROB)
    dh.storeSubGraphs(subgraphs)

    LP_test_score = cla.testClassifierSubgraphs(clf, X_test, y_test, entity2embedding, relation2embedding, subgraphs)
    #LP_test_score = clf.score(X_test, y_test)
    print(f'finished LP Score')
    # Get reliability value for KG
    # TODO make reliability value only for subgraph
    # Done
    reliability_score = reliability(all_triples_set, emb_train_triples, emb_model, entity2embedding, relation2embedding, subgraphs, checkScore=True)

    print(f'LP_score for given KG: {LP_test_score}')
    print(f'Reliabilty value for given KG: {reliability_score}')

    # TODO check for correlation, csv of data, measure runtime
    #
    #
    #
    # TODO Sanity Check, of the score functions
    # norm of (head + relation - tail)
    # should be same value as score_hrt in TransE
    # done
    #
    #
    # TODO
    # Sanity Check, that mapping of id's from nodes/relations are consistent trough the project/program
    # Done
    


