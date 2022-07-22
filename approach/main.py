import embedding as emb
import datahandler as dh
import classifier as cla
import settings as set

import math
import torch

from sklearn.model_selection import train_test_split
from pykeen.triples import TriplesFactory

def sigmoid(x, max_score, min_score):
    x_norm = 12 * (x - min_score) / (max_score - min_score) - 6

    return 1 / (1 + math.exp(-x_norm))


def reliability(all_triples, emb_train_triples, model):
    '''
    Getting the reliability sum as currently defined
    TODO max min finding should be improved
    '''
    all_triples = all_triples.tolist()
    max_score = 0
    first = True
    min_score = 0

    # Finding max and min values necessary for sigmoid
    # TODO make that part more optimized and easily readable
    for h in range(emb_train_triples.num_entities):
        for t in range(emb_train_triples.num_entities):
            for r in range(emb_train_triples.num_relations):
                ten = torch.tensor([[h,r,t]])
                f_r = model.score_hrt(ten)
                f_r = f_r.detach().numpy()[0][0]
                if first:
                    first = False
                    min_score = f_r
                    max_score = f_r
                if min_score > f_r:
                    min_score = f_r
                if max_score < f_r:
                    max_score = f_r

    # Getting the reliability sum as currently defined
    # But with already using sigmoid function to make values closer to each other
    sum = 0
    for h in range(emb_train_triples.num_entities):
        for t in range(emb_train_triples.num_entities):
            for r in range(emb_train_triples.num_relations):
                ten = torch.tensor([[h,r,t]])
                f_r = model.score_hrt(ten)
                f_r = f_r.detach().numpy()[0][0]
                f_r = sigmoid(f_r, max_score, min_score)
                if ([h,r,t] in all_triples):
                    sum += 1-f_r
                else:
                    sum += f_r
    return sum
    


if __name__ == "__main__":
    # Getting Data from dataset, as tensor of triples and mappings from entity to ids
    all_triples, entity_to_id_map, relation_to_id_map = emb.getDataFromPykeen(datasetname=set.DATASETNAME)
    
    # Split Data between embedding and LP classifier part
    emb_triples, LP_triples = train_test_split(all_triples, test_size=set.LP_EMB_SPLIT)

    # Split Data between embedding train and test
    emb_train_set, emb_test_set = train_test_split(emb_triples, test_size=set.TRAIN_TEST_SPLIT)
    emb_train_triples = TriplesFactory(emb_train_set,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
    emb_test_triples = TriplesFactory(emb_test_set,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)

    # Train and create embedding model
    # TODO currently not stored after creation, should be done
    emb_model = emb.trainEmbedding(emb_train_triples, emb_test_triples, random_seed=42)
    entity2embedding, relation2embedding = emb.createEmbeddingMaps(emb_model, emb_train_triples)
    
    # Get negative triples
    LP_triples_pos = LP_triples.tolist()
    LP_triples_neg = dh.createNegTriple(LP_triples_pos, emb_train_triples)

    # Split LP data for training and testing
    # train classifier and test it
    # TODO only test classifier on the respective subgraphs
    X_train, X_test, y_train, y_test = cla.prepareTrainTestData(LP_triples_pos, LP_triples_neg, entity2embedding, relation2embedding, emb_train_triples)
    clf = cla.trainLPClassifier(X_train, y_train)
    LP_test_score = clf.score(X_test, y_test)

    # Get reliability value for KG
    # TODO make reliability value only for subgraph
    reliability_score = reliability(all_triples, emb_train_triples, emb_model)

    print(f'LP_score for given KG: {LP_test_score}')
    print(f'Reliabilty value for given KG: {reliability_score}')




