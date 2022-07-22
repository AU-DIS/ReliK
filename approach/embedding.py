import torch

import pykeen.datasets as dat

from pykeen.models import TransE
from pykeen.pipeline import pipeline

def getDataFromPykeen(datasetname='Nations'):
    '''
    Using datasets from the pykeen library, and preparing data for our implementaton
    '''
    if datasetname == 'Nations':
        dataset = dat.Nations()
    elif datasetname == 'Countries':
        dataset = dat.Countries()
    elif datasetname == 'Kinships':
        dataset = dat.Kinships()

    entity_to_id_map = dataset.entity_to_id
    relation_to_id_map = dataset.relation_to_id
    all_triples = torch.cat((dataset.training.mapped_triples,dataset.validation.mapped_triples,dataset.testing.mapped_triples))

    return all_triples, entity_to_id_map, relation_to_id_map

def trainEmbedding(training_set, test_set, random_seed=None, saveModel = False):
    '''
    Train embedding for given triples
    '''
    if random_seed == None:
        result = pipeline(training=training_set,testing=test_set,model=TransE,training_loop='LCWA')
    else:
        result = pipeline(training=training_set,testing=test_set,model=TransE,random_seed=random_seed,training_loop='LCWA')

    return result.model

def createEmbeddingMaps(model, triples):
    '''
    create maps of the embedding to the respective entities and relations, for easier reuse
    '''
    e_emb = model.entity_embeddings.cpu()
    entity_ids = torch.LongTensor(range(triples.num_entities))
    e_emb_numpy = e_emb(entity_ids).detach().numpy()
    entity2embedding = {}
    for eid in range(triples.num_entities):
        e = triples.entity_id_to_label[eid]
        entity2embedding[e] = list(e_emb_numpy[eid])

    r_emb = model.relation_embeddings.cpu()
    relation_ids = torch.LongTensor(range(triples.num_relations))
    r_emb_numpy = r_emb(relation_ids).detach().numpy()
    relation2embedding = {}
    for rid in range(triples.num_relations):
        r = triples.relation_id_to_label[rid]
        relation2embedding[r] = list(r_emb_numpy[rid])

    return entity2embedding, relation2embedding
