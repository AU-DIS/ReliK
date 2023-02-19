import torch

from pykeen.triples import TriplesFactory

import embedding as emb
import settings as sett
import datahandler as dh

def getKFoldEmbeddings():
    models = []
    LP_triples_pos = []
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
    
    if sett.EMBEDDING_TYPE == 'TransE':
        entity2embedding, relation2embedding = emb.createEmbeddingMaps_TransE(models[0], emb_train_triples)
    elif sett.EMBEDDING_TYPE == 'DistMult':
        entity2embedding, relation2embedding = emb.createEmbeddingMaps_DistMult(models[0], emb_train_triples)
    else:
        entity2embedding, relation2embedding = emb.createEmbeddingMaps_DistMult(models[0], emb_train_triples)
    
    return models, all_triples, all_triples_set, LP_triples_pos, emb_train, entity2embedding, relation2embedding, entity_to_id_map, relation_to_id_map

def run_eval():
    return
