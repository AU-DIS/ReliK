import torch

import pykeen.datasets as dat

from pykeen.models import TransE
from pykeen.models import DistMult
from pykeen.models import RotatE
from pykeen.models import PairRE
from pykeen.models import SimplE
from pykeen.models import ConvE
from pykeen.models import ComplEx
from pykeen.models import CompGCN
from pykeen.pipeline import pipeline
import timeit
from typing import cast

def getDataFromPykeen(datasetname: str='Nations'):
    '''
    Using datasets from the pykeen library, and preparing data for our implementaton
    '''
    if datasetname == 'Nations':
        dataset = dat.Nations()
    elif datasetname == 'Countries':
        print('HELLO')
        dataset = dat.Countries()
    elif datasetname == 'Kinships':
        dataset = dat.Kinships()
    elif datasetname == 'UML':
        dataset = dat.UMLS()
    elif datasetname == 'YAGO3-10':
        dataset = dat.YAGO310()
    elif datasetname == 'Hetionet':
         dataset = dat.Hetionet()
    elif datasetname == 'FB15k':
        dataset = dat.FB15k()
    elif datasetname == 'DBpedia50':
        dataset = dat.DBpedia50()
    elif datasetname == 'CodexSmall':
        dataset = dat.CoDExSmall()
    elif datasetname == 'CodexMedium':
        dataset = dat.CoDExMedium()
    elif datasetname == 'CodexLarge':
        dataset = dat.CoDExLarge()
    elif datasetname == 'FB15k237':
        dataset = dat.FB15k237()

    entity_to_id_map = dataset.entity_to_id
    relation_to_id_map = dataset.relation_to_id
    #all_triples_tensor = torch.cat((dataset.training.mapped_triples,dataset.validation.mapped_triples,dataset.testing.mapped_triples))
    all_triples_tensor = dataset.training.mapped_triples
    all_triples_set = set[tuple[int,int,int]]()
    for tup in all_triples_tensor.tolist():
        all_triples_set.add((tup[0],tup[1],tup[2]))
    for tup in dataset.validation.mapped_triples.tolist():
        all_triples_set.add((tup[0],tup[1],tup[2]))
    for tup in dataset.testing.mapped_triples.tolist():
        all_triples_set.add((tup[0],tup[1],tup[2]))
    validation_triples = dataset.validation
    test_triples = dataset.testing

    return all_triples_tensor, all_triples_set, entity_to_id_map, relation_to_id_map, test_triples, validation_triples

def trainEmbeddingMore(training_set, test_set, validation_set, random_seed=None, saveModel = False, savename="Test", embedd="TransE", dimension = 50, epoch_nmb = 100):
    '''
    Train embedding for given triples
    '''
    if embedd == 'TransE':
        mod = TransE
    elif embedd == 'DistMult':
        mod = DistMult
    elif embedd == 'RotatE':
        mod = RotatE
    elif embedd == 'PairRE':
        mod = PairRE
    elif embedd == 'SimplE':
        mod = SimplE
    elif embedd == 'ConvE':
        mod = ConvE
    elif embedd == 'ComplEx':
        mod = ComplEx
    elif embedd == 'CompGCN':
        mod = CompGCN
    
    result = pipeline(training=training_set,testing=test_set,validation=validation_set,model=mod,model_kwargs=dict(embedding_dim=512),
        training_loop='sLCWA',training_kwargs=dict(num_epochs=100, batch_size=128),stopper='early',stopper_kwargs=dict(patience=10,relative_delta=0.0001,frequency=50),
        evaluation_kwargs=dict(batch_size=128),random_seed=random_seed        
    )
    if saveModel:
        result.save_to_directory(f"approach/trainedEmbeddings/{savename}")

    return result.model, result.training

def trainEmbedding(training_set, test_set, random_seed=None, saveModel = False, savename="Test", embedd="TransE", dimension = 50, epoch_nmb=5):
    '''
    Train embedding for given triples
    '''
    if embedd == 'TransE':
        if random_seed == None:
            result = pipeline(training=training_set,testing=test_set,model=TransE,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))
        else:
            result = pipeline(training=training_set,testing=test_set,model=TransE,random_seed=random_seed,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))
    elif embedd == 'DistMult':
        if random_seed == None:
            result = pipeline(training=training_set,testing=test_set,model=DistMult,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))
        else:
            result = pipeline(training=training_set,testing=test_set,model=DistMult,random_seed=random_seed,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))
    elif embedd == 'RotatE':
        if random_seed == None:
            result = pipeline(training=training_set,testing=test_set,model=RotatE,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))
        else:
            result = pipeline(training=training_set,testing=test_set,model=RotatE,random_seed=random_seed,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))
    elif embedd == 'PairRE':
        if random_seed == None:
            result = pipeline(training=training_set,testing=test_set,model=PairRE,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))
        else:
            result = pipeline(training=training_set,testing=test_set,model=PairRE,random_seed=random_seed,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))
    elif embedd == 'SimplE':
        if random_seed == None:
            result = pipeline(training=training_set,testing=test_set,model=SimplE,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))
        else:
            result = pipeline(training=training_set,testing=test_set,model=SimplE,random_seed=random_seed,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))
    elif embedd == 'ConvE':
        if random_seed == None:
            result = pipeline(training=training_set,testing=test_set,model=ConvE,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))
        else:
            result = pipeline(training=training_set,testing=test_set,model=ConvE,random_seed=random_seed,training_loop='sLCWA', model_kwargs=dict(embedding_dim=dimension),training_kwargs=dict(num_epochs=epoch_nmb))

    if saveModel:
        result.save_to_directory(f"approach/trainedEmbeddings/{savename}")

    return result.model, result.training

def trainEmbeddingOutOfBox(training_set, test_set, validation_set, random_seed=None, saveModel = False, savename="Test", embedd="TransE"):
    '''
    Train embedding for given triples
    '''
    if embedd == 'TransE':
        if random_seed == None:
            result = pipeline(training=training_set,testing=test_set,model=TransE,training_loop='sLCWA')#,training_kwargs=dict(num_epochs=10))
        else:
            result = pipeline(training=training_set,testing=test_set,model=TransE,random_seed=random_seed,training_loop='sLCWA')#,training_kwargs=dict(num_epochs=10))
    elif embedd == 'DistMult':
        if random_seed == None:
            result = pipeline(training=training_set,testing=test_set,model=DistMult,training_loop='sLCWA')#,training_kwargs=dict(num_epochs=10))
        else:
            result = pipeline(training=training_set,testing=test_set,model=DistMult,random_seed=random_seed,training_loop='sLCWA')#,training_kwargs=dict(num_epochs=10))
    elif embedd == 'RotatE':
        if random_seed == None:
            result = pipeline(training=training_set,testing=test_set,model=RotatE,training_loop='sLCWA')#,training_kwargs=dict(num_epochs=10))
        else:
            result = pipeline(training=training_set,testing=test_set,model=RotatE,random_seed=random_seed,training_loop='sLCWA')#,training_kwargs=dict(num_epochs=10))
    elif embedd == 'PairRE':
        if random_seed == None:
            result = pipeline(training=training_set,testing=test_set,model=PairRE,training_loop='sLCWA')#,training_kwargs=dict(num_epochs=10))
        else:
            result = pipeline(training=training_set,testing=test_set,model=PairRE,random_seed=random_seed,training_loop='sLCWA')#,training_kwargs=dict(num_epochs=10))
    elif embedd == 'SimplE':
        if random_seed == None:
            result = pipeline(training=training_set,testing=test_set,model=SimplE,training_loop='sLCWA')#,training_kwargs=dict(num_epochs=10))
        else:
            result = pipeline(training=training_set,testing=test_set,model=SimplE,random_seed=random_seed,training_loop='sLCWA')#,training_kwargs=dict(num_epochs=10))

    if saveModel:
        result.save_to_directory(f"approach/trainedEmbeddings/{savename}")

    return result.model, result.training

def loadModel(savename="Test", device='cuda:0'):
    model = torch.load(f"approach/trainedEmbeddings/{savename}/trained_model.pkl", map_location=device)
    return model

def createEmbeddingMaps_TransE(model, triples):
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
        # Sanity Check if conversion stays consistent from id to labels
        assert triples.entity_to_id[e] == eid, 'Entity IDs not consistent'

    r_emb = model.relation_embeddings.cpu()
    relation_ids = torch.LongTensor(range(triples.num_relations))
    r_emb_numpy = r_emb(relation_ids).detach().numpy()
    relation2embedding = {}
    for rid in range(triples.num_relations):
        r = triples.relation_id_to_label[rid]
        relation2embedding[r] = list(r_emb_numpy[rid])
        # Sanity Check if conversion stays consistent from id to labels
        assert triples.relation_to_id[r] == rid, 'Relation IDs not consistent'

    return entity2embedding, relation2embedding

def createEmbeddingMaps_DistMult(model, triples):
    '''
    create maps of the embedding to the respective entities and relations, for easier reuse
    '''
    e_emb = model.entity_representations
    entity_ids = torch.LongTensor([*range(triples.num_entities)])
    e_emb_numpy = e_emb[0](entity_ids).detach().numpy()
    entity2embedding = {}
    for eid in range(triples.num_entities):
        e = triples.entity_id_to_label[eid]
        entity2embedding[e] = list(e_emb_numpy[eid])
        # Sanity Check if conversion stays consistent from id to labels
        assert triples.entity_to_id[e] == eid, 'Entity IDs not consistent'

    r_emb = model.relation_representations
    relation_ids = torch.LongTensor([*range(triples.num_relations)])
    r_emb_numpy = r_emb[0](relation_ids).detach().numpy()
    relation2embedding = {}
    for rid in range(triples.num_relations):
        r = triples.relation_id_to_label[rid]
        relation2embedding[r] = list(r_emb_numpy[rid])
        
        # Sanity Check if conversion stays consistent from id to labels
        assert triples.relation_to_id[r] == rid, 'Relation IDs not consistent'

    return entity2embedding, relation2embedding

def getScoreForTripleListSubgraphs(X_test, emb_train_triples, model, subgraphs):
    score_list = []
    for subgraph in subgraphs:
        sum = 0
        had_element = False
        for tp in X_test:
            if (tp[0] in subgraph) and (tp[2] in subgraph):
                ten = torch.tensor([[emb_train_triples.entity_to_id[tp[0]],emb_train_triples.relation_to_id[tp[1]],emb_train_triples.entity_to_id[tp[2]]]])
                score = model.score_hrt(ten)
                score = score.detach().numpy()[0][0]
                sum += score
                had_element = True
        if had_element:
            score_list.append(sum)
        else:
            score_list.append(-1)
    return score_list

def getScoreForTripleList(X_test, emb_train_triples, model):
    score_list = []
    for tp in X_test:
        ten = torch.tensor([[emb_train_triples.entity_to_id[tp[0]],emb_train_triples.relation_to_id[tp[1]],emb_train_triples.entity_to_id[tp[2]]]])
        score = model.score_hrt(ten)
        score = score.detach().numpy()[0][0]
        score_list.append(score)
    return score_list

def baselineLP_relation(model, subgraphs, emb_train_triples, X_test, all_triples):
    start_time_clf_training = timeit.default_timer()
    LP_score_list = []
    for subgraph in subgraphs:
        sum = 0
        counter = 0
        for tp in X_test:
            if (emb_train_triples.entity_id_to_label[tp[0]] in subgraph) and (emb_train_triples.entity_id_to_label[tp[2]] in subgraph):
                tmp_scores = dict()
                for relation in range(emb_train_triples.num_relations):
                    tup = (tp[0],relation,tp[2])
                    if tup in all_triples and relation != tp[1]:
                        continue
                    ten = torch.tensor([[tp[0],relation,tp[2]]])
                    score = model.score_hrt(ten)
                    score = score.detach().numpy()[0][0]
                    tmp_scores[relation] = score
                id = max(tmp_scores, key=tmp_scores.get)
                if id == tp[1]:
                    sum += 1
                else:
                    sum += 0
                counter += 1
        if counter > 2:
            LP_score_list.append(sum/counter)
        else:
            LP_score_list.append(-100)
    return LP_score_list, start_time_clf_training, start_time_clf_training, start_time_clf_training, start_time_clf_training

def baselineLP_tail(model, subgraphs, emb_train_triples, X_test, all_triples):
    start_time_clf_training = timeit.default_timer()
    LP_score_list = []
    for subgraph in subgraphs:
        sum = 0
        counter = 0
        for tp in X_test:
            if (emb_train_triples.entity_id_to_label[tp[0]] in subgraph) and (emb_train_triples.entity_id_to_label[tp[2]] in subgraph):
                tmp_scores = dict()
                for tail in range(emb_train_triples.num_entities):
                    tup = (tp[0],tp[1],tail)
                    if tup in all_triples and tail != tp[2]:
                        continue
                    ten = torch.tensor([[tp[0],tp[1],tail]])
                    score = model.score_hrt(ten)
                    score = score.detach().numpy()[0][0]
                    tmp_scores[tail] = score
                id = max(tmp_scores, key=tmp_scores.get)
                if id == tp[2]:
                    sum += 1
                else:
                    sum += 0
                counter += 1
        if counter > 2:
            LP_score_list.append(sum/counter)
        else:
            LP_score_list.append(-100)
    return LP_score_list, start_time_clf_training, start_time_clf_training, start_time_clf_training, start_time_clf_training