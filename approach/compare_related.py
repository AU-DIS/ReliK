import pykeen.datasets as dat
from pykeen.models import TransE
from pykeen.models import DistMult
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import settings as sett
import csv
import os
import torch

import main 

def fullGraphLP_basic_tail(model, LP_triples_pos, emb_train_triples, all_triples):
    LP_score_list = []
    sum = 0
    for tp in LP_triples_pos:
        tmp_scores = dict()
        for tail in range(emb_train_triples.num_entities):
            tup = (tp[0],tp[1],tail)
            if tup in all_triples and tail != tp[2]:
                #print(f'Found pos triple that is not tested triple with tail:{tail} instead of test: {tp[2]}')
                continue
            ten = torch.tensor([[tp[0],tp[1],tail]])
            score = model.score_hrt(ten)
            score = score.detach().numpy()[0][0]
            tmp_scores[tail] = score
        id = max(tmp_scores, key=tmp_scores.get)
        if id == tp[2]:
            sum += 1
            LP_score_list.append(1)
        else:
            sum += 0
            LP_score_list.append(0)
    fullgraph_score = sum/len(LP_score_list)
    return fullgraph_score, LP_score_list

def fullGraphLP_basic_relation(model, LP_triples_pos, emb_train_triples, all_triples):
    LP_score_list = []
    sum = 0
    for tp in LP_triples_pos:
        tmp_scores = dict()
        for relation in range(emb_train_triples.num_relations):
            tup = (tp[0],relation,tp[2])
            if tup in all_triples and relation != tp[1]:
                #print(f'Found pos triple that is not tested triple with relation:{relation} instead of test: {tp[1]}')
                continue
            ten = torch.tensor([[tp[0],relation,tp[2]]])
            score = model.score_hrt(ten)
            score = score.detach().numpy()[0][0]
            tmp_scores[relation] = score
        id = max(tmp_scores, key=tmp_scores.get)
        if id == tp[1]:
            sum += 1
            LP_score_list.append(1)
        else:
            sum += 0
            LP_score_list.append(0)
    fullgraph_score = sum/len(LP_score_list)
    return fullgraph_score, LP_score_list

def compare_to_related():
    if sett.DATASETNAME == 'CodexSmall':
        dataset = dat.CoDExSmall()
    elif sett.DATASETNAME == 'CodexMedium':
        dataset = dat.CoDExMedium()
    elif sett.DATASETNAME == 'CodexLarge':
        dataset = dat.CoDExLarge()
    test_data = dataset.testing
    train_data = dataset.training
    validation_data = dataset.validation

    all_triples_set = set()
    all_triples_tensor = torch.cat((test_data.mapped_triples,train_data.mapped_triples))
    for tup in all_triples_tensor.tolist():
        all_triples_set.add((tup[0],tup[1],tup[2]))

    result = pipeline(training=train_data,testing=test_data,validation=validation_data,model=TransE,model_kwargs=dict(embedding_dim=512, scoring_fct_norm=1),
        training_loop='LCWA',training_kwargs=dict(num_epochs=50, batch_size=128),stopper='early',stopper_kwargs=dict(patience=10,relative_delta=0.0001,frequency=50),
        evaluation_kwargs=dict(batch_size=128)        
    )
    score_t = fullGraphLP_basic_tail(result.model, test_data.mapped_triples.tolist(), test_data, all_triples_set)
    score_r = fullGraphLP_basic_relation(result.model, test_data.mapped_triples.tolist(), test_data, all_triples_set)
    
    path = f'approach/scoreData/compareData'
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)
    
    c = open(f'{path}/{sett.DATASETNAME}_comparison.csv', "w")
    writer = csv.writer(c)
    data = ['tail','relation','triple classification']
    writer.writerow(data)
    data = [score_t[0],score_r[0]]
    writer.writerow(data)
    c.close()
    return

def compare_to_related2():
    dataset = dat.CoDExSmall()
    test_data = dataset.testing
    train_data = dataset.training
    validation_data = dataset.validation #0.19803063457330417

    all_triples_set = set()
    all_triples_tensor = torch.cat((test_data.mapped_triples,train_data.mapped_triples))
    for tup in all_triples_tensor.tolist():
        all_triples_set.add((tup[0],tup[1],tup[2]))

    result = pipeline(training=train_data,testing=test_data,validation=validation_data,model=TransE,model_kwargs=dict(embedding_dim=512, scoring_fct_norm=1),
        optimizer='Adagrad',optimizer_kwargs=dict(lr=0.04121772717931592),training_loop='LCWA',training_kwargs=dict(num_epochs=400, batch_size=128),stopper='early',
        stopper_kwargs=dict(patience=10,relative_delta=0.0001,frequency=50),
        evaluation_kwargs=dict(batch_size=128)
    )
    #lr_scheduler='ReduceLROnPlateau', lr_scheduler_kwargs=dict(mode='max',factor=0.95, patience=6, threshold=0.0001)
    score = fullGraphLP_basic_tail(result.model, test_data.mapped_triples.tolist(), test_data, all_triples_set)
    #score = fullGraphLP_basic_relation(result.model, test_data.mapped_triples.tolist(), test_data, all_triples_set)
    print(score[0])