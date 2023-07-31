import torch
from sklearn.metrics import classification_report

import classifier as cla

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

def fullGraphLP_classifier(LP_triples_pos, LP_triples_neg, emb_train_triples, entity2embedding, relation2embedding):
    X_train, X_test, y_train, y_test = cla.prepareTrainTestData(LP_triples_pos, LP_triples_neg, emb_train_triples, test_size=0.5)
    clf = cla.trainClassifier(X_train, y_train, entity2embedding, relation2embedding)
    LP_score_list = cla.testClassifier(clf, X_test, y_test, entity2embedding, relation2embedding)
    fullgraph_score = sum(LP_score_list)/len(LP_score_list)
    return fullgraph_score, LP_score_list

def fullGraphLP_F1_tail(model, LP_triples_pos, emb_train_triples, all_triples):
    LP_pred_list = []
    LP_true_list = []
    label_names = []
    for label in range(emb_train_triples.num_entities):
        label_names.append(emb_train_triples.entity_id_to_label[label])
    for tp in LP_triples_pos:
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
        LP_pred_list.append(id)
        LP_true_list.append(tp[2])

    result_dict = classification_report(y_true=LP_true_list, y_pred=LP_pred_list, output_dict=True)
    return result_dict

def fullGraphLP_F1_relation(model, LP_triples_pos, emb_train_triples, all_triples):
    LP_pred_list = []
    LP_true_list = []
    label_names = []
    for label in range(emb_train_triples.num_relations):
        label_names.append(emb_train_triples.relation_id_to_label[label])
    for tp in LP_triples_pos:
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
        LP_pred_list.append(id)
        LP_true_list.append(tp[1])

    result_dict = classification_report(y_true=LP_true_list, y_pred=LP_pred_list, output_dict=True)
    return result_dict

def fullGraphLP_F1(model, LP_triples_pos, emb_train_triples, all_triples):

    result_dict_r = fullGraphLP_F1_relation(model, LP_triples_pos, emb_train_triples, all_triples)
    result_dict_t = fullGraphLP_F1_tail(model, LP_triples_pos, emb_train_triples, all_triples)

    return result_dict_t, result_dict_r