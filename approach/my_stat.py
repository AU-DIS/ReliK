import torch
from sklearn.metrics import classification_report

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

    result_dict = classification_report(y_true=LP_true_list, y_pred=LP_pred_list)
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

    result_dict = classification_report(y_true=LP_true_list, y_pred=LP_pred_list)
    return result_dict

def fullGraphLP_F1(model, LP_triples_pos, emb_train_triples, all_triples):
    LP_pred_list_r = []
    LP_true_list_r = []
    label_names_r = []
    LP_pred_list_t = []
    LP_true_list_t = []
    label_names_t = []

    tail_names_set = set()
    relation_names_set = set()

    for label in range(emb_train_triples.num_relations):
        label_names_r.append(emb_train_triples.relation_id_to_label[label])
        relation_names_set.add(label)
    for label in range(emb_train_triples.num_entities):
        label_names_t.append(emb_train_triples.entity_id_to_label[label])
        tail_names_set.add(label)
    for tp in LP_triples_pos:
        if tp[1] in relation_names_set:
            relation_names_set.remove(tp[1])

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
        LP_pred_list_r.append(id)
        LP_true_list_r.append(tp[1])

        if tp[2] in tail_names_set:
            tail_names_set.remove(tp[2])

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
        LP_pred_list_t.append(id)
        LP_true_list_t.append(tp[2])
    for i in tail_names_set:
        label_names_t.remove(emb_train_triples.entity_id_to_label[i])
    for i in relation_names_set:
        label_names_r.remove(emb_train_triples.relation_id_to_label[i])

    result_dict_r = classification_report(y_true=LP_true_list_r, y_pred=LP_pred_list_r, output_dict=True)
    result_dict_t = classification_report(y_true=LP_true_list_t, y_pred=LP_pred_list_t, output_dict=True)
    return result_dict_t, result_dict_r