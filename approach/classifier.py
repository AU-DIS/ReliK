import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import torch

def trainClassifier(X_train, Y_train, entity2embedding, relation2embedding, type='LogisticRegression'):
    '''
    Train specific Classifier
    '''
    if type == 'SVC':
        clf = SVC()
    elif type == 'LogisticRegression':
        clf = LogisticRegression(max_iter=5000)
    elif type == 'LinearRegression':
        clf = LinearRegression()
    elif type == 'gboost':
        clf = GradientBoostingClassifier()
    elif type == 'randomForest':
        clf = RandomForestClassifier()
    X_train_emb = []
    for tp in X_train:
        X_train_emb.append([entity2embedding[tp[0]],relation2embedding[tp[1]],entity2embedding[tp[2]]])
    X_train_emb = np.array(X_train_emb)
    nsamples, nx, ny = X_train_emb.shape
    X_train_emb = X_train_emb.reshape((nsamples,nx*ny))
    clf.fit(X_train_emb, Y_train)
    return clf


def prepareTrainTestData(pos_triples, neg_triples, triples, test_size=0.33):
    '''
    creating data for classifier training/testing, with labels from the triples
    '''
    ds_pos = [] 
    ds_neg = []
    ds_all = []
    for t in pos_triples:
        ds_pos.append([triples.entity_id_to_label[t[0]], triples.relation_id_to_label[t[1]], triples.entity_id_to_label[t[2]], 1])
    for t in neg_triples:
        ds_neg.append([triples.entity_id_to_label[t[0]], triples.relation_id_to_label[t[1]], triples.entity_id_to_label[t[2]], 0])
    ds_all = ds_pos + ds_neg

    dataset = np.array(ds_all)

    X = dataset[:, :-1]
    y = dataset[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def prepareTrainTestDataSplit(pos_triples, neg_triples, triples, entity_to_id_map, relation_to_id_map, test_size=0.33):
    '''
    creating data for classifier training/testing, with labels from the triples
    '''
    ds_pos_X = []
    ds_pos_y = []
    ds_neg_X = []
    ds_neg_y = [] 
    ds_neg = []
    ds_all = []
    for t in pos_triples:
        ds_pos_X.append(torch.tensor([[t[0],t[1],t[2]]]))
        ds_pos_y.append(1)
    for t in neg_triples:
        ds_neg_X.append(torch.tensor([[t[0],t[1],t[2]]]))
        ds_neg_y.append(0)
    #ds_all = ds_pos + ds_neg

    X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(ds_pos_X, ds_pos_y, test_size=test_size)

    X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(ds_neg_X, ds_neg_y, test_size=test_size)
    return X_train_pos, X_test_pos, y_train_pos, y_test_pos, X_train_neg, X_test_neg, y_train_neg, y_test_neg

def testClassifier(classifier, X_test, y_test, entity2embedding, relation2embedding):
    LP_test_score = []
    index = 0
    for tp in X_test:
        LP_test_score.append(classifier.score([[*entity2embedding[tp[0]],*relation2embedding[tp[1]],*entity2embedding[tp[2]]]],[y_test[index]]))
        index += 1
    return LP_test_score

def testClassifierSubgraphs(classifier, X_test, y_test, entity2embedding, relation2embedding, subgraphs):
    LP_test_score = []
    for subgraph in subgraphs:
        bigcount = 0
        X_test_emb = []
        y_test_emb = []
        index = 0
        for tp in X_test:
            if ((tp[0] in subgraph) or (tp[2] in subgraph)):
                X_test_emb.append([entity2embedding[tp[0]],relation2embedding[tp[1]],entity2embedding[tp[2]]])
                y_test_emb.append(y_test[index])
            index += 1
        y_test_emb = np.array(y_test_emb)
        if X_test_emb == []:
            LP_test_score.append(-100)
            continue
        if len(X_test_emb) == 1:
            LP_test_score.append(-100)
            continue
        X_test_emb = np.array(X_test_emb)
        nsamples, nx, ny = X_test_emb.shape
        X_test_emb = X_test_emb.reshape((nsamples,nx*ny))
        LP_test_score.append(classifier.score(X_test_emb, y_test_emb))
        bigcount += 1
        if bigcount % 100 == 0:
            print(f'Have tested {bigcount} subgraphs')

    return LP_test_score