import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def trainLPClassifier(X_train, Y_train, type='SVC'):
    '''
    Train specific Classifier
    '''
    if type == 'SVC':
        clf = SVC(max_iter=2000)
    elif type == 'LogisticRegression':
        clf = LogisticRegression(max_iter=2000)
    
    clf.fit(X_train, Y_train)

    return clf


def prepareTrainTestData(pos_triples, neg_triples, entity2embedding, relation2embedding, triples):
    '''
    creating data for classifier training/testing, with labels from the triples
    '''
    ds_pos = [] 
    ds_neg = []
    ds_all = []

    for t in pos_triples:
        ds_pos.append([*entity2embedding[triples.entity_id_to_label[t[0]]], *relation2embedding[triples.relation_id_to_label[t[1]]], *entity2embedding[triples.entity_id_to_label[t[2]]], 1])
    for t in neg_triples:
        ds_neg.append([*entity2embedding[triples.entity_id_to_label[t[0]]], *relation2embedding[triples.relation_id_to_label[t[1]]], *entity2embedding[triples.entity_id_to_label[t[2]]], 0])

    ds_all = ds_pos + ds_neg

    dataset = np.array(ds_all)

    X = dataset[:, :-1]
    y = dataset[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_train, X_test, y_train, y_test