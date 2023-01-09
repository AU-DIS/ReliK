import embedding as emb
import datahandler as dh
import classifier as cla
import settings as sett
import reliability as rel
import compare_related as compre
import my_stat

import torch
import csv
import timeit
import os
import random

from sklearn.model_selection import train_test_split
from pykeen.triples import TriplesFactory

def retrieveOrTrainEmbedding():
    # Getting Data from dataset, as tensor of triples and mappings from entity to ids
    all_triples, all_triples_set, entity_to_id_map, relation_to_id_map, test_triples, validation_triples = emb.getDataFromPykeen(datasetname=sett.DATASETNAME)
    
    # Split Data between embedding and LP classifier part
    if sett.OUT_OF_BOX or sett.MORE_TRAIN:
        emb_triples = TriplesFactory(all_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
        emb_train_triples = TriplesFactory(all_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
        LP_triples = test_triples.mapped_triples
        LP_triples_pos = LP_triples.tolist()
    else:
        emb_triples, LP_triples = train_test_split(all_triples, test_size=sett.LP_EMB_SPLIT)
        if sett.MAKE_TRAINING_SMALLER:
            emb_triples, throwaway = train_test_split(emb_triples, test_size=sett.SMALLER_RATIO)
            LP_triples, throwaway = train_test_split(LP_triples, test_size=sett.SMALLER_RATIO)
        LP_triples_pos = LP_triples.tolist()

        # Split Data between embedding train and test
        emb_train_triples = TriplesFactory(emb_triples,entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)
        emb_test_triples = TriplesFactory(emb_triples[:1,:],entity_to_id=entity_to_id_map,relation_to_id=relation_to_id_map)

        # Sanity Check if triples have been changed in the Factory
        #assert torch.equal(emb_triples,emb_train_triples.mapped_triples), "Triples changed in creation of Triples Factory"
        #assert entity_to_id_map == emb_train_triples.entity_to_id, "Entity mapping changed in creation of Triples Factory"
        #assert relation_to_id_map == emb_train_triples.relation_to_id, "Relation mapping changed in creation of Triples Factory"

    # Train and create embedding model
    
    isFile = os.path.isfile(f"approach/trainedEmbeddings/{sett.SAVENAME}/trained_model.pkl")
    if sett.LOAD_MODEL and isFile:
        emb_model = emb.loadModel(savename=sett.SAVENAME)
    else:
        if sett.OUT_OF_BOX:
            emb_model, emb_triples_used = emb.trainEmbeddingOutOfBox(emb_triples, test_triples, validation_triples, random_seed=42, saveModel=sett.STORE_MODEL, savename = sett.SAVENAME, embedd = sett.EMBEDDING_TYPE)
        elif sett.MORE_TRAIN:
            emb_model, emb_triples_used = emb.trainEmbeddingMore(emb_train_triples, test_triples, validation_triples, random_seed=42, saveModel=sett.STORE_MODEL, savename = sett.SAVENAME, embedd = sett.EMBEDDING_TYPE, dimension = sett.DIMENSIONS, epoch_nmb = sett.EPOCH_NMB)
        else:
            emb_model, emb_triples_used = emb.trainEmbedding(emb_train_triples, emb_test_triples, random_seed=42, saveModel=sett.STORE_MODEL, savename = sett.SAVENAME, embedd = sett.EMBEDDING_TYPE, dimension = sett.DIMENSIONS)
    
    if sett.EMBEDDING_TYPE == 'TransE':
        entity2embedding, relation2embedding = emb.createEmbeddingMaps_TransE(emb_model, emb_train_triples)
    elif sett.EMBEDDING_TYPE == 'DistMult':
        entity2embedding, relation2embedding = emb.createEmbeddingMaps_DistMult(emb_model, emb_train_triples)
    else:
        entity2embedding, relation2embedding = emb.createEmbeddingMaps_DistMult(emb_model, emb_train_triples)

    # Sanity Check if triples have been changed while doing the embedding
    #if not sett.LOAD_MODEL:
        #assert torch.equal(emb_triples,emb_triples_used.mapped_triples), "Triples changed after embedding"
        #assert entity_to_id_map == emb_triples_used.entity_to_id, "Entity mapping changed after embedding"
        #assert relation_to_id_map == emb_triples_used.relation_to_id, "Relation mapping changed after embedding"

    return emb_model, all_triples, all_triples_set, LP_triples_pos, emb_train_triples, entity2embedding, relation2embedding, entity_to_id_map, relation_to_id_map

def makeNegTriples(all_triples_set, all_triples, emb_train_triples):
    neg_triples, related_nodes = dh.createNegTripleHT(all_triples_set, all_triples.tolist(), emb_train_triples)
    some_neg_triples = dh.createNegTripleRelation(all_triples_set, all_triples.tolist(), emb_train_triples)
    throwaway, LP_triples_neg = train_test_split(neg_triples, test_size=sett.LP_EMB_SPLIT)

    return neg_triples, some_neg_triples, LP_triples_neg, related_nodes

def makeTCPart(LP_triples_pos, LP_triples_neg, entity2embedding, relation2embedding, subgraphs, emb_train_triples):

    start_time_clf_training = timeit.default_timer()
    X_train, X_test, y_train, y_test = cla.prepareTrainTestData(LP_triples_pos, LP_triples_neg, emb_train_triples)
    clf = cla.trainClassifier(X_train, y_train, entity2embedding, relation2embedding)
    end_time_clf_training = timeit.default_timer()
    LP_test_score = -1
    start_time_LP_score = timeit.default_timer()
    
    LP_test_score = cla.testClassifierSubgraphs(clf, X_test, y_test, entity2embedding, relation2embedding, subgraphs)
    end_time_LP_score = timeit.default_timer()

    return LP_test_score, start_time_clf_training, end_time_clf_training, start_time_LP_score, end_time_LP_score

if __name__ == "__main__":
    start_time_complete = timeit.default_timer()
    # Get and do Embedding
    start_time_emb_training = timeit.default_timer()
    if sett.COMPARISON_TO_RELATED:
        compre.compare_to_related()
        quit()
    emb_model, all_triples, all_triples_set, LP_triples_pos, emb_train_triples, entity2embedding, relation2embedding, entity_to_id_map, relation_to_id_map = retrieveOrTrainEmbedding()
    end_time_emb_training = timeit.default_timer()
    print(f'finished Embedding')

    # Get negative triples
    start_time_create_neg_samples = timeit.default_timer()
    isFile = os.path.isfile(f"approach/trainedEmbeddings/{sett.SAVENAME}/neg_triples.csv")
    if sett.LOAD_TRIPLES and isFile:
        neg_triples = dh.loadTriples(f"approach/trainedEmbeddings/{sett.SAVENAME}/neg_triples")
        some_neg_triples = dh.loadTriples(f"approach/trainedEmbeddings/{sett.SAVENAME}/some_neg_triples")
        throwaway, LP_triples_neg = train_test_split(neg_triples, test_size=sett.LP_EMB_SPLIT)
    else:
        neg_triples, some_neg_triples, LP_triples_neg, related_nodes = makeNegTriples(all_triples_set, all_triples, emb_train_triples)
        dh.storeTriples(f"approach/trainedEmbeddings/{sett.SAVENAME}/neg_triples", neg_triples)
        dh.storeTriples(f"approach/trainedEmbeddings/{sett.SAVENAME}/some_neg_triples", some_neg_triples)
    
    isFile = os.path.isfile(f"approach/trainedEmbeddings/{sett.SAVENAME}/related.csv")
    if sett.LOAD_TRIPLES and isFile:
        related_nodes = dh.loadRelated(f"approach/trainedEmbeddings/{sett.SAVENAME}/related")
    else:
        t1, t2, t3, related_nodes = makeNegTriples(all_triples_set, all_triples, emb_train_triples)
        dh.storeRelated(f"approach/trainedEmbeddings/{sett.SAVENAME}/related", related_nodes)
    end_time_create_neg_samples = timeit.default_timer()
    
    path = f'approach/scoreData/{sett.NAME_OF_RUN}'

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)

    if sett.DOFULLGRAPHLP:
        fullgraph_score_tail, score_list_tail = my_stat.fullGraphLP_basic_tail(emb_model, LP_triples_pos, emb_train_triples, all_triples_set)
        fullgraph_score_relation, score_list_relation = my_stat.fullGraphLP_basic_relation(emb_model, LP_triples_pos, emb_train_triples, all_triples_set)
        fullgraph_score_clf, score_list_clf = my_stat.fullGraphLP_classifier(LP_triples_pos, LP_triples_neg, emb_train_triples, entity2embedding, relation2embedding)
        c = open(f'{path}/{sett.NAME_OF_RUN}_fullGraph.csv', "w")
        writer = csv.writer(c)
        data = ['triple', 'basic head', 'basic relation', 'classifier']
        writer.writerow(data)
        data = [-100, fullgraph_score_tail, fullgraph_score_relation, fullgraph_score_clf]
        writer.writerow(data)
        for i in range(min(len(score_list_tail),len(score_list_relation))):
            data = [i, score_list_tail[i], score_list_relation[i], score_list_clf[i]]
            writer.writerow(data)
        c.close()
    
    if sett.DO_F1:
        fg_score_tail, fg_score_relation = my_stat.fullGraphLP_F1(emb_model, LP_triples_pos, emb_train_triples, all_triples_set)

        c = open(f'{path}/{sett.NAME_OF_RUN}_fullGraph_tail_prediction_stats.csv', "w")
        writer = csv.writer(c)
        data = ['entity', 'precision', 'recall', 'f1-score', 'support']
        writer.writerow(data)
        for i in fg_score_tail:
            if i=='accuracy':
                data =[i,'','',fg_score_tail[i],'']
            else:
                data = [i,fg_score_tail[i]['precision'], fg_score_tail[i]['recall'], fg_score_tail[i]['f1-score'], fg_score_tail[i]['support']]
            writer.writerow(data)
        c.close()

        c = open(f'{path}/{sett.NAME_OF_RUN}_fullGraph_relation_prediction_stats.csv', "w")
        writer = csv.writer(c)
        data = ['entity', 'precision', 'recall', 'f1-score', 'support']
        writer.writerow(data)
        for i in fg_score_relation:
            if i=='accuracy':
                data =[i,'','',fg_score_tail[i],'']
            else:
                data = [i,fg_score_relation[i]['precision'], fg_score_relation[i]['recall'], fg_score_relation[i]['f1-score'], fg_score_relation[i]['support']]
            writer.writerow(data)
            
        c.close()

    if sett.DOSCORE:
        # Create or get subgraphs
        start_time_create_subgraphs = timeit.default_timer()
        isFile = os.path.isfile(f"approach/trainedEmbeddings/{sett.SAVENAME}/subgraphs_{sett.SIZE_OF_SUBGRAPHS}.csv")
        if sett.LOAD_SUBGRAPHS and isFile:
            subgraphs = dh.loadSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}")
            print(f'loaded Subgraphs')
            if len(subgraphs) < sett.AMOUNT_OF_SUBGRAPHS:
                subgraphs_new = dh.createSubGraphs(all_triples, entity_to_id_map, relation_to_id_map, size_of_graphs=sett.SIZE_OF_SUBGRAPHS, number_of_graphs=(sett.AMOUNT_OF_SUBGRAPHS-len(subgraphs)), restart=sett.RESET_PROB)
                dh.storeSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}",subgraphs_new)
                subgraphs = subgraphs + subgraphs_new
                print(f'created Subgraphs')
            if len(subgraphs) > sett.AMOUNT_OF_SUBGRAPHS:
                subgraphs = random.sample(subgraphs, sett.AMOUNT_OF_SUBGRAPHS)
        else:
            subgraphs = dh.createSubGraphs(all_triples, entity_to_id_map, relation_to_id_map, size_of_graphs=sett.SIZE_OF_SUBGRAPHS, number_of_graphs=sett.AMOUNT_OF_SUBGRAPHS, restart=sett.RESET_PROB)
            dh.storeSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}",subgraphs)
            print(f'created Subgraphs')
        end_time_create_subgraphs = timeit.default_timer()
        
        # train classifier and test it
        LP_test_score_tail, start_time_clf_training, end_time_clf_training, start_time_LP_score, end_time_LP_score = emb.baselineLP_tail(emb_model, subgraphs, emb_train_triples, LP_triples_pos, all_triples_set)
        LP_test_score_rel, start_time_clf_training, end_time_clf_training, start_time_LP_score, end_time_LP_score = emb.baselineLP_relation(emb_model, subgraphs, emb_train_triples, LP_triples_pos, all_triples_set)
        LP_test_score, start_time_clf_training, end_time_clf_training, start_time_LP_score, end_time_LP_score = makeTCPart(LP_triples_pos, LP_triples_neg, entity2embedding, relation2embedding, subgraphs, emb_train_triples)
        print(f'finished LP Score')
        
        # Get reliability value for KG
        start_time_reliability = timeit.default_timer()
        if sett.EMBEDDING_TYPE == 'DistMult':
            local_reliability_score = rel.reliability_DistMult_local_normalization_as_Sum(all_triples_set, emb_train_triples, emb_model, entity2embedding, relation2embedding, subgraphs)
        else:
            local_reliability_score = rel.reliability_local_normalization_as_Sum(all_triples_set, emb_train_triples, emb_model, entity2embedding, relation2embedding, subgraphs, norm=2)
        end_time_reliability = timeit.default_timer()

        if sett.DOSCORE_RELATION_BASED and sett.EMBEDDING_TYPE == 'DistMult':
            relation_reliability_score = rel.reliability_DistMult_local_normalization_as_Sum_Rel_Freq(all_triples_set, emb_train_triples, emb_model, entity2embedding, relation2embedding, subgraphs)
        elif sett.DOSCORE_RELATION_BASED:
            relation_reliability_score = rel.reliability_local_normalization_as_Sum_Rel_Freq(all_triples_set, emb_train_triples, emb_model, entity2embedding, relation2embedding, subgraphs, norm=2)
        end_time_complete = timeit.default_timer()

        time_complete = end_time_complete - start_time_complete
        time_emb_training = end_time_emb_training - start_time_emb_training
        time_clf_training = end_time_clf_training - start_time_clf_training
        time_LP_score = end_time_LP_score - start_time_LP_score
        time_reliability = end_time_reliability - start_time_reliability
        time_measured = time_emb_training + time_clf_training + time_LP_score + time_reliability
        time_percentage = time_measured/time_complete
        print()
        print(f'Complete time: {format(time_complete, ".4f")}')
        print(f'Embedding training time: {format(time_emb_training, ".4f")}')
        print(f'Classifier training time: {format(time_clf_training, ".4f")}')
        print(f'Link Prediction score time: {format(time_LP_score, ".4f")}')
        print(f'Reliability score time: {format(time_reliability, ".4f")}')
        print(f'Time from measured steps {format(time_measured, ".4f")}')
        print(f'Time precentage from measured steps {format(time_percentage, ".4f")}')
        print()

        if sett.DOSCORE_RELATION_BASED:
            c = open(f'{path}/{sett.NAME_OF_RUN}_rel.csv', "w")
            writer = csv.writer(c)
            data = ['subgraph', 'LP_test_score', 'local_reliability_score', 'reliabiliy_relation', 'LP_basic_tail', 'LP_basic_relation']
            writer.writerow(data)
            for i in range(len(LP_test_score_tail)):
                data = [i, LP_test_score[i], local_reliability_score[i], relation_reliability_score[i], LP_test_score_tail[i], LP_test_score_rel[i]]
                writer.writerow(data)
        else:
            c = open(f'{path}/{sett.NAME_OF_RUN}.csv', "w")
            writer = csv.writer(c)
            data = ['subgraph', 'LP_test_score', 'local_reliability_score', 'LP_basic_tail', 'LP_basic_relation']
            writer.writerow(data)
            for i in range(len(LP_test_score)):
                data = [i, LP_test_score[i], local_reliability_score[i], LP_test_score_tail[i], LP_test_score_rel[i]]
                writer.writerow(data)
        c.close()
        
        newFile = True
        if os.path.exists('approach/scoreData/timeData.csv'):
            newFile = False
        
        c = open(f'approach/scoreData/timeData.csv', "a+")
        writer = csv.writer(c)
        if newFile:
            data = ['experiment run','dataset', 'number of subgraphs', 'size of subgraphs', 'complete time', 'embedding training time', 'classifier training time', 'LP score time', 'reliability time', 'time from measured', 'percentage of measured', 'norm']
            writer.writerow(data)
        data = [sett.NAME_OF_RUN, sett.DATASETNAME, sett.AMOUNT_OF_SUBGRAPHS, sett.SIZE_OF_SUBGRAPHS, time_complete, time_emb_training, time_clf_training, time_LP_score, time_reliability, time_measured, time_percentage]
        writer.writerow(data)
        c.close()

    if sett.DO_AVERAGE:
        isFile = os.path.isfile(f"approach/trainedEmbeddings/{sett.SAVENAME}/subgraphs_{sett.SIZE_OF_SUBGRAPHS}.csv")
        if sett.LOAD_SUBGRAPHS and isFile:
            subgraphs = dh.loadSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}")
            print(f'loaded Subgraphs')
            if len(subgraphs) < sett.AMOUNT_OF_SUBGRAPHS:
                subgraphs_new = dh.createSubGraphs(all_triples, entity_to_id_map, relation_to_id_map, size_of_graphs=sett.SIZE_OF_SUBGRAPHS, number_of_graphs=(sett.AMOUNT_OF_SUBGRAPHS-len(subgraphs)), restart=sett.RESET_PROB)
                dh.storeSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}",subgraphs_new)
                subgraphs = subgraphs + subgraphs_new
                print(f'created Subgraphs')
            if len(subgraphs) > sett.AMOUNT_OF_SUBGRAPHS:
                subgraphs = random.sample(subgraphs, sett.AMOUNT_OF_SUBGRAPHS)
        else:
            subgraphs = dh.createSubGraphs(all_triples, entity_to_id_map, relation_to_id_map, size_of_graphs=sett.SIZE_OF_SUBGRAPHS, number_of_graphs=sett.AMOUNT_OF_SUBGRAPHS, restart=sett.RESET_PROB)
            dh.storeSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}",subgraphs)
            print(f'created Subgraphs')
        local_reliability_score, max_score, min_score = rel.reliability_local_normalization_as_Difference_in_Average(all_triples_set, emb_train_triples, emb_model, entity2embedding, relation2embedding, subgraphs)
        c = open(f'{path}/{sett.NAME_OF_RUN}_average.csv', "w")
        writer = csv.writer(c)
        data = ['subgraph', 'local_reliability_score', max_score, min_score]
        writer.writerow(data)
        for i in range(len(local_reliability_score)):
            data = [i, local_reliability_score[i]]
            writer.writerow(data)
        c.close()

    if sett.DO_RATIO:
        isFile = os.path.isfile(f"approach/trainedEmbeddings/{sett.SAVENAME}/subgraphs_{sett.SIZE_OF_SUBGRAPHS}.csv")
        if sett.LOAD_SUBGRAPHS and isFile:
            subgraphs = dh.loadSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}")
            print(f'loaded Subgraphs')
            if len(subgraphs) < sett.AMOUNT_OF_SUBGRAPHS:
                subgraphs_new = dh.createSubGraphs(all_triples, entity_to_id_map, relation_to_id_map, size_of_graphs=sett.SIZE_OF_SUBGRAPHS, number_of_graphs=(sett.AMOUNT_OF_SUBGRAPHS-len(subgraphs)), restart=sett.RESET_PROB)
                dh.storeSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}",subgraphs_new)
                subgraphs = subgraphs + subgraphs_new
                print(f'created Subgraphs')
            if len(subgraphs) > sett.AMOUNT_OF_SUBGRAPHS:
                subgraphs = random.sample(subgraphs, sett.AMOUNT_OF_SUBGRAPHS)
        else:
            subgraphs = dh.createSubGraphs(all_triples, entity_to_id_map, relation_to_id_map, size_of_graphs=sett.SIZE_OF_SUBGRAPHS, number_of_graphs=sett.AMOUNT_OF_SUBGRAPHS, restart=sett.RESET_PROB)
            dh.storeSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}",subgraphs)
            print(f'created Subgraphs')
        if sett.EMBEDDING_TYPE == 'DistMult':
            local_reliability_score = rel.reliability_DistMult_local_normalization_Ratio(all_triples_set, emb_train_triples, emb_model, entity2embedding, relation2embedding, subgraphs)
        else:
            local_reliability_score = rel.reliability_local_normalization_Ratio(all_triples_set, emb_train_triples, emb_model, entity2embedding, relation2embedding, subgraphs)
        c = open(f'{path}/{sett.NAME_OF_RUN}_ratio.csv', "w")
        writer = csv.writer(c)
        data = ['subgraph', 'local_reliability_score', max_score, min_score]
        writer.writerow(data)
        for i in range(len(local_reliability_score)):
            data = [i, local_reliability_score[i]]
            writer.writerow(data)
        c.close()

    if sett.DO_MIN_MAX:
        isFile = os.path.isfile(f"approach/trainedEmbeddings/{sett.SAVENAME}/subgraphs_{sett.SIZE_OF_SUBGRAPHS}.csv")
        if sett.LOAD_SUBGRAPHS and isFile:
            subgraphs = dh.loadSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}")
            print(f'loaded Subgraphs')
            if len(subgraphs) < sett.AMOUNT_OF_SUBGRAPHS:
                subgraphs_new = dh.createSubGraphs(all_triples, entity_to_id_map, relation_to_id_map, size_of_graphs=sett.SIZE_OF_SUBGRAPHS, number_of_graphs=(sett.AMOUNT_OF_SUBGRAPHS-len(subgraphs)), restart=sett.RESET_PROB)
                dh.storeSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}",subgraphs_new)
                subgraphs = subgraphs + subgraphs_new
                print(f'created Subgraphs')
            if len(subgraphs) > sett.AMOUNT_OF_SUBGRAPHS:
                subgraphs = random.sample(subgraphs, sett.AMOUNT_OF_SUBGRAPHS)
        else:
            subgraphs = dh.createSubGraphs(all_triples, entity_to_id_map, relation_to_id_map, size_of_graphs=sett.SIZE_OF_SUBGRAPHS, number_of_graphs=sett.AMOUNT_OF_SUBGRAPHS, restart=sett.RESET_PROB)
            dh.storeSubGraphs(f"approach/trainedEmbeddings/{sett.SAVENAME}",subgraphs)
            print(f'created Subgraphs')
        local_reliability_score, max_pos, min_pos, max_neg, min_neg = rel.reliability_local_normalization_as_Difference_MaxMin(all_triples_set, emb_train_triples, emb_model, entity2embedding, relation2embedding, subgraphs)
        c = open(f'{path}/{sett.NAME_OF_RUN}_maxmin.csv', "w")
        writer = csv.writer(c)
        data = ['subgraph', 'local_reliability_score', 'max_pos', 'min_pos', 'max_neg', 'min_neg']
        writer.writerow(data)
        for i in range(len(local_reliability_score)):
            data = [i, local_reliability_score[i], max_pos[i], min_pos[i], max_neg[i], min_neg[i]]
            writer.writerow(data)
        c.close()
        


    if sett.DODIS:
        # data and scores between positive, negative and somewhat negatives triples
        # to calculate distribution

        X_some_neg, y_some_neg = dh.convertListToData(some_neg_triples, emb_train_triples, pos_sample=False)
        X_neg, y_neg = dh.convertListToData(neg_triples, emb_train_triples, pos_sample=False)
        X_pos, y_pos = dh.convertListToData(all_triples.tolist(), emb_train_triples, pos_sample=True)

        emb_dis_score_pos = emb.getScoreForTripleList(X_pos, emb_train_triples, emb_model)
        emb_dis_score_neg = emb.getScoreForTripleList(X_neg, emb_train_triples, emb_model)
        emb_dis_score_some_neg = emb.getScoreForTripleList(X_some_neg, emb_train_triples, emb_model)

        c = open(f'{path}/{sett.NAME_OF_RUN}_emb_dis.csv', "w")
        writer = csv.writer(c)
        data = ['count','pos','some_neg','neg']
        writer.writerow(data)
        count = 0
        for ele in emb_dis_score_pos:
            data = [count]
            data.append(emb_dis_score_pos[count])
            data.append(emb_dis_score_some_neg[count])
            data.append(emb_dis_score_neg[count])
            writer.writerow(data)
            count += 1
        c.close()
    if sett.DODIS_RELATION:
        # data and scores between positive, negative and somewhat negatives triples
        # to calculate distribution
        X_some_neg, y_some_neg = dh.convertListToData_Relation(some_neg_triples, emb_train_triples, pos_sample=False)
        X_neg, y_neg = dh.convertListToData_Relation(neg_triples, emb_train_triples, pos_sample=False)
        X_pos, y_pos = dh.convertListToData_Relation(all_triples.tolist(), emb_train_triples, pos_sample=True)

        for i in range(emb_train_triples.num_relations):
            emb_dis_score_pos = emb.getScoreForTripleList(X_pos[i], emb_train_triples, emb_model)
            emb_dis_score_neg = emb.getScoreForTripleList(X_neg[i], emb_train_triples, emb_model)
            emb_dis_score_some_neg = emb.getScoreForTripleList(X_some_neg[i], emb_train_triples, emb_model)

            c = open(f'{path}/{sett.NAME_OF_RUN}_emb_dis_rel_{emb_train_triples.relation_id_to_label[i]}.csv', "w")
            writer = csv.writer(c)
            data = ['count','pos','some_neg','neg']
            writer.writerow(data)
            count = 0
            for i in range(min(len(emb_dis_score_pos),len(emb_dis_score_some_neg),len(emb_dis_score_neg))):
                data = [i]
                data.append(emb_dis_score_pos[i])
                
                data.append(emb_dis_score_some_neg[i])
                data.append(emb_dis_score_neg[i])
                writer.writerow(data)
            c.close()

    
