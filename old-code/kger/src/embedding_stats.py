import sys, getopt, os, codecs, math, time
import load


def range_query_setup(vectors):
    inverted_coord_index = [{} for j in range(0,len(vectors[0]))]
    sorted_coord = [[] for j in range(0,len(vectors[0]))]
    for j in range(0,len(vectors[0])):
        j_coord = [vectors[i][j] for i in range(0,len(vectors))]
        sorted_coord[j] = sorted(list(set(j_coord)))
        for i in range(0,len(vectors)):
            v = vectors[i][j]
            if v not in inverted_coord_index[j]:
                inverted_coord_index[j][v] = set()
            inverted_coord_index[j][v].add(i)

    return (inverted_coord_index,sorted_coord)

def range_query_setup_incremental(vectors, valid_vectors, j):
    inverted_coord_index = {}
    sorted_coord = []

    j_coord = [vectors[i][j] for i in valid_vectors]
    sorted_coord = sorted(list(set(j_coord)))

    for i in valid_vectors:
        v = vectors[i][j]
        if v not in inverted_coord_index:
            inverted_coord_index[v] = set()
        inverted_coord_index[v].add(i)

    return (inverted_coord_index,sorted_coord)

"""
def range_query(range_query_min_coord,range_query_max_coord,inverted_coord_index,sorted_coord):
    output = set()
    for j in range(0,len(range_query_min_coord)):
        output_j = set()
        min = range_query_min_coord[j]
        max = range_query_max_coord[j]
        i_min = search_min(min,sorted_coord[j])
        i_max = search_max(max,sorted_coord[j])
        if i_min != -1 and i_max != -1:
            for i in range(i_min,i_max+1):
                v = sorted_coord[j][i]
                v_objects = inverted_coord_index[j][v]
                output_j = output_j.union(v_objects)
        output = output.union(output_j) if j == 0 else output.intersection(output_j)
    return output
"""

def range_query(range_query_min_coord,range_query_max_coord,inverted_coord_index,sorted_coord):
    new_coord = [(j,range_query_max_coord[j]-range_query_min_coord[j]) for j in range(0,len(range_query_min_coord))]
    new_coord_sorted = sorted(new_coord, key=lambda tup: tup[1])
    output = set()
    #for j in range(0,len(range_query_min_coord)):
    count = 0
    for (j,_) in new_coord_sorted:
        output_j = []
        min = range_query_min_coord[j]
        max = range_query_max_coord[j]
        i_min = search_min(min,sorted_coord[j])
        i_max = search_max(max,sorted_coord[j])
        if i_min != -1 and i_max != -1:
            for i in range(i_min,i_max+1):
                v = sorted_coord[j][i]
                v_objects = inverted_coord_index[j][v]
                if count == 0:
                    output_j += v_objects
                else:
                    for o in v_objects:
                        if o in output:
                            output_j.append(o)
        output = set(output_j)
        if len(output) == 0:
            return output
        count += 1
    return output

def range_query_incremental(vectors,range_query_min_coord,range_query_max_coord,inverted_coord_index,sorted_coord,start_j):
    filtered_coord = set(range(0,len(range_query_min_coord)))
    filtered_coord.remove(start_j)
    filtered_coord_aug = [(j,range_query_max_coord[j]-range_query_min_coord[j]) for j in filtered_coord]
    filtered_coord_aug_sorted = sorted(filtered_coord_aug, key=lambda tup: tup[1])
    new_coord = [(start_j,-1)] + filtered_coord_aug_sorted
    output = set()
    #for j in range(0,len(range_query_min_coord)):
    count = 0
    for (j,_) in new_coord:
        if count > 0:
            (inverted_coord_index,sorted_coord) = range_query_setup_incremental(vectors,output,j)
        output_j = []
        min = range_query_min_coord[j]
        max = range_query_max_coord[j]
        i_min = search_min(min,sorted_coord)
        i_max = search_max(max,sorted_coord)
        if i_min != -1 and i_max != -1:
            for i in range(i_min,i_max+1):
                v = sorted_coord[i]
                v_objects = inverted_coord_index[v]
                if count == 0:
                    output_j += v_objects
                else:
                    for o in v_objects:
                        if o in output:
                            output_j.append(o)
        output = set(output_j)
        if len(output) == 0:
            return output
        count += 1
    return output

def range_query_naive(vectors,range_query_min_coord,range_query_max_coord):
    new_coord = [(j,range_query_max_coord[j]-range_query_min_coord[j]) for j in range(0,len(range_query_min_coord))]
    new_coord_sorted = sorted(new_coord, key=lambda tup: tup[1])
    output = range(0,len(vectors))
    for (j,_) in new_coord_sorted:
        new_output = []
        for i in output:
            if vectors[i][j] >= range_query_min_coord[j] and vectors[i][j] <= range_query_max_coord[j]:
                new_output.append(i)
        output = new_output
        if len(output) == 0:
            return set()
    return set(output)

#return the index in of the smallest value in <values> that is >= <key> (return -1 is <key> is > of all values in <values>)
def search_min(key,values):
    #return linear_search_min(key,values)
    return binary_search_min(key,values)

#return the index in of the largest value in <values> that is <= <key> (return -1 is <key> is < of all values in <values>)
def search_max(key,values):
    #return linear_search_max(key,values)
    return binary_search_max(key,values)

def linear_search_min(key,values):
    if len(values) == 0 or key > values[-1]:
        return -1
    for i in range(0,len(values)):
        if values[i] >= key:
            return i
    return -1

def linear_search_max(key,values):
    if len(values) == 0 or key < values[0]:
        return -1
    for i in reversed(range(0,len(values))):
        if values[i] <= key:
            return i
    return -1

def binary_search_min(key,values):
    if len(values) == 0 or key > values[-1]:
        return -1
    l = 0
    r = len(values)-1
    while r >= l:
        m = int((l+r)/2)
        if key == values[m]:
            return m
        if key < values[m]:
            r = m-1
        else:
            l = m+1
    return m if values[m] > key else m+1

def binary_search_max(key,values):
    if len(values) == 0 or key < values[0]:
        return -1
    l = 0
    r = len(values)-1
    while r >= l:
        m = int((l+r)/2)
        if key == values[m]:
            return m
        if key < values[m]:
            r = m-1
        else:
            l = m+1
    return m if values[m] < key else m-1

def get_first_order_moment(vectors):
    moments = [float(vectors[0][j]) for j in range(0,len(vectors[0]))]
    for i in range(1,len(vectors)):
        for j in range(0,len(vectors[i])):
            moments[j] += vectors[i][j]
    for j in range(0,len(moments)):
        moments[j] /= len(vectors)
    return moments

def get_second_order_moment(vectors):
    moments = [float(vectors[0][j]*vectors[0][j]) for j in range(0,len(vectors[0]))]
    for i in range(1,len(vectors)):
        for j in range(0,len(vectors[i])):
            moments[j] += vectors[i][j]*vectors[i][j]
    for j in range(0,len(moments)):
        moments[j] /= len(vectors)
    return moments

#fast avg squared Euclidean distance between all pairs of vectors in the two given sets
def fast_avg_euclidean_dist(vectors1,vectors2):
    first_order_moment1 = get_first_order_moment(vectors1)
    first_order_moment2 = get_first_order_moment(vectors2)
    second_order_moment1 = get_second_order_moment(vectors1)
    second_order_moment2 = get_second_order_moment(vectors2)

    dist = 0.0
    for j in range(0,len(first_order_moment1)):
        dist += second_order_moment1[j] - 2*first_order_moment1[j]*first_order_moment2[j] + second_order_moment2[j]
    return dist

def fast_avg_euclidean_dist_momentsgiven(first_order_moment1,first_order_moment2,second_order_moment1,second_order_moment2):
    dist = 0.0
    for j in range(0,len(first_order_moment1)):
        dist += second_order_moment1[j] - 2*first_order_moment1[j]*first_order_moment2[j] + second_order_moment2[j]
    return dist

def minmax_coordinates(embeddings):
    min_coordinates = [embeddings[0][i] for i in range(0,len(embeddings[0]))]
    max_coordinates = [embeddings[0][i] for i in range(0,len(embeddings[0]))]
    for e in embeddings:
        for i in range(0,len(e)):
            if e[i] < min_coordinates[i]:
                min_coordinates[i] = e[i]
            if e[i] > max_coordinates[i]:
                max_coordinates[i] = e[i]
    return (min_coordinates,max_coordinates)

#squared Euclidean distance
def euclidean_dist(v1,v2):
    d = 0.0
    for i in range(0,len(v1)):
        d += (v1[i]-v2[i])*(v1[i]-v2[i])
    return d

#area of the minimum bounding hyper-rectangle
def mbr_area(min_coordinates,max_coordinates):
    area = 1.0
    for i in range(0,len(min_coordinates)):
        area *= (max_coordinates[i] - min_coordinates[i])
    return area

#length of the diagonal of the minimum bounding hyper-rectangle
def mbr_diagonal(min_coordinates,max_coordinates):
    return euclidean_dist(min_coordinates,max_coordinates)

def min_mbr_edge(min_coordinates,max_coordinates):
    new_coord = [(j,max_coordinates[j]-min_coordinates[j]) for j in range(0,len(min_coordinates))]
    new_coord_sorted = sorted(new_coord, key=lambda tup: tup[1])
    (j,_) = new_coord_sorted[0]
    return j

def centroid(embeddings):
    c = [embeddings[0][j] for j in range(0,len(embeddings[0]))]
    for i in range(1,len(embeddings)):
        for j in range(0,len(embeddings[i])):
            c[j] += embeddings[i][j]
    for j in range(0,len(embeddings[0])):
        c[j] /= len(embeddings)
    return c

"""
def pairwise_dist(src_ids,target_ids,src_embeddings,target_embeddings):
    dist = {}
    for x_id in src_ids:
        x_embedding = src_embeddings[x_id]
        for y_id in target_ids:
            y_embedding = target_embeddings[y_id]
            d = get_dist(x_embedding,y_embedding)
            dist[(x_id,y_id)] = d
    return dist

def avg_dist(src_ids,target_ids,pairwise_dist):
    dist_sum = 0.0
    count = 0
    for i in src_ids:
        dist_sum_i = 0.0
        count_i = 0
        for j in target_ids:
            dist_sum_i += pairwise_dist[(i,j)]
            count_i += 1
        dist_avg_i = 0.0 if count_i == 0 else dist_sum_i/count_i
        dist_sum += dist_avg_i
        count += 1
    dist_avg = 0.0 if count == 0 else dist_sum/count
    return dist_avg
"""

def reduce_dim(embeddings):
    epsilon = 0.0001
    dims_tobefiltered = []
    for j in range(0,len(embeddings[0])):
        vj = embeddings[0][j]
        all_equal_vj = True
        i = 1
        while i < len(embeddings) and all_equal_vj:
            all_equal_vj = abs(embeddings[i][j] - vj) <= epsilon
            i += 1
        if all_equal_vj:
            dims_tobefiltered += vj
    return dims_tobefiltered

def get_entitypairs2rulefacts(rule2example,entity2id,rule2id):
    entitypairs2rulefacts = {}
    for t in rule2example.values():
        for ex in t:
            example = ex[0]
            facts = flattenize(example,entity2id,rule2id)
            key = []
            for f in facts:
                e1 = f[0]
                e2 = f[2]
                e12 = (e1,e2)
                key.append(e12)
            key = frozenset(key)
            if key not in entitypairs2rulefacts.keys():
                entitypairs2rulefacts[key] = set()
            entitypairs2rulefacts[key].add(frozenset(facts))
    return entitypairs2rulefacts

def get_entitypair2allrelations(entitypairs,kg,entity2id,relation2id):
    entitypair2allrelations = {}
    for (e1,e2) in entitypairs:
        epair = (e1,e2)
        if e1 in kg.keys() and e2 in kg[e1]:
            if epair not in entitypair2allrelations.keys():
                entitypair2allrelations[epair] = set()
            for rel in kg[e1][e2]:
                entitypair2allrelations[epair].add(rel)
    return entitypair2allrelations

def get_entitypair2relpowset(entitypair2allrelations):
    entitypair2relpowset = {}
    for p in entitypair2allrelations.keys():
        powset = allsubsets(list(entitypair2allrelations[p]))
        entitypair2relpowset[p] = powset
    return entitypair2relpowset

def get_allentitypairsfacts(entitypair2relpowset):
    all_facts = set()
    remaining_pairs = list(entitypair2relpowset.keys())
    current_facts = set()
    get_allentitypairsfacts_rec(entitypair2relpowset,remaining_pairs,current_facts,all_facts)
    return all_facts

def get_allentitypairsfacts_rec(entitypair2relpowset,remaining_pairs,current_facts,all_facts):
    if len(remaining_pairs) == 0:
        all_facts.add(frozenset(current_facts))
    else:
        p = remaining_pairs[0]
        rel_powset = entitypair2relpowset[p]
        for rset in rel_powset:
            if rset:
                new_facts = []
                for r in rset:
                    f = (p[0], r, p[1])
                    new_facts.append(f)
                for f in new_facts:
                    current_facts.add(f)
                get_allentitypairsfacts_rec(entitypair2relpowset,remaining_pairs[1:],current_facts,all_facts)
                for f in new_facts:
                    current_facts.remove(f)

def allsubsets(s):
    all_subsets = []
    pow_set_size = (int) (math.pow(2, len(s)))
    # Run from counter 000..0 to 111..1
    for counter in range(0, pow_set_size):
        current_s = set()
        for j in range(0, len(s)):
            # Check if jth bit in the
            # counter is set If set then
            # print jth element from set
            if((counter & (1 << j)) > 0):
                current_s.add(s[j])
        all_subsets.append(current_s)
    return all_subsets

def flattenize(rule,entity2id,relation2id):
    flat_rule_set = set()
    flat_rule = rule[0] + rule[-1:]
    for f in flat_rule:
        f_tuple = (entity2id[f[0]], relation2id[f[1]], entity2id[f[2]])
        flat_rule_set.add(f_tuple)
    return flat_rule_set

def rearrange_kg(kg):
    rearranged_kg = {}
    for rel in kg.keys():
        for subj in kg[rel]:
            if subj not in rearranged_kg.keys():
                rearranged_kg[subj] = {}
            #print(rel)
            #print(subj)
            #print(kg[rel][subj])
            for obj in kg[rel][subj]:
                obj = int(obj)
                if obj not in rearranged_kg[subj].keys():
                    rearranged_kg[subj][obj] = set()
                rearranged_kg[subj][obj].add(rel)
    return rearranged_kg

def select_dictionary_items(keys,dict):
    new_dict = {}
    for k in keys:
        new_dict[k] = dict[k]
    return new_dict

def exampletostring(example):
    s = ''
    for atom in example[0]:
        s += atom[0] + ' ' + atom[1] + ' ' + atom[2]
    s += ' => '
    s += example[1][0] + ' ' + example[1][1] + ' ' + example[1][2]
    return s

def exampletoentitypairset(example,entity2id):
    pairset = set()
    flat_example = example[0] + example[-1:]
    for atom in flat_example:
        epair = (entity2id[atom[0]], entity2id[atom[2]])
        pairset.add(epair)
    return pairset

def get_refermbr_diag(example,nrelexample,entitypairs2allnonrulefacts,entity2id,ent_embeddings,rel_embeddings):
    pairset = frozenset(exampletoentitypairset(example,entity2id))
    if pairset not in entitypairs2allnonrulefacts.keys():
        return -1.0

    diag_sum = 0.0
    count = 0
    allnonrulefactsets = entitypairs2allnonrulefacts[pairset]

    ex_entities = set()
    for p in pairset:
        ex_entities.add(p[0])
        ex_entities.add(p[1])
    ex_ent_embeddings = [ent_embeddings[e] for e in ex_entities]

    for factset in allnonrulefactsets:
        factset_rels = set()
        for f in factset:
            factset_rels.add(f[1])
        if len(factset_rels) == nrelexample:
            factset_rel_embeddings = [rel_embeddings[r] for r in factset_rels]
            (mbr_min_coord,mbr_max_coord) = minmax_coordinates(ex_ent_embeddings+factset_rel_embeddings)
            d = mbr_diagonal(mbr_min_coord,mbr_max_coord)
            diag_sum += d
            count += 1
    diag = diag_sum/count if count > 0 else -1
    return diag

def debug():
    #vectors = [[1,2,3],[3,2,1],[2,2,2]]
    #print(get_first_order_moment(vectors))
    #print(get_second_order_moment(vectors))
    #print(fast_avg_euclidean_dist([[2,2,3],[1,3,4]],[[2,2,3],[1,3,4]]))
    print(str(search_min(8,[7,11,15,18,20])) + ' ' + str(search_max(8,[7,11,15,18,20])))
    print(str(search_min(15,[7,11,15,18,20])) + ' ' + str(search_max(15,[7,11,15,18,20])))
    print(str(search_min(19,[7,11,15,18,20])) + ' ' + str(search_max(19,[7,11,15,18,20])))
    print(str(search_min(25,[7,11,15,18,20])) + ' ' + str(search_max(25,[7,11,15,18,20])))
    print(str(search_min(5,[7,11,15,18,20])) + ' ' + str(search_max(5,[7,11,15,18,20])))

    vectors = [[1,3,5],[3,6,8],[-1,6,-5],[3,-3,0],[4,-1,0]]
    (inverted_coord_index,sorted_coord) = range_query_setup(vectors)
    #print(inverted_coord_index)
    #print(sorted_coord)
    range_query_min_coord = [1,-5,0]
    range_query_max_coord = [10,10,5]
    range_query_result = range_query(range_query_min_coord,range_query_max_coord,inverted_coord_index,sorted_coord)
    print(range_query_result)

    sys.exit(-1)

if __name__ == '__main__':
    #debug()
    #print(allsubsets(list([1,2,3])))
    #sys.exit(-1)

    rule_support_file = None
    embedding_file = None
    kg_folder = None
    output_file = None
    short_params = "r:e:k:o:"
    long_params = ["rulesupportfile=","embeddingfile=","kgfolder=","outputfile="]
    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_params, long_params)
    except getopt.error as err:
        # Output error, and return with an error code
        print("embedding_stats.py -r <rule_support_file> -e <embedding_file> -k <kg_folder> -o <output_file>")
        #print (str(err))
        sys.exit(2)

    for arg, value in arguments:
        if arg in ("-r", "--rulesupportfile"):
            rule_support_file = value
        elif arg in ("-e", "--embeddingfile"):
            embedding_file = value
        elif arg in ("-k", "--kgfolder"):
            kg_folder = value
        elif arg in ("-o", "--outputfile"):
            output_file = value

    (ent_embeddings,rel_embeddings) = load.load_embeddings(embedding_file)
    print('Embeddings successfully loaded!')
    (entity2id,relation2id,kg) = load.load_openke_dataset(kg_folder)
    rearranged_kg = rearrange_kg(kg)
    print('OpenKE dataset successfully loaded!')

    print('Computing (first-order and second-order) moments of all entities and relations')
    fo_moment_alle = get_first_order_moment(ent_embeddings)
    so_moment_alle = get_second_order_moment(ent_embeddings)
    fo_moment_allr = get_first_order_moment(rel_embeddings)
    so_moment_allr = get_second_order_moment(rel_embeddings)

    print('Computing MBRs')
    (alle_min_coord,alle_max_coord) = minmax_coordinates(ent_embeddings)
    (aller_min_coord,aller_max_coord) = minmax_coordinates(ent_embeddings+rel_embeddings)
    (allr_min_coord,allr_max_coord) = minmax_coordinates(rel_embeddings)
    alle_mbr_diag = mbr_diagonal(alle_min_coord,alle_max_coord)
    aller_mbr_diag = mbr_diagonal(aller_min_coord,aller_max_coord)
    allr_mbr_diag = mbr_diagonal(allr_min_coord,allr_max_coord)

    #print('Computing data structures for range queries')
    #(inverted_coord_index_e,sorted_coord_e) = range_query_setup(ent_embeddings)
    #(inverted_coord_index_er,sorted_coord_er) = range_query_setup(ent_embeddings+rel_embeddings)
    #(inverted_coord_index_r,sorted_coord_r) = range_query_setup(rel_embeddings)

    #print('Computing data structures for range queries (incremental query processing)')
    #min_alle_mbr_edge = min_mbr_edge(alle_min_coord,alle_max_coord)
    #min_allr_mbr_edge = min_mbr_edge(allr_min_coord,allr_max_coord)
    #(inverted_coord_index_e_incr,sorted_coord_e_incr) = range_query_setup_incremental(ent_embeddings,set(range(0,len(ent_embeddings))),min_alle_mbr_edge)
    #(inverted_coord_index_r_incr,sorted_coord_r_incr) = range_query_setup_incremental(rel_embeddings,set(range(0,len(rel_embeddings))),min_allr_mbr_edge)

    heading = 'RULE' + '\t' + 'POSITIVE_EXAMPLE' + '\t'\
    '#ENTITIES_EXAMPLE' + '\t' + '#RELATIONS_EXAMPLE' + '\t'\
    '#ENTITIES_E-MBR' + '\t' + '#ENTITIES_ER-MBR' + '\t'\
    '#RELATIONS_R-MBR' + '\t' + '#RELATIONS_ER-MBR' + '\t'\
    'AVG_DIST_E2E' + '\t' + 'AVG_DIST_E2ALLE' + '\t'\
    'AVG_DIST_E2R' + '\t' + 'AVG_DIST_E2ALLR' + '\t'\
    'AVG_DIST_R2R' + '\t' + 'AVG_DIST_R2ALLR' + '\t'\
    'E-MBR_DIAG' + '\t' + 'ALLE-MBR_DIAG' + '\t'\
    'ER-MBR_DIAG' + '\t' + 'REF-ER-MBR_DIAG' + '\t' + 'ALLER-MBR_DIAG' + '\t'\
    'R-MBR_DIAG' + '\t' + 'ALLR-MBR_DIAG'
    if rule_support_file and output_file:
        rule2example = load.load_rule_support(rule_support_file)
        entitypairs2rulefacts = get_entitypairs2rulefacts(rule2example,entity2id,relation2id)
        allruleentitypairs = set()
        for s in entitypairs2rulefacts.keys():
            for p in s:
                allruleentitypairs.add(p)
        entitypair2allrelations = get_entitypair2allrelations(allruleentitypairs,rearranged_kg,entity2id,relation2id)
        entitypair2relpowset = get_entitypair2relpowset(entitypair2allrelations)
        entitypairs2allnonrulefacts = {}
        for pset in entitypairs2rulefacts.keys():
            rulefacts = entitypairs2rulefacts[pset]
            allfacts = get_allentitypairsfacts(select_dictionary_items(pset,entitypair2relpowset))
            nonrulefacts = set()
            #print(rulefacts)
            #print(allfacts)
            #sys.exit(-1)
            for f in allfacts:
                if f not in rulefacts:
                    nonrulefacts.add(f)
            #sanity check
            for f in rulefacts:
                if f not in allfacts:
                    print('ERROR!!!')
                    print('Rule facts:')
                    for f in rulefacts:
                        print(f)
                    print('\nAll facts:')
                    for f in allfacts:
                        print(f)
                    sys.exit(-2)
            if len(nonrulefacts) > 0:
                entitypairs2allnonrulefacts[pset] = nonrulefacts

        #print(len(entitypairs2rulefacts))
        #print(len(entitypairs2allnonrulefacts))
        #sys.exit(-1)
        """
        first_key = next(iter(entitypairs2rulefacts.keys()))
        first_value = entitypairs2rulefacts[first_key]
        print(len(entitypairs2rulefacts.keys()))
        print(first_key)
        print(first_value)
        print(str(len(first_value)))

        entitypair2relation = select_dictionary_items(first_key,entitypairs2allrelations)
        print(entitypair2relation)
        allentitypairsfacts = get_allentitypairsfacts(entitypair2relation)
        print(allentitypairsfacts)


        counter = 0
        printed = False
        count_rulefactslessthanallfacts = 0
        for k in entitypairs2rulefacts.keys():
            n_rulefacts = len(entitypairs2rulefacts[k])
            n_allfacts = len(get_allentitypairsfacts(select_dictionary_items(k,entitypairs2allrelations)))
            if n_allfacts > n_rulefacts:
                count_rulefactslessthanallfacts += 1
            if len(k) > 2:
                v = entitypairs2rulefacts[k]
                entitypair2relation = select_dictionary_items(k,entitypairs2allrelations)
                length2 = 0
                lengthmore2 = 0
                for p in entitypair2relation.keys():
                    if len(entitypair2relation[p]) == 2:
                        length2 += 1
                    elif len(entitypair2relation[p]) > 2:
                        lengthmore2 += 1
                if length2 >= 2 and lengthmore2 >= 0:
                    counter += 1
                    if not printed:
                        print(k)
                        print(v)
                        print(str(len(v)))
                        print(entitypair2relation)
                        allentitypairsfacts = get_allentitypairsfacts(entitypair2relation)
                        print(allentitypairsfacts)
                        printed = True

        print(str(counter) + ' over a total of ' + str(len(entitypairs2rulefacts)))
        print('count_rulefactslessthanallfacts: ' + str(count_rulefactslessthanallfacts))
        sys.exit(-1)
        """

        count = 0
        count_ref_er_mbr_nonnegativediag = 0
        output = open(output_file, 'w')
        output.write(heading)
        print('Started processing rules')
        for rule in rule2example.keys():
            rule_stats_sum = []
            rule_stats_square_sum = []
            fo_moment_current_rule_e = []
            so_moment_current_rule_e = []
            fo_moment_current_rule_r = []
            so_moment_current_rule_r = []
            n_rule_entities = 0
            n_rule_relations = 0
            n_examples = 0
            for (example,entities,relations) in rule2example[rule]:
                current_ent_embeddings = [ent_embeddings[entity2id[e]] for e in entities]
                fo_moment_currente = get_first_order_moment(current_ent_embeddings)
                so_moment_currente = get_second_order_moment(current_ent_embeddings)
                current_rel_embeddings = [rel_embeddings[relation2id[r]] for r in relations]
                fo_moment_currentr = get_first_order_moment(current_rel_embeddings)
                so_moment_currentr = get_second_order_moment(current_rel_embeddings)

                e2e_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currente,fo_moment_currente,so_moment_currente,so_moment_currente)
                e2alle_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currente,fo_moment_alle,so_moment_currente,so_moment_alle)
                e2r_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currente,fo_moment_currentr,so_moment_currente,so_moment_currentr)
                e2allr_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currente,fo_moment_allr,so_moment_currente,so_moment_allr)
                r2r_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currentr,fo_moment_currentr,so_moment_currentr,so_moment_currentr)
                r2allr_avgdist = fast_avg_euclidean_dist_momentsgiven(fo_moment_currentr,fo_moment_allr,so_moment_currentr,so_moment_allr)

                (e_mbr_min_coord,e_mbr_max_coord) = minmax_coordinates(current_ent_embeddings)
                (er_mbr_min_coord,er_mbr_max_coord) = minmax_coordinates(current_ent_embeddings+current_rel_embeddings)
                (r_mbr_min_coord,r_mbr_max_coord) = minmax_coordinates(current_rel_embeddings)

                start = time.time()
                e_mbr_entities = range_query_naive(ent_embeddings,e_mbr_min_coord,e_mbr_max_coord)
                er_mbr_entities = range_query_naive(ent_embeddings,er_mbr_min_coord,er_mbr_max_coord)
                r_mbr_relations = range_query_naive(rel_embeddings,r_mbr_min_coord,r_mbr_max_coord)
                er_mbr_relations = range_query_naive(rel_embeddings,er_mbr_min_coord,er_mbr_max_coord)
                end = time.time()

                e_mbr_diag = mbr_diagonal(e_mbr_min_coord,e_mbr_max_coord)
                er_mbr_diag = mbr_diagonal(er_mbr_min_coord,er_mbr_max_coord)
                r_mbr_diag = mbr_diagonal(r_mbr_min_coord,r_mbr_max_coord)

                ref_er_mbr_diag = get_refermbr_diag(example,len(relations),entitypairs2allnonrulefacts,entity2id,ent_embeddings,rel_embeddings)
                if ref_er_mbr_diag > 0.0:
                    count_ref_er_mbr_nonnegativediag += 1

                output_stats = [len(entities),len(relations),len(e_mbr_entities),len(er_mbr_entities),len(r_mbr_relations),len(er_mbr_relations),e2e_avgdist,e2alle_avgdist,e2r_avgdist,e2allr_avgdist,r2r_avgdist,r2allr_avgdist,e_mbr_diag,alle_mbr_diag,er_mbr_diag,ref_er_mbr_diag,aller_mbr_diag,r_mbr_diag,allr_mbr_diag]


                if n_examples == 0:
                    rule_stats_sum = [s for s in output_stats]
                    rule_stats_square_sum = [s*s for s in output_stats]
                    fo_moment_current_rule_e = [x*len(current_ent_embeddings) for x in fo_moment_currente]
                    so_moment_current_rule_e = [x*len(current_ent_embeddings) for x in so_moment_currente]
                    fo_moment_current_rule_r = [x*len(current_rel_embeddings) for x in fo_moment_currentr]
                    so_moment_current_rule_r = [x*len(current_rel_embeddings) for x in so_moment_currentr]
                else:
                    for i in range(0,len(output_stats)):
                        rule_stats_sum[i] += output_stats[i]
                        rule_stats_square_sum[i] += output_stats[i]*output_stats[i]
                    for i in range(0,len(fo_moment_currente)):
                        fo_moment_current_rule_e[i] += fo_moment_currente[i]*len(current_ent_embeddings)
                    for i in range(0,len(so_moment_currente)):
                        so_moment_current_rule_e[i] += so_moment_currente[i]*len(current_ent_embeddings)
                    for i in range(0,len(fo_moment_currentr)):
                        fo_moment_current_rule_r[i] += fo_moment_currentr[i]*len(current_rel_embeddings)
                    for i in range(0,len(so_moment_currentr)):
                        so_moment_current_rule_r[i] += so_moment_currentr[i]*len(current_rel_embeddings)

                n_rule_entities += len(current_ent_embeddings)
                n_rule_relations += len(current_rel_embeddings)
                n_examples += 1

                output_line = rule + '\t' + exampletostring(example) + '\t' + '\t'.join([str(s) for s in output_stats])
                output.write('\n' + output_line)
                output.flush()

                count += 1
                #if ref_er_mbr_diag > 0.0:
                    #print(output_line)
                print('#processed examples: ' + str(count) + '---Range-query time: ' + str(int(round((end-start)*1000))) + 'ms')

            rule_stats_avg = [float(s)/n_examples for s in rule_stats_sum]
            rule_stats_stddev = [math.sqrt(max(0,float(rule_stats_square_sum[i])/n_examples - rule_stats_avg[i]*rule_stats_avg[i])) for i in range(0,len(rule_stats_square_sum))]
            #for i in range(0,len(rule_stats)):
            #    rule_stats[i] = float(rule_stats[i])/n_examples
            output_line_avg = rule + '\t' + 'RULE AVG STATS (over ' + str(n_examples) + ' examples)' + '\t' + '\t'.join([str(s) for s in rule_stats_avg])
            output_line_stddev = rule + '\t' + 'RULE STD-DEV STATS (over ' + str(n_examples) + ' examples)' + '\t' + '\t'.join([str(s) for s in rule_stats_stddev])
            output.write('\n' + output_line_avg)
            output.write('\n' + output_line_stddev)
            output.flush()

            """
            for i in range(0,len(fo_moment_current_rule_e)):
                fo_moment_current_rule_e[i] = float(fo_moment_current_rule_e[i])/n_rule_entities
            for i in range(0,len(so_moment_current_rule_e)):
                so_moment_current_rule_e[i] = float(so_moment_current_rule_e[i])/n_rule_entities
            for i in range(0,len(fo_moment_current_rule_r)):
                fo_moment_current_rule_r[i] = float(fo_moment_current_rule_r[i])/n_rule_relations
            for i in range(0,len(so_moment_current_rule_r)):
                so_moment_current_rule_r[i] = float(so_moment_current_rule_r[i])/n_rule_relations
            avg_interexample_dist_e = fast_avg_euclidean_dist_momentsgiven(fo_moment_current_rule_e,fo_moment_current_rule_e,so_moment_current_rule_e,so_moment_current_rule_e)
            avg_interexample_dist_er = fast_avg_euclidean_dist_momentsgiven(fo_moment_current_rule_e,fo_moment_current_rule_r,so_moment_current_rule_e,so_moment_current_rule_r)
            avg_interexample_dist_r = fast_avg_euclidean_dist_momentsgiven(fo_moment_current_rule_r,fo_moment_current_rule_r,so_moment_current_rule_r,so_moment_current_rule_r)
            output.write('\n' + rule + '\t' + 'AVG INTER-EXAMPLE DISTANCE BETWEEN ENTITIES' + '\t' + str(avg_interexample_dist_e))
            output.write('\n' + rule + '\t' + 'AVG INTER-EXAMPLE DISTANCE BETWEEN ENTITIES AND RELATIONS' + '\t' + str(avg_interexample_dist_er))
            output.write('\n' + rule + '\t' + 'AVG INTER-EXAMPLE DISTANCE BETWEEN RELATIONS' + '\t' + str(avg_interexample_dist_r))
            output.write('\n')
            output.flush()
            """
        output.close()
        print('count_ref_er_mbr_nonnegativediag: ' + str(count_ref_er_mbr_nonnegativediag))
    else:
        print("ERROR: rule-support file and/or output file not provided")
        sys.exit(2)
