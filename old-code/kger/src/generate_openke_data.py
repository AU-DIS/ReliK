import sys, getopt, os, codecs
import load

def dump_map(map,output_file):
    if output_file:
        #output = open(output_file, 'w')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as output:
            output.write(str(len(map)))
            for k in map.keys():
                output.write('\n' + k + '\t' + str(map[k]))
            output.close()
    else:
        print("ERROR: output file not provided")
        sys.exit(2)


if __name__ == '__main__':
    input_file = None
    output_folder = None
    short_params = "i:o:"
    long_params = ["inputdata=","outputfolder="]
    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_params, long_params)
    except getopt.error as err:
        # Output error, and return with an error code
        print("generate_openke_data.py -i <input_dataset> -o <output_folder>")
        #print (str(err))
        sys.exit(2)

    for arg, value in arguments:
        if arg in ("-i", "--inputdata"):
            input_file = value
        elif arg in ("-o", "--outputfolder"):
            output_folder = value

    entities = set()
    relations = set()
    facts = 0
    if input_file:
        raw_data = codecs.open(input_file, 'r', encoding='utf-8', errors='ignore')
        line = raw_data.readline()
        while line:
            facts += 1
            tokens = line.split('\t')
            subject = tokens[0].strip()
            predicate = tokens[1].strip()
            object = tokens[2].strip()[0:-1]
            entities.add(subject)
            entities.add(object)
            relations.add(predicate)
            line = raw_data.readline()
        raw_data.close()
    else:
        print("ERROR: input file not provided")
        sys.exit(2)

    entity2id = {}
    sorted_entities = sorted(list(entities))#.sort()
    id = 0
    for e in sorted_entities:
        entity2id[e] = id
        id += 1

    relation2id = {}
    sorted_relations = sorted(list(relations))#.sort()
    id = 0
    for r in sorted_relations:
        relation2id[r] = id
        id += 1

    dump_map(entity2id, output_folder + os.path.sep + 'entity2id.txt')
    dump_map(relation2id, output_folder + os.path.sep + 'relation2id.txt')

    output_file = output_folder + os.path.sep + 'train2id.txt'
    if output_file:
        output = open(output_file, 'w')
        output.write(str(facts))
        raw_data = codecs.open(input_file, 'r', encoding='utf-8', errors='ignore')
        line = raw_data.readline()
        while line:
            tokens = line.split('\t')
            subjectid = entity2id[tokens[0].strip()]
            predicateid = relation2id[tokens[1].strip()]
            objectid = entity2id[tokens[2].strip()[0:-1]]
            output.write('\n' + str(subjectid) + ' ' + str(objectid) + ' ' + str(predicateid))
            line = raw_data.readline()
        raw_data.close()
        output.close()
    else:
        print("ERROR: output file not provided")
        sys.exit(2)
