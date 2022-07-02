import sys, getopt, os, codecs, copy
import load

kg = None
reverse_kg = None
rules = None

def get_facts(head,predicate,tail,negative):
    facts = []
    if head[0] != '?' and tail[0] != '?': #both head and tail have been already instantiated
        if not negative:
            if predicate in kg.keys() and head in kg[predicate].keys() and tail in kg[predicate][head]:
                facts.append([head,predicate,tail])
        else:
            if predicate not in kg.keys() or head not in kg[predicate].keys() or tail not in kg[predicate][head]:
                facts.append([head,predicate,tail])
    elif head[0] != '?': #only head has been already instantiated
        if predicate in kg.keys() and head in kg[predicate].keys():
            for t in kg[predicate][head]:
                facts.append([head,predicate,t])
    elif tail[0] != '?': #only tail has been already instantiated
        if predicate in reverse_kg.keys() and tail in reverse_kg[predicate].keys():
            for h in reverse_kg[predicate][tail]:
                facts.append([h,predicate,tail])
    else: #neither head nor tail have been already instantiated
        if predicate in kg.keys():
            for h in kg[predicate].keys():
                for t in kg[predicate][h]:
                    facts.append([h,predicate,t])
    return facts


def rule_support(flat_rule,negative):
    #print("Flat rule: " + str(flat_rule))
    var = {}
    support = []
    current_instantiation = []
    rule_support_rec(flat_rule,var,support,current_instantiation,negative)
    return support


def rule_support_rec(remaining_atoms,var,support,current_instantiation,negative):
    if not remaining_atoms:
        support.append(copy.deepcopy(current_instantiation))
        #support.append(current_instantiation)
    else:
        a = remaining_atoms[0]
        head = a[0]
        predicate = a[1]
        tail = a[2]
        head_i = head if head not in var.keys() else var[head]
        tail_i = tail if tail not in var.keys() else var[tail]

        a_kg_facts = get_facts(head_i,predicate,tail_i,False) if len(remaining_atoms) > 1 else get_facts(head_i,predicate,tail_i,negative)

        for fact in a_kg_facts:
            current_instantiation.append(fact)
            #print(current_instantiation)
            if head_i == head: #head not instantiated by previous recursive calls
                fact_head = fact[0]
                var[head] = fact_head
            if tail_i == tail: #tail not instantiated by previous recursive calls
                fact_tail = fact[2]
                var[tail] = fact_tail
            rule_support_rec(remaining_atoms[1:],var,support,current_instantiation,negative)

            #clean-up local instantiations and variable bindings
            current_instantiation.pop()
            if head_i == head:
                del var[head]
            if tail_i == tail:
                del var[tail]


def debug(p_ea, p_ab, p_eb):

    print()
    #rule = rules[0]
    #rule = rules[-1]
    rule = rules[134]
    #rule = rules[36]

    #flat_rule = rule[0:1] + rule[1]
    flat_rule = rule[0] + rule[-1:]
    #support = rule_support(flat_rule,False)
    support = rule_support(flat_rule,True)
    print("Support (size " + str(len(support)) + "):")
    for x in support:
        print(x)

    s = []
    for e in kg[p_ea].keys():
        for a in kg[p_ea][e]:
            if a in kg[p_ab].keys():
                for b in kg[p_ab][a]:
                    if e in kg[p_eb].keys() and b in kg[p_eb][e]:
                        x = [[a, p_ab, b],[e, p_eb, b],[e, p_ea, a]]
                        s.append(x)
    print("\nDEBUG (size " + str(len(s)) + "):")
    for x in s:
        print(x)
    sys.exit(-1)


def parse_rule(example):
    #print(example)
    entities = set()
    relations = set()
    example_string = ''

    for atom in example[0:-1]:
        entities.add(atom[0])
        entities.add(atom[2])
        relations.add(atom[1])
        example_string += atom[0] + ' ' + atom[1] + ' ' + atom[2] + ' '

    example_string += '=> '
    head = example[-1]
    entities.add(head[0])
    entities.add(head[2])
    relations.add(head[1])
    example_string += head[0] + ' ' + head[1] + ' ' + head[2]

    entities_string = ' '.join(list(entities))
    relations_string = ' '.join(list(relations))

    return (example_string, entities_string, relations_string)


if __name__ == '__main__':
    rule_file = None
    kg_file = None
    output_file = None
    negative_support = False
    short_params = "k:r:o:n"
    long_params = ["kgfile=","rulefile=","outputfile=","negativesupport"]
    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_params, long_params)
    except getopt.error as err:
        # Output error, and return with an error code
        print("rule_support.py -k <kg_file> -r <rule_file> -o <output_file> -n")
        #print (str(err))
        sys.exit(2)

    for arg, value in arguments:
        if arg in ("-k", "--kgfile"):
            kg_file = value
        elif arg in ("-r", "--rulefile"):
            rule_file = value
        elif arg in ("-o", "--outputfile"):
            output_file = value
        elif arg in ("-n", "--negativesupport"):
            negative_support = True

    (kg, reverse_kg) = load.load_kg(kg_file)
    rules = load.load_rules(rule_file)

    print('\n-------------------------------------')
    kg_facts = 0
    for p in kg.keys():
        for s in kg[p].keys():
            kg_facts += len(kg[p][s])
    print("#facts in KG: " + str(kg_facts))
    reverse_kg_facts = 0
    for p in reverse_kg.keys():
        for s in reverse_kg[p].keys():
            reverse_kg_facts += len(reverse_kg[p][s])
    print("#facts in reverse-KG (sanity check): " + str(reverse_kg_facts))
    print("#rules: " + str(len(rules)))
    print('-------------------------------------\n')

    #debug('<isMarriedTo>','<isLeaderOf>','<isLeaderOf>')
    #debug('<influences>','<influences>','<influences>')
    #debug('<isLocatedIn>', '<hasOfficialLanguage>', '<hasOfficialLanguage>')

    #write output
    tot_examples = 0
    if output_file:
        output = open(output_file, 'w')
        s = 'NEGATIVE_' if negative_support else "POSITIVE_"
        output.write('RULE' + '\t' + s + 'EXAMPLE' + '\t' + 'ENTITIES' + '\t' + 'RELATIONS')
        for rule in rules:
            #flat_rule = rule[0:1] + rule[1]
            flat_rule = rule[0] + rule[-1:]
            (rule_string, _, _) = parse_rule(flat_rule)
            support = rule_support(flat_rule,negative_support)
            tot_examples += len(support)
            for example in support:
                (example_string, entities, relations) = parse_rule(example)
                output.write('\n' + rule_string + '\t' + example_string + '\t' + entities + '\t' + relations)
        output.close()
        print(str(tot_examples) + ' examples successfully written!')
    else:
        print("ERROR: output file not provided")
        sys.exit(2)
