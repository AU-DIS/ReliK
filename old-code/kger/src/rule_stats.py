import sys, getopt, os, codecs

if __name__ == '__main__':
    rule_file = None
    short_params = "r:"
    long_params = ["rulefile="]
    try:
        arguments, values = getopt.getopt(sys.argv[1:], short_params, long_params)
    except getopt.error as err:
        # Output error, and return with an error code
        print("rule_stats.py -r <rule_file>")
        #print (str(err))
        sys.exit(2)

    for arg, value in arguments:
        if arg in ("-r", "--rulefile"):
            rule_file = value

    number_of_rules = 0
    rule_body_tokens_min = sys.maxsize
    rule_body_tokens_max = 0
    rule_body_tokens_sum = 0
    #process input rule_file
    if rule_file:
        rules = codecs.open(rule_file, 'r', encoding='utf-8', errors='ignore')
        line = rules.readline() #skip heading
        line = rules.readline()
        #print(line.split("\t")[0].split("=>")[0].split())
        while line:
            rule = line.split("\t")[0]
            rule_tokens = line.split("=>")
            body = rule_tokens[0]
            head = rule_tokens[1]
            body_tokens = len(body.split())

            rule_body_tokens_sum += body_tokens
            number_of_rules += 1
            if body_tokens < rule_body_tokens_min:
                rule_body_tokens_min = body_tokens
            if body_tokens > rule_body_tokens_max:
                rule_body_tokens_max = body_tokens

            line = rules.readline()
        rules.close()
    else:
        print("ERROR: rule file not provided")
        sys.exit(2)

    print()
    print(rule_file)
    print("---------------------------------")
    print("Number of rules: " + str(number_of_rules))
    print("Rule body tokens MIN: " + str(rule_body_tokens_min))
    print("Rule body tokens MAX: " + str(rule_body_tokens_max))
    #print("Rule body tokens SUM: " + str(rule_body_tokens_sum))
    print("Rule body tokens AVG: " + str(float(rule_body_tokens_sum)/number_of_rules))
    print("---------------------------------")
    print()
