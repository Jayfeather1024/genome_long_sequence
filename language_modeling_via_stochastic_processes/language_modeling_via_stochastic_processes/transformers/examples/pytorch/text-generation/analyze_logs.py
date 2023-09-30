import re, sys, os


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print (f'Usage: python {sys.argv[0]} <input-log>')
        sys.exit(1)


    settings = ['GT', 'BRIDGE', 'RANDOM']
    sections = ['ABSTRACT', 'HISTORY', 'GEOGRAPHY', 'DEMOGRAPHICS']
    all_results = {}

    input_log = sys.argv[1]
    with open(input_log) as fin:
        for line in fin:
            for setting in settings:
                for section in sections:
                    #if 'length recent' in line and setting in line:
                    #    import pdb; pdb.set_trace()
                    m = re.match(fr'.*{setting}\/\[ {section} \] length recent ([\.\d]+).*?', line)
                    if m:
                        #import pdb; pdb.set_trace()
                        if setting not in all_results:
                            all_results[setting] = {}
                        all_results[setting][section] = float(m.group(1))




    print (all_results)

    test_stats = {'ABSTRACT': 73.5, 'HISTORY': 180.2, 'GEOGRAPHY': 85.2, 'DEMOGRAPHICS': 332.5}
    for setting in settings:
        results = all_results[setting]
        total_percent_diff = 0
        for section in sections:
            diff = abs(test_stats[section] - results[section]) / test_stats[section]
            total_percent_diff += diff
        avg_percent_diff = total_percent_diff / len(sections) * 100
        print (setting, avg_percent_diff)


    input_log = sys.argv[1]
    with open(input_log) as fin:
        for line in fin:
            for setting in settings:
                for section in sections:
                    #if 'length recent' in line and setting in line:
                    #    import pdb; pdb.set_trace()
                    m = re.match(fr'.*{setting}\/\[ {section} \] redundant recent ([\.\d]+).*?', line)
                    if m:
                        #import pdb; pdb.set_trace()
                        if setting not in all_results:
                            all_results[setting] = {}
                        all_results[setting][section] = float(m.group(1))





    for setting in settings:
        results = all_results[setting]
        total_percent_diff = 0
        for section in sections:
            diff = results[section]
            total_percent_diff += diff
        avg_percent_diff = total_percent_diff / len(sections) * 100
        print (setting, avg_percent_diff)
