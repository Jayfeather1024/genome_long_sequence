import re, sys, os, math
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

def get_scores(input_log, verbose=False):
    settings = ['GT', 'BRIDGE', 'RANDOM']
    settings = ['GT', 'BRIDGE']
    sections = ['ABSTRACT', 'HISTORY', 'GEOGRAPHY', 'DEMOGRAPHICS']
    sections = ['TITLE', 'METHOD', 'STEP']
    all_results = {}
    all_results_rr = {}
    all_results_ordering = {}

    mappings = {'GT': 'true', 'BRIDGE': 'bridge', 'RANDOM': 'random'}
    for setting in settings:
        all_results[setting] = {}
        all_results_rr[setting] = {}
        all_results_ordering[setting] = []
        results = all_results[setting]
        results_rr = all_results_rr[setting]
        results_ordering = all_results_ordering[setting]
        m = mappings[setting]
        #filename = f'{sys.argv[1]}{m}CLEmbs_samplenew120_metrics.csv'
        #filename = f'{sys.argv[1]}{m}CLEmbs_12_metrics.csv'
        #filename = f'{sys.argv[1]}{m}CLEmbs_120_metrics.csv'
        #filename = f'{sys.argv[1]}{m}CLEmbs_12_metrics.csv'
        #if not os.path.exists(filename):
        filename = f'{input_log}{m}CLEmbs_120_metrics.csv'
        #if 'large' in filename: #not os.path.exists(filename):
        #    #filename = f'{sys.argv[1]}{m}CLEmbs_samplenew37_metrics.csv'
        #    filename = f'{sys.argv[1]}{m}CLEmbs_sampleold37_metrics.csv'
        mm = {}
        rr = {}
        oo = {}
        ordering_id = None
        with open(filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if len(mm) == 0:
                    for id, e in enumerate(row):
                        if e == 'ordering':
                            ordering_id = id
                        for section in sections:
                            if e == f'[ {section} ] length':
                                mm[section] = id
                            if e == f'[ {section} ] present':
                                oo[section] = id
                            if e == f'[ {section} ] redundant':
                                rr[section] = id
                else:
                    results_ordering.append(1 if (row[ordering_id]=='True' and all([row[oo[section]]=='True' for section in sections])) else 0)
                    for section in sections:
                        if section not in results_rr:
                            results_rr[section] = []
                        #if section not in rr:
                        #    import pdb; pdb.set_trace()
                        #if rr[section] >= len(row):
                        #    import pdb; pdb.set_trace()
                        if section not in rr:
                            print (filename)
                            continue
                        if rr[section] >= len(row):
                            continue
                        results_rr[section].append(1 if (row[rr[section]]=='True') else 0)
                        if section not in results:
                            results[section] = []
                        if row[mm[section]] == '':
                            continue
                            #import pdb; pdb.set_trace()
                        results[section].append(float(row[mm[section]]))
        for section in sections:
            results[section] = np.array(results[section])
        for section in sections:
            results_rr[section] = np.array(results_rr[section])
        all_results_ordering[setting] = np.array(results_ordering)

    for setting in settings:
        results = all_results[setting]
        results_rr = all_results_rr[setting]
        if verbose:
            print (f'setting: {setting}')
        for section in sections:
            a = results[section]
            a_rr = results_rr[section]
            #import pdb; pdb.set_trace()
            if verbose:
                print (f'section: {section}, avg length: {a.mean()}, std: {np.std(a) / math.sqrt(a.shape[0])}, med length: {np.median(a)}, redundant: {np.mean(a_rr)}')

    #sns.kdeplot(all_results['GT']['ABSTRACT'], shade=1, color='red', label='GT')
    #sns.kdeplot(all_results['BRIDGE']['ABSTRACT'], shade=1, color='green', label='BRIDGE')
    #
    #plt.savefig('density_comparison.png')
    #
    #
    #for section in sections:
    #    test = scipy.stats.ks_2samp(all_results['GT'][section], all_results['BRIDGE'][section])
    #    print (f'GT vs BRIDGE {section}', test)
    #    test = scipy.stats.ks_2samp(all_results['GT'][section], all_results['RANDOM'][section])
    #    print (f'GT vs RANDOM {section}', test)
    return_dict = {}
    test_stats = {'ABSTRACT': 73.5, 'HISTORY': 180.2, 'GEOGRAPHY': 85.2, 'DEMOGRAPHICS': 332.5}
    test_stats = {'TITLE': 10.4, 'METHOD': 8.7, 'STEP': 480.2}
    for setting in settings:
        results = all_results[setting]
        results_rr = all_results_rr[setting]
        total_percent_diff = 0
        total_redundant = 0
        for section in sections:
            diff = abs(test_stats[section] - results[section].mean()) / test_stats[section]
            total_redundant += results_rr[section].mean()
            total_percent_diff += diff
        avg_percent_diff = total_percent_diff / len(sections) * 100
        avg_redundant = total_redundant / len(sections) * 100
        avg_ordering = all_results_ordering[setting].mean()
        #print (setting, avg_percent_diff, avg_redundant)
        return_dict[setting]=(avg_percent_diff, avg_redundant, avg_ordering)
    return return_dict
#    with open(input_log) as fin:
#        for line in fin:
#            for setting in settings:
#                for section in sections:
#                    #if 'length recent' in line and setting in line:
#                    #    import pdb; pdb.set_trace()
#                    m = re.match(fr'.*{setting}\/\[ {section} \] length recent ([\.\d]+).*?', line)
#                    if m:
#                        #import pdb; pdb.set_trace()
#                        if setting not in all_results:
#                            all_results[setting] = {}
#                        all_results[setting][section] = float(m.group(1))
#
#
#
#
#    print (all_results)
#
#    test_stats = {'ABSTRACT': 73.5, 'HISTORY': 180.2, 'GEOGRAPHY': 85.2, 'DEMOGRAPHICS': 332.5}
#    for setting in settings:
#        results = all_results[setting]
#        total_percent_diff = 0
#        for section in sections:
#            diff = abs(test_stats[section] - results[section]) / test_stats[section]
#            total_percent_diff += diff
#        avg_percent_diff = total_percent_diff / len(sections) * 100
#        print (setting, avg_percent_diff)
#
#
#    input_log = sys.argv[1]
#    with open(input_log) as fin:
#        for line in fin:
#            for setting in settings:
#                for section in sections:
#                    #if 'length recent' in line and setting in line:
#                    #    import pdb; pdb.set_trace()
#                    m = re.match(fr'.*{setting}\/\[ {section} \] redundant recent ([\.\d]+).*?', line)
#                    if m:
#                        #import pdb; pdb.set_trace()
#                        if setting not in all_results:
#                            all_results[setting] = {}
#                        all_results[setting][section] = float(m.group(1))
#
#
#
#
#
#    for setting in settings:
#        results = all_results[setting]
#        total_percent_diff = 0
#        for section in sections:
#            diff = results[section]
#            total_percent_diff += diff
#        avg_percent_diff = total_percent_diff / len(sections) * 100
#        print (setting, avg_percent_diff)
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print (f'Usage: python {sys.argv[0]} <input-log-base>')
        sys.exit(1)

    return_dict = get_scores(sys.argv[1], verbose=True)
    for setting in return_dict:
        avg_percent_diff, avg_redundant, avg_ordering = return_dict[setting]
        print (setting, avg_percent_diff, avg_redundant, avg_ordering)


