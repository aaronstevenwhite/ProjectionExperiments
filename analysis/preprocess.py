import os, sys, copy, argparse
from collections import defaultdict

##################
## argument parser
##################

## instantiate parser
parser = argparse.ArgumentParser(description='Preprocess raw data from Ibex.')

parser.add_argument('--rootdir', 
                    type=str,
                    default='..')
parser.add_argument('--experiment', 
                    type=str,
                    choices=['frame', 'triad', 'likert'])

args = parser.parse_args()

######################
## preprocess raw data
######################

## set directories
data_dir = os.path.join(args.rootdir, 'data', args.experiment)
materials_dir = os.path.join(args.rootdir, 'materials', args.experiment)

## load data
raw_data_path = os.path.join(data_dir, args.experiment+'.ibex')
raw_data = [line.strip().split(',') for line in open(raw_data_path) if line[0] != '#'] 

## load configuration data (if applicable)
if args.experiment != 'likert':
    conf_path = os.path.join(materials_dir, args.experiment+'.conf')
    conf = [line.strip().split() for line in open(conf_path)]

## create header
if args.experiment == 'triad':
    conf_head = ['verb0', 'verb1', 'verb2', 'responseindex']  
elif args.experiment == 'likert':
    conf_head = ['verb0', 'verb1']  
elif args.experiment == 'frame':
    conf_head = ['verb', 'frame']

header = ['subj', 'item', 'trial'] + conf_head + ['response', 'rt']

## set counters (used to filter participants who did task twice)
subj_counts = defaultdict(int)
item_counts = defaultdict(int)

total_per_item = {'triad' : 3,
                  'frame' : 3,
                  'likert' : 4}

total_per_subj = {'triad' : 203,
                  'frame' : 102,
                  'likert' : 124}

## construct output filename and process
out_fname = os.path.join(data_dir, args.experiment+'.preprocessed')

with open(out_fname, 'w') as data:
    data.write(','.join(header)+'\n')

    for line in raw_data:

        if line[5] not in ['p', 'practice'] and line[2] != 'Form':

            if len(line) == 8 and args.experiment == 'likert':
                verb0, verb1 = line[7].split(' | ')
                item_conf = [verb0, verb1]

            if len(line) == 11:
                _, subj, _, itemnum, _, _, trial, _, response, _, rt  = line

                subj_counts[subj] += 1
                item_counts[itemnum] += 1

                if args.experiment != 'likert':
                    item_conf = copy.copy(conf[int(itemnum)-7])

                if args.experiment == 'triad':
                    response_index = str(item_conf.index(response))
                    item_conf += [response_index]

                if item_counts[itemnum] <= total_per_item[args.experiment] and subj_counts[subj] <= total_per_subj[args.experiment]:
                    datum = [subj, itemnum, trial] + item_conf + [response, rt]
                    data.write(','.join(datum)+'\n')

