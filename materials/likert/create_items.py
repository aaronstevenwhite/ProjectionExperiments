import random
import sys
from string import Template

verbs_fname = '/home/aaronsteven/experiments/triad+frame/experiment/materials/triad/lists/verbs.list'
verbs = [line.strip() for line in open(verbs_fname)] + ['know']

verb_pairs = [[verb1, verb2] for i, verb1 in enumerate(verbs) for verb2 in verbs[i+1:]]

verb_pairs_partitioned = []
curr_partition = []

for verb_pair in verb_pairs:
    verb_pair_rev = [verb_pair[1], verb_pair[0]]

    if len(curr_partition) == 14:
        curr_partition = [verb_pair] + curr_partition
        verb_pairs_partitioned.append(curr_partition)
        curr_partition = []
    elif len(curr_partition) == 15:
        verb_pairs_partitioned.append(curr_partition)
        curr_partition = [verb_pair]
    else:
        curr_partition += [verb_pair]

    curr_partition += [verb_pair_rev]

verb_pairs_partitioned.append(curr_partition)

def create_controller(verb_pair, i):
    verb1, verb2 = verb_pair

    return '[["similarity",'+str(i+1)+'], "AcceptabilityJudgment", {s: \''+verb1+' | '+verb2+'\'}]'

test_controllers = [create_controller(verb_pair, i) for i, partition in enumerate(verb_pairs_partitioned) for verb_pair in partition]

def create_debrief_controller(size=15):
    alpha = [str(i) for i in range(10)] + ['A', 'B', 'C', 'D', 'E', 'F']
    code = ''.join(random.sample(alpha, size))
    return '[["debrief", 0], "Message", {html: "<p><center>Please enter the following code into Mechanical Turk.</center></p><p><center><b>'+code+'</b></center></p>"}]'

debrief_controllers = [create_debrief_controller() for i in range(60)]

conf_temp = lambda items: Template('''
var shuffleSequence = seq("consent", "setcounter", "intro", "practice", "begin", sepWith("sep", randomize("similarity")), "sr", "debrief");
var practiceItemTypes = ["practice"];

var manualSendResults = true;

var defaults = [
    "Separator", {
        transfer: 500,
        hideProgressBar: true,
        normalMessage: "+"
    },
    "Message", {
        hideProgressBar: true
    },
    "AcceptabilityJudgment", {
        q: 'How similar are these two verbs?',
        as: ["1", "2", "3", "4", "5", "6", "7"],
        presentAsScale: true,
        instructions: "Use number keys or click boxes to answer.",
        leftComment: "Very dissimilar", rightComment: "Very similar"
    },
    "Form", { 
        hideProgressBar: true,
        continueOnReturn: false   
    }

];

var items = [
	["consent", "Form", {
        html: { include: "consent.html" },
		validators: {age: function (s) { if (s.match(/^\d+$$/)) return true;
							else return "Bad value for age"; }}
    } ],

	["intro", "Message", {html: { include: "intro.html" }}],

	["practice", "AcceptabilityJudgment", {s: "crack | break"}],
	["practice", "AcceptabilityJudgment", {s: "eat | fail"}],
	["practice", "AcceptabilityJudgment", {s: "stand | sit"}],
	["practice", "AcceptabilityJudgment", {s: "tease | gouge"}],
	["practice", "AcceptabilityJudgment", {s: "park | drive"}],

	["begin", "Message", {
				html: { include: "begin.html" },
				} ],

        ["setcounter", "__SetCounter__", { }],
        ["sr", "__SendResults__", { }],

    ["sep", "Separator", { }],

    $items

];
''').substitute(items=items)

items = ',\n'.join(test_controllers) + ',\n' + ',\n'.join(debrief_controllers)
conf = conf_temp(items)

with open(sys.argv[1], 'w') as f:
    f.write(conf)
