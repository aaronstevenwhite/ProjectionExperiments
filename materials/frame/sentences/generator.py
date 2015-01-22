#!/bin/python

## generate frame-verb pairings for frame experiment

## Dictionary of verbs with 
# only specify pp if different from past
# only specify 1pres if different from root

from itertools import combinations, permutations
from random import shuffle
import numpy as np

verbdict = {
'say' 			: {'past' : 'said'}, 
'tell' 			: {'past' : 'told'}, 
'deny'			: {'past' : 'denied', 'pres' : 'denies'}, 
'believe'		: {'past' : 'believed'}, 
'understand'	: {'past' : 'understood'}, 
'think'			: {'past' : 'thought'}, 
'doubt'			: {'past' : 'doubted'}, 
'realize'		: {'past' : 'realized'}, 
'forget'		: {'past' : 'forgot', 'pp' : 'forgotten'}, 
'remember'		: {'past' : 'remembered'}, 
'hope'			: {'past' : 'hoped'}, 
'worry'			: {'past' : 'worried', 'pres' : 'worries'}, 
'hate'			: {'past' : 'hated'}, 
'love'			: {'past' : 'loved'}, 
'bother'		: {'past' : 'bothered', 'pres' : 'bothers'}, 
'amaze'			: {'past' : 'amazed'}, 
'need'			: {'past' : 'needed'}, 
'want'			: {'past' : 'wanted'}, 
'demand'		: {'past' : 'demanded'}, 
'promise'		: {'past' : 'promised'}, 
'allow'			: {'past' : 'allowed'}, 
'forbid'		: {'past' : 'forbade', 'pp' : 'forbidden'}, 
'expect'		: {'past' : 'expected'}, 
'guess'			: {'past' : 'guessed', 'pres' : 'guesses'}, 
'feel'			: {'past' : 'felt'}, 
'hear'			: {'past' : 'heard'}, 
'see'			: {'past' : 'saw', 'pp' : 'seen'}, 
'imagine'		: {'past' : 'imagined'}, 
'pretend'		: {'past' : 'pretended'}, 
'suppose' 		: {'past' : 'supposed'}
}

be = {
	'1pres' : ('am', 'is'),
	'past' 	: ('was', 'was')
}

## proper nouns (33)

proper = [
	'Jean',
	'Beth',
	'Fletch',
	'Annette',
	'Steve',
	'Jan',
	'Austin',
	'Genie',
	'Bonnie',
	'Carrie',
	'Cathy',
	'Lee',
	'Paz',
	'Belle',
	'Zeke',
	'Fred',
	'Nan',
	'Vera',
	'Kate',
	'Sam',
	'Derek',
	'Mel',
	'Lea',
	'Wendy',
	'Kurt',
	'Tino',
	'Hilary',
	'Sadie',
	'Bess',
	'Jill',
	'Hal',
	'Rob',
	'Gary',
	'Serge'
]

## animate nouns nominative

first = ['I']*34
pronoun = ['she', 'he']*17

sub = zip(first, pronoun, proper) 

subgen = (x for x in sub)

sub_acc = [
	'him',
	'her',
	'me'
]

## animate nouns accusative

comp_anim = [
	'her', 
	'him',
	'them'
]

## inanimate nouns

comp_inanim = [
	'the table', 
	'the cup', 
	'the bottle'
]

## content

content = [
	'the story',
	'the news',
	'the problem'
]

## WH

wh = [
	'where',
	'how',
	'why'
]

## small clause

sc = [
	'go to the store',
	'eat a sandwich',
	'bake bread'
]

## gerund

ger = [
	'going to the store',
	'eating a sandwich',
	'baking bread'
]

##
# chose English verbs that don't inflect for past for the clause

fin = [
	'he read a book to the kid', 
	'they put some food in the fish bowl',
	'she fit the part'
]

## non-finite clauses

inf_nosub = ['to ' + ' '.join(clause.split()[1:]) for clause in fin]
inf_wsub = ['for ' + noun + ' ' + clause for noun, clause in zip(comp_anim, inf_nosub)]
ecm = [' '.join(clause.split()[1:]) for clause in inf_wsub]


## expletive subject

there = [
	'there to be a bird at the feeder', 
	'there to be a truck at the curb', 
	'there to be a seat at the bar'
]


##
## frames (33)
##

xp = lambda xp: lambda subject: lambda verb: subject[0].upper() + subject[1:] + ' ' + verb[0] + xp
xpxp = lambda xp1: lambda xp2: lambda subject: lambda verb: xp(xp1)(subject)(verb) + ' ' + xp2
slift = lambda clause: lambda subject: lambda verb: clause[0].upper() + clause[1:] + ', ' + subject + ' ' + verb[0]
passive = lambda clause: lambda subject: lambda verb: subject[0].upper() + subject[1:] + ' ' + verb[2] + ' ' + verb[1] + ' ' + clause
quote = lambda clause: lambda subject: lambda verb: subject[0].upper() + subject[1:] + ' ' + verb[0] + ' "' + clause + '"'
degree = lambda clause: lambda subject: lambda verb: 'What ' + subject + ' ' + verb[0] + ' most of all ' + verb[3] + ' ' + clause
explsubj = lambda clause: lambda expl: lambda subject: lambda thunk: lambda verb: expl + ' ' + verb[4] + ' ' + subject + ' ' + clause

framedict = {
	'null' 				: [xp('') for x in next(subgen)], 
	'to' 				: [xp(' to') for x in next(subgen)],
	'so' 				: [xp(' so') for x in next(subgen)],  
	'degree_fin' 		: [degree('that ' + clause) for clause in fin],
	'degree_inf' 		: [degree(clause) for clause in inf_nosub], 
	'object_control'	: [xp(' ' + clause) for clause in there],
	'finite_overtC' 	: [xp(' that ' + clause) for clause in fin],
	'if'				: [xp(' if ' + clause) for clause in fin], 
	'NP_PP' 			: [xpxp(' ' + animnoun)('about ' + noun) for animnoun, noun in zip(comp_anim, comp_inanim)],
	'passive_fin'		: [passive('that ' + clause) for clause in fin],
	'passive_inf' 		: [passive(clause) for clause in inf_nosub], 
	'inf+Q' 			: [xpxp(' ' + whword)(clause) for whword, clause in zip(wh, inf_nosub)], 
	'finite+Q' 			: [xpxp(' ' + whword)(clause) for whword, clause in zip(wh, fin)], 
	'NP' 				: [xp(' ' + noun) for noun in comp_inanim], 
	'inf_overtC' 		: [xp(' ' + clause) for clause in inf_wsub], 
	'NP_CP_nullC' 		: [xpxp(' ' + animnoun)(clause) for animnoun, clause in zip(comp_anim, fin)], 
	'ecm' 				: [xp(' ' + clause) for clause in ecm], 
	'control' 			: [xp(' ' + clause) for clause in inf_nosub], 
	'PP_CP' 			: [xpxp(' to ' + animnoun)('that ' + clause) for animnoun, clause in zip(comp_anim, fin)], 
	'small_clause' 		: [xpxp(' ' + animnoun)(vp) for animnoun, vp in zip(comp_anim, sc)], 
	#'quotative' 		: [quote(clause) for clause in fin], 
	'NP_NP' 			: [xpxp(' ' + animnoun)(noun) for animnoun, noun in zip(comp_anim, comp_inanim)], 
	'expletive_object' 	: [xp(' it that ' + clause) for clause in fin], 
	'NP_finite_overtC' 	: [xpxp(' ' + animnoun)('that ' + clause) for animnoun, clause in zip(comp_anim, fin)], 
	'finite_nullC' 		: [xp(' ' + clause) for clause in fin], 
	'about' 			: [xp(' about ' + noun) for noun in comp_inanim], 
	'slift_first' 		: [slift(clause)('I') for clause in fin], 
	'slift_proper' 		: [slift(clause)(p) for p, clause in zip(proper[0:3], fin)],
	'gerundive' 		: [xp(' ' + vp) for vp in ger], 
	'expl_sub_fin'		: [explsubj('that ' + clause)('It')(subject) for subject, clause in zip(sub_acc, fin)],
	'expl_sub_nonfin' 	: [explsubj(clause)('It')(subject) for subject, clause in zip(sub_acc, inf_nosub)],
	'expl_sub_fin_wh'	: [explsubj(whword + ' ' + clause)('It')(subject) for whword, subject, clause in zip(wh, sub_acc, fin)],
	'expl_sub_nonfin_wh': [explsubj(whword + ' ' + clause)('It')(subject) for whword, subject, clause in zip(wh, sub_acc, inf_nosub)],
	'expl_sub_deg_fin' 	: [explsubj('most of all is that ' + clause)('What')(subject) for subject, clause in zip(sub_acc, fin)],
	'expl_sub_deg_inf' 	: [explsubj('most of all is ' + clause)('What')(subject) for subject, clause in zip(sub_acc, inf_nosub)]
}


def find_verb_forms(verbdict):
	vd = {v : [] for v in verbdict}
	for v in verbdict:
		verb = [verbdict[v]['past']]

		try:
			verb.append(verbdict[v]['pp'])
		except KeyError:
			verb.append(verbdict[v]['past'])
				
		verb.extend(be['past'])
	
		try:
			verb.append(verbdict[v]['pres'])
		except KeyError:
			verb.append(v + 's')

		vd[v] = verb

	return vd


def gen_sents(sub=sub, verbdict=verbdict, framedict=framedict):

	def add(vxf, v, frametype, complete):
		try:
			vxf[v][frametype].append(complete)
		except:
			vxf[v][frametype] = [complete]

	verb_forms = find_verb_forms(verbdict)

	vxf = {v : {} for v in verbdict}

	sub_gen = (s for s in sub)

	for frametype in framedict:
		curr_subs = sub_gen.next()
		if 'slift' not in frametype:
			needsverb = {s : f(s) for f, s in zip(framedict[frametype], curr_subs)}
			for s in needsverb:
				for v in vxf:
					complete = needsverb[s](verb_forms[v])
					add(vxf, v, frametype, complete)
		else:
			for f in framedict[frametype]:
				for v in vxf:
					if 'first' in frametype:
						complete = f([v])
					else:						
						complete = f(verb_forms[v])
					add(vxf, v, frametype, complete)
	
	return vxf	

def print_vxf(vxf, verb):
	out = open(verb, 'w')
	for frame in vxf[verb]:
		fs = '.\n'.join(vxf[verb][frame]) + '.\n'
		out.write(fs)
	out.close()

def print_vxf_all(vxf):
	for verb in vxf:
		print_vxf(vxf, verb)


def make_ibex_controllers(vxf, tokens_per_type = 3, obs_per_token = 3, subjects = 90):
	verb_types = len(vxf.keys())
	frame_types = len(vxf.values()[0].keys())
		
	total_sents = verb_types * frame_types * tokens_per_type
	total_obs = total_sents * obs_per_token
	groups = total_obs / subjects 
	batches = subjects / obs_per_token	

	def gen_item_sequence(groups=groups, total_sents=total_sents):
		## n_{i, 1, 1} = i
		## n_{i, j, k} = n_{i, j, k-1} + group_size + 1

		frame_types = total_sents / groups

		nums = range(total_sents)
		nums = np.array([nums[i:i+frame_types] for i in range(0, total_sents, frame_types)])

		batches = 30

		sequences = np.zeros([batches, groups], dtype=np.int)

		for i in range(batches):
			k = i - 1
			for j in range(groups):
				try:
					k += 1
					sequences[i, j] = nums[j, k]
				except IndexError:
					sequences[i, j] = sequences[i, j-1] + 1
					k = 0

		return sequences

	seq_mat = gen_item_sequence()

	def flatten_vxf(vxf):
		indices = range(3)*34*10
		indices = indices + indices[1:] + indices[0:1] + indices[2:] + indices[0:2]

		flat_list = []

		i = 0
		for j in range(3):
			for frame in vxf.values()[0]:
				for verb in vxf:
					index = indices[i]
					sent = vxf[verb][frame][index]
					flat_list.append((sent, verb, frame))
					i += 1

		return flat_list

	flat_list = flatten_vxf(vxf)

	def find_items_mat(seq_mat=seq_mat, flat_list=flat_list):

		find_sent = lambda i: flat_list[i]

		items_lists = []

		for l, sub in enumerate(seq_mat):
			sub_sents = map(find_sent, sub)
			shuffle(sub_sents)
			items_lists.append(sub_sents)

		return items_lists

	items_mat = np.array(find_items_mat())

	gen_cont = lambda sent: lambda gnum: '\t[[\"frame\", ' + str(gnum) + "], \"AcceptabilityJudgment\", {s: '" + sent + ".'}],"

	need_groups = [(n, gen_cont(s[0]), s[1:]) for nums, sents in zip(seq_mat, items_mat) for n, s in zip(nums, sents)]
	need_groups.sort()
	almost_done = [(f(n/30 + 1), pair) for n, f, pair in need_groups]

	conts, pairs = zip(*almost_done)	
	pairs = ['\t'.join(list(p)) for p in pairs]

	return conts, pairs, seq_mat, items_mat

#x = make_ibex_controllers(gen_sents())


def print_ibex_data(data, pairs):
	f = open('frame_data.js', 'w')
	g = open('vf_pairs.list', 'w')

	f.write(data)
	g.write(pairs)

def main():
	header = '''var shuffleSequence = seq("consent", "intro", "p", "begin", sepWith("sep", "frame"));
var practiceItemTypes = ["p"];

var defaults = [
	"Separator", {
		transfer: 500,
		hideProgressBar: true,
		normalMessage: "+"
	},
	"Message", {
		transfer: "keypress",
		hideProgressBar: true
	},
	"AcceptabilityJudgment", {
		q: '',
		as: ["1", "2", "3", "4", "5", "6", "7"],
        presentAsScale: true,
        instructions: "Use number keys or click boxes to answer.",
        leftComment: "(Very bad)", rightComment: "(Perfect)"
	},
	"Form", { hideProgressBar: true }
];

var items = [

	["consent", "Form", {
        html: { include: "frame_consent.html" },
		validators: {age: function (s) { if (s.match(/^\d+$/)) return true; 
							else return "Bad value for age"; }}
    } ],

	["intro", "Message", {html: { include: "frame_intro.html" }}],

	["p", "AcceptabilityJudgment", {s: "Gary ran that he ate a piece of cake."}],
	["p", "AcceptabilityJudgment", {s: "Stacy built a house for him."}],
	["p", "AcceptabilityJudgment", {s: "I failed going to the store."}],

	["begin", "Message", {
				html: { include: "frame_beg.html" },
				} ],

    ["sep", "Separator", { }],

''' 
	
	vxf = gen_sents(sub, verbdict, framedict)
	print_vxf_all(vxf)

	conts, pairs, seq_mat, items_mat = make_ibex_controllers(vxf)

	def cat_for_out(data_list):
		out = '\n'.join(data_list)
		return out

	conts = cat_for_out(conts)
	pairs = cat_for_out(pairs)

	data_almost_complete = header + conts 

	d_cont_func = lambda debrief: "\t[[\"frame\", 103], \"Form\", {html: { include: \"" + debrief + "\" }}],"

	debrief_conts = []

	for i in range(30):
		d = 'frame_debrief' + str(i) + '.html'
		debrief_conts.append(d_cont_func(d))

	debrief_conts = '\n'.join(debrief_conts)

	data_complete = data_almost_complete + '\n' + debrief_conts + '\n\n];'
	print_ibex_data(data_complete, pairs)

	return seq_mat, items_mat

def extract_from_raw(results):
	demo = []
	answ = []
	last = '1'

	meta = {}
	data = []

	for line in results:
		print line
		split = line.strip().split(',')
		print split
		if line[0] != '#':
			#if split[5] == 'consent':
			#	pass
				#meta[split[7]] = ' '.join(split[8:])
			if split[5] == 'frame' and len(split[-1]) > 2:
				data.append(split[-1])
		elif last[0] != '#':
			if meta:
				demo.append(meta)
				meta = {}
			elif data:
				answ.append(data)
				data = []
			
		last = line

	return (demo, answ)

if __name__ == '__main__':
	pass
	seq, item = main()
