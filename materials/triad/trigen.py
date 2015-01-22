#!/bin/python

import random
import string

fr = open('final.list')
verbs  = [v.strip() for v in fr]

triads = []

for v1 in verbs: 
	for v2 in verbs: 
		for v3 in verbs:
 			if len(set([v1, v2, v3])) == 3: 
				if set([v1, v2, v3]) not in triads:
					triads.append(set([v1, v2, v3]))

for i in range(100):
	random.shuffle(triads)

divisors = []
for i in range(10, 31, 1):
	if len(triads)%i == 0:
		divisors.append((i, len(triads)/i))

psize = divisors[(len(divisors)/2)][0]

part  = [triads[i:i+psize] for i in range(0, len(triads), psize)]

tlist = open('triads.list', 'w')
conf = open('triads.conf.list', 'w')

j = 1

for p in part:
	for t in p:
		t = list(t)
		l = '\t'.join(t) + '\n'	
		tlist.write(l)

		m = '[["triad", ' + str(j) + '], \"Question\", {as: [\"' + t[0] + '\", \"' + t[1] + '\", \"' + t[2] + '\"]}],\n'
		conf.write(m)
	j += 1

tlist.close()
conf.close()
