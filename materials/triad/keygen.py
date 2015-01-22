#!/bin/python

## key generator for ibex

from random import sample
from string import hexdigits

def keygen(size=15):
	key = ''.join([sample(hexdigits, 1)[0] for i in range(size)])
	return key
	

def controlgen(size=15):
	key = keygen(size)
	return (key, '[["key", 1], "Message", {html: "' + key + '", transfer: "click"}]')

def controlgen_all(n=20, size=15):
	controldict = dict([controlgen(size=15) for i in range(n)])
	
	keystr = '\n'.join(controldict.keys())
	controlstr = ',\n'.join(controldict.values())

	return keystr, controlstr

if __name__ == '__main__':
	keys, controllers = controlgen_all()

	kf = open('keys.list', 'w')
	kcf = open('key_controllers.list', 'w')

	kf.write(keys)
	kcf.write(controllers)

	kf.close()
	kcf.close()
