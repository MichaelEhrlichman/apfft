#!/home/shanksj/misc/anaconda/bin/python

import apfft
import numpy as np
import os

try:
	os.remove('py.dump')
except:
	pass

for i in range(21,2000,6):
	print("Evaluating N = ", i)
	apfft.apfft_demo(i, False, True)

#print("Mean accuracy: {}".format(accuracy.mean()))
#print("Standard deviation of accuracy: {}".format(accuracy.std()))
