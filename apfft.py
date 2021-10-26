#!/usr/bin/env python3

import sys
import numpy as np
import numpy.fft as npfft
import numpy.random as npran
import matplotlib.pyplot as plt

def gau(xn,N):
	sig = 0.3
	if abs(xn) <= N:
		return np.exp(-0.5*((xn-(N-1.0)/2.0)/(sig*(N-1.0)/2.0))**2)
	else:
		return 0.0

def rec(xn,N):  #Rectangular window.
	if abs(xn) <= N:
		return 1.0 / N
	else:
		return 0.0

def han(xn,N):  #Hanning window
	if abs(xn) <= N:
		#return (np.sin(np.pi*xn/(N-1))**2)/np.sqrt(N-1)
		return (np.sin(np.pi*xn/(N-1))**2)/(N-1)* 2
	else:
		return 0.0

def bnt(xn,N):  #Blackman-Nuttall window.  High dynamic range, supposedly.
	a0 = 0.3635819
	a1 = 0.4891775
	a2 = 0.1365995
	a3 = 0.0106411
	if abs(xn) <= N:
		return a0 - a1*np.cos(2*np.pi*xn/(N-1)) + a2*np.cos(4*np.pi*xn/(N-1)) - a3*np.cos(6*np.pi*xn/(N-1))
	else:
		return 0.0

windows = {'gau' : gau,
           'rec' : rec,
					 'han' : han,
					 'bnt' : bnt}

def make_convoluted_window(winType,N):
	Nsamples = 2*N-1

	win1 = np.zeros(N)
	win2 = np.zeros(N)
	for ix in range(N):
		win1[ix] = windows[winType](ix,N)
		win2[ix] = win1[ix]

	#Make a convolution of two windows
	wc = np.zeros(Nsamples)
	for ix in range(-N+1,0):
		wc[ix+N-1] = sum( win1[-ix:N] * win2[0:N+ix] )
		wc[Nsamples-1-(ix+N-1)] = wc[ix+N-1]
	wc[N-1] = sum( win1[0:N] * win2[0:N] )

	with open('py.window','w') as f:
		for ix in range(len(wc)):
			f.write("{} {}\n".format(ix+1,wc[ix]))

	return wc

def apfft(data,winType):
	Nsamples = data.size
	N = int((Nsamples+1)/2)

	wc = make_convoluted_window(winType,N)
	#print("wc area: ", np.sum(wc))
	dataWin = data*wc

	shifted = np.zeros(N)
	shifted[0] = dataWin[N-1]
	for ix in range(1,N):
		shifted[ix] = dataWin[ix-1] + dataWin[N+ix-1]
	#shifted = shifted/N

	#with open('py.apvector','w') as f:
	#	for ix in range(len(shifted)):
	#		f.write("{} {}\n".format(ix+1,shifted[ix]))

	fftShifted = npfft.fft(shifted)
	maxix = np.argmax(abs(fftShifted[0:int(N/2)]))
	freq = (1.0*maxix)/N
	phase=np.arctan2(np.imag(fftShifted[maxix]),np.real(fftShifted[maxix]))
	amp = abs(fftShifted[maxix])

	return phase, freq, amp

def cor_apfft(x,winType):
	Nsamples = x.size
	print("Corrected ApFFT Samples (must be integer): ", (Nsamples+1)/3)
	N = int((Nsamples+1)/3)
	[phase1, freq1, amp1] = apfft(x[0:2*N-1],winType)
	[phase2, freq2, amp2] = apfft(x[N:3*N-1],winType)
	d = (phase2-phase1)/2.0/np.pi
	if d > 0.5:
		d = d - 1.0
	elif d <= -0.5:
		d = d + 1.0
	freq = freq1 + d/N

	phase = 2*phase1-phase2
	if phase < 0:
		phase = phase + 2.0*np.pi
	elif phase > 2.0*np.pi:
		phase = phase - 2.0*np.pi

	if winType == 'han':
		amp = (np.pi*d*(1-d*d)/np.sin(np.pi*d))**2 * amp1 * 2
	elif winType == 'rec':
		amp = (np.pi*d/np.sin(np.pi*d))**2 * amp1 * 2
	else:
		raise ValueError("window type error: ", winType)

	return phase, freq, amp
	

def apfft_demo(N=1002, verbose=False, dump=False):
	#N= 1002 # 2N-1 samples are {-N+1, ..., 0, ..., N-1}
	Nsamples= 2*N-1
	if verbose: 
		print("N is ", N)
		print("Nsamples  is ", Nsamples)

	fa = 0.021356111211
	fracPhase = 0.313213213
	aa = 2.2
	pa = fracPhase * 2. * np.pi
	Anoise = 0.01

	winType = 'han'

	n=np.arange(-N+1, N)
	#n=np.arange(1, 2*N)
	noise = Anoise*aa*((np.random.random(Nsamples)*2)-1.0)
	x=aa*np.cos(2*np.pi*fa*n + pa) + noise

	pa0 = 2*np.pi*fa*(-N+0) + pa
	if pa0 < 0:
		pa0 = pa0 + 2*np.pi*(abs(int(pa0/2/np.pi))+1)
		#pa00 = pa0
		#ix = 1
		#while pa00 < 0:
		#	pa00 = pa0 + 2*np.pi*ix
		#	ix += 1
		#pa0 = pa00
	elif pa0 > 2*np.pi:
		pa0 = pa0 - 2*np.pi*abs(int(pa0/2/np.pi))
		#pa00 = pa0
		#ix = 1
		#while pa00 > 2*np.pi:
		#	pa00 = pa0 - 2*np.pi*ix
		#	ix += 1
		#pa0 = pa00

	#with open('py.signal','w') as f:
	#	for ix in range(len(x)):
	#		#print(ix, x[ix])
	#		f.write("{} {}\n".format(ix+1,x[ix]))

	[phase, freq, amp] = apfft(x,winType)

	[cor_phase, cor_freq, cor_amp] = cor_apfft(x,winType)

	if dump:
		with open('py.dump','a') as f:
			f.write("{}, {} {} {}\n".format(Nsamples, abs((cor_freq-fa)/fa) , abs((cor_phase-pa0)/pa0), abs((cor_amp-aa)/aa)))

	if verbose:
		print("   Actual frequency:  {0:0.12f}".format(fa))
		print(" Detected frequency:  {0:0.12f} ({1:0.12f})".format(freq,(freq-fa)/fa))
		print("Corrected frequency:  {0:0.12f} ({1:0.12f})".format(cor_freq,(cor_freq-fa)/fa))
		print()
		print("       Actual phase:  {0:0.12f}".format(pa))
		print("     Detected phase:  {0:0.12f} ({1:0.12f})".format(phase,(phase-pa)/pa))
		print()
		print("      Actual phase0:  {0:0.12f}".format(pa0))
		print("   Corrected phase0:  {0:0.12f} ({1:0.12f})".format(cor_phase,(cor_phase-pa0)/pa0))
		print()
		print("   Actual amplitude:  {0:0.12f}".format(aa))
		print(" Detected amplitude:  {0:0.12f} ({1:0.12f})".format(amp,(amp-aa)/aa))
		print("Corrected amplitude:  {0:0.12f} ({1:0.12f})".format(cor_amp,(cor_amp-aa)/aa))
		with open('accuracy_phase.dat','a') as f:
			f.write("{3}   {0}   {1}   {2}\n".format(pa,abs(phase),(pa-abs(phase))/pa,Nsamples))
		with open('accuracy_freq.dat','a') as f:
			f.write("{3}   {0}   {1}   {2}\n".format(fa,abs(freq),(fa-abs(freq))/fa,Nsamples))


if __name__ == "__main__":
	_,_ = apfft_demo(True)










