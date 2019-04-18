#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:03:50 2019
test the basic functions of this simulation environment
@author: lijiwei
"""
from brian2 import *
prefs.codegen.target ='numpy'
start_scope()
 
#N = 100
#tau = 10*ms
#v0_max = 3.
#duration = 1000*ms
#sigma = 0.2

#eqs = '''
#dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5 : 1 (unless refractory)
#v0 : 1
#'''
# 
#G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', refractory=5*ms,method='euler')
#M= SpikeMonitor(G)
#G.v0 = 'i*v0_max/(N-1)' #'rand()'
#
#run(duration)
#
#figure(figsize=(12,4))
#subplot(121)
#plot(M.t/ms, M.i, '.k')
#xlabel('Time (ms)')
#ylabel('Neuron index')
#subplot(122)
#plot(G.v0, M.count/duration)
#xlabel('v0')
#ylabel('Firing rate (sp/s)')
#
##spikemon = SpikeMonitor(G)
## 
##run(50*ms)
## 
##plot(spikemon.t/ms, spikemon.i, '.k')
##xlabel('Time (ms)')
##ylabel('Neuron index')

N = 1000
tau = 10*ms
vr = -70*mV
vt0 = -50*mV
delta_vt0 = 5*mV
tau_t = 100*ms
sigma = 0.5*(vt0-vr)
v_drive = 2*(vt0-vr)
duration = 100*ms
 
eqs = '''
dv/dt = (v_drive+vr-v)/tau + sigma*xi*tau**-0.5 : volt
dvt/dt = (vt0-vt)/tau_t : volt
'''
 
reset = '''
v = vr
vt += delta_vt0
'''
 
G = NeuronGroup(N, eqs, threshold='v>vt', reset=reset, refractory=5*ms, method='euler')
spikemon = SpikeMonitor(G)
statemonv = StateMonitor(G,'v',record=3)
statemonvt=StateMonitor(G,'vt',record=3)

 
G.v = 'rand()*(vt0-vr)+vr'
G.vt = vt0
 
run(duration)
 
figure(figsize =(12,4))
subplot(121)
hist(spikemon.t/ms, 100, histtype='stepfilled', facecolor='k', weights=ones(len(spikemon))/(N*defaultclock.dt))
xlabel('Time (ms)')
ylabel('Instantaneous firing rate (sp/s)')
subplot(122)
plot(statemonv.t/ms, statemonv.v[0])
plot(statemonvt.t/ms, statemonvt.vt[0])
xlabel('Time (ms)')
ylabel('v')