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

tau = 5*ms


#eqs ='''
#dv/dt = (sin(2*pi*100*Hz*t)-v)/tau :1
#''' 
##eqs ='''
##dv/dt = (1-v)/tau :1
##''' 
#G = NeuronGroup(1, eqs, method='euler')#method='exact'
#M = StateMonitor(G, 'v', record=0)
#
#G.v = 5 # initial value
#
#print('Before v = %s' %G.v[0])
#run(100*ms)
#print('After v = %s' %G.v[0])
##%%
#plot(M.t/ms, M.v[0])
#xlabel('Time (ms)')
#ylabel('v')
#legend('aa','aa')
#
eqs = '''
dv/dt = (1-v)/tau : 1 
'''
 
G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', refractory=15*ms, method='exact')
 
statemon = StateMonitor(G, 'v', record=0)
spikemon = SpikeMonitor(G)

run(50*ms)
plot(statemon.t/ms, statemon.v[0])
for t in spikemon.t:
    axvline(t/ms, ls='--', c='C1', lw=3)
    axhline(0.8, ls=':', c='C2', lw=3)
xlabel('Time (ms)')
ylabel('v');