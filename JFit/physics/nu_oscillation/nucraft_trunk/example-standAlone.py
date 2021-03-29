#!/usr/bin/env python
from __future__ import print_function, division

from numpy import *
max = amax
min = amin

import os, sys
from time import time
from sys import stdout

from NuCraft import *

# Example script that reproduces the left side of figure 1 from the
# Akhmedov/Rassaque/Smirnov paper arxiv 1205.7071;
# for the core region, results differ because NuCraft takes into account
# the deviation of Y_e from 0.5 due to heavy elements in the Earth's core.

# number of energy bins
eBins = 2
zBins = 200
# zenith angles for the four plots
# zList = arccos([-1., -0.8, -0.4, -0.1])
zList = arccos(linspace(-1, 0, zBins))
# energy range in GeV
eList = linspace(0.5, 4, eBins)

# parameters from arxiv 1205.7071
theta23 = arcsin(sqrt(0.545)) / pi * 180.
theta13 = arcsin(sqrt(0.0218)) / pi * 180.
theta12 = arcsin(sqrt(0.307)) / pi * 180.
DM21 = 7.53e-5
DM31 = 2.453e-3 + DM21
DM31_IO = -2.546e-3 + DM21

# Akhemdov is implicitely assuming an electron-to-neutron ratio of 0.5;
# he is also using the approximation DM31 = DM32;
# if you want to reproduce his numbers exactly, switch the lines below, and turn
# atmosphereMode to 0 (no handling of the atmosphere because of )
AkhmedovOsci = NuCraft((1., DM21, DM31 - DM21), [(1, 2, theta12),
                                                 (1, 3, theta13, 0),
                                                 (2, 3, theta23)],
                       earthModel=EarthModel("prem", y=(0.5, 0.5, 0.5)),
                       detectorDepth=0.7,
                       atmHeight=0.)
AkhmedovOsci_IO = NuCraft((1., DM21, DM31_IO - DM21), [(1, 2, theta12),
                                                       (1, 3, theta13, 0),
                                                       (2, 3, theta23)],
                          earthModel=EarthModel("prem", y=(0.5, 0.5, 0.5)),
                          detectorDepth=0.7,
                          atmHeight=0.)
# AkhmedovOsci = NuCraft((1., DM21, DM31), [(1,2,theta12),(1,3,theta13,0),(2,3,theta23)])

# To compute weights with a non-zero CP-violating phase, replace     ^  this zero
# by the corresponding angle (in degrees); this will add the phase to the theta13 mixing matrix,
# as it is done in the standard parametrization; alternatively, you can also add CP-violating
# phases to the other matrices, but in the 3-flavor case more than one phase are redundant.

# atmosphereMode = 0   # use fixed atmsopheric depth (set to 0 km if line 42 is not commented out)
atmosphereMode = 3  # default: efficiently calculate eight path lenghts per neutrino and take the average

# This parameter governs the precision with which nuCraft computes the weights; it is the upper
# limit for the deviation of the sum of the resulting probabilities from unitarity.
# You can verify this by checking the output plot example-standAlone2.png.
numPrec = 5e-4

# 12, -12:  NuE, NuEBar
# 14, -14:  NuMu, NuMuBar
# 16, -16:  NuTau, NuTauBar
pType = 14

print("Calculating...")
# two methods of using the nuCraft instance:

# saving the current time to measure the time needed for the execution of the following code
t = time()

### using particles
"""
from collections import namedtuple
DatPart = namedtuple("DatPart", ("zen", "eTrunc", "eMuex", "eMill", "eTrumi"))
SimPart = namedtuple("SimPart", DatPart._fields + ("atmWeight", "zenMC", "eMC", "eMuon", "mcType", "oscProb"))

prob = array([zeros([3,eBins]),  zeros([3,eBins]), zeros([3,eBins]), zeros([3,eBins])])

for index, zenith in enumerate(zList):
   for eIndex, energy in enumerate(eList):
      
      p = SimPart(0.,0.,0.,0.,0.,0.,
                  zenith,energy,0.,pType, -1.)
      
      # one call to nuCraft per particle; could also pass all particles in a list
      pList = AkhmedovOsci.CalcWeights([p], numPrec=numPrec, atmMode=atmosphereMode)
      prob[index][:, eIndex] = pList[0].oscProb
"""

### using lists (arrays, actually)
zListLong, eListLong = meshgrid(zList, eList)
zListLong = zListLong.flatten()
eListLong = eListLong.flatten()
tListLong = ones_like(eListLong) * pType

# actual call to nuCraft for weight calculations:
prob = AkhmedovOsci.CalcWeights((tListLong, eListLong, zListLong),
                                numPrec=numPrec,
                                atmMode=atmosphereMode)
prob_IO = AkhmedovOsci_IO.CalcWeights((tListLong, eListLong, zListLong),
                                      numPrec=numPrec,
                                      atmMode=atmosphereMode)

prob = rollaxis(array(prob).reshape(len(eList), len(zList), -1), 0, 3)
prob_IO = rollaxis(array(prob_IO).reshape(len(eList), len(zList), -1), 0, 3)
# rollaxis is only needed to get the same shape as prob from above,
# i.e., four elements for the different zenith angles, of which each is an
# array of 3 x eBins (three oscillation probabilities for every energy bin)

print("Calculating the probabilities took %f seconds." % (time() - t))
print("Plotting...")

try:
    import matplotlib
except ImportError:
    raise ImportError(
        "Sorry, matplotlib does not seem to be installed. All calculations were finished, "
        + "though, so nuCraft seems to work well.")
if float(matplotlib.__version__[0:3]) < 1.5:
    matplotlib.use(
        'Agg')  # to enable headless output, i.e., when no display is connected
    matplotlib.rc('axes', grid=True, color_cycle=['b', 'r', 'k'])
    # matplotlib.rc('text', usetex=True)   # uncomment this line if plot labels are garbled
else:
    matplotlib.rc('axes',
                  grid=True,
                  prop_cycle=matplotlib.rcsetup.cycler(color=['b', 'r', 'k']))
from pylab import *

# plot the probabilities
fig = figure(figsize=(6, 4))
ax1G = fig.add_subplot(111)
i_e = 1
ax1G.set_title(r"$\nu_\mu \to\nu_\mu$ @ %.1f GeV" % (eList[i_e]))
ax1G.set_xlabel(r"Zenith angle [degree]")
ax1G.plot(zList / pi * 180, prob[:, 1, i_e], label='NO')
ax1G.plot(zList / pi * 180, prob_IO[:, 1, i_e], label='IO')
ax1G.legend()
matplotlib.pyplot.show()
fig.savefig('./pics/Angel_%.1fGeV.png' % (eList[i_e]))

# ax1 = fig.add_subplot(411)
# ax1.plot(eList, prob[0].T)
# ax1.set_title('NuMu to NuE (blue), NuMu (red), and NuTau (black)')
# ax1.set_ylabel(r'zenith angle $%d^\circ$' % (zList[0] / pi * 180))
# # ax1.set_xlim([1, 20])
# ax2 = fig.add_subplot(412)
# ax2.plot(eList, prob[1].T)
# ax2.set_ylabel(r'zenith angle $%d^\circ$' % (zList[1] / pi * 180))
# # ax2.set_xlim([1, 20])
# ax3 = fig.add_subplot(413)
# ax3.plot(eList, prob[2].T)
# ax3.set_ylabel(r'zenith angle $%d^\circ$' % (zList[2] / pi * 180))
# # ax3.set_xlim([1, 20])
# ax4 = fig.add_subplot(414)
# ax4.plot(eList, prob[3].T)
# ax4.set_ylabel(r'zenith angle $%d^\circ$' % (zList[3] / pi * 180))
# # ax4.set_xlim([1, 20])
# ax4.set_xlabel("neutrino energy / GeV")

# savefig("./pics/example-standAlone1.png", dpi=160)

# plot the verification of unitarity (sum(P) = 1)
# figa = figure(figsize=(6, 10))
# ax1a = figa.add_subplot(411)
# ax1a.plot(eList, sum(prob[0], 0))
# ax1a.set_title('unitarity test')
# ax1a.set_ylabel(r'zenith angle $%d^\circ$' % (zList[0] / pi * 180))
# # ax1a.set_xlim([1, 20])
# ax2a = figa.add_subplot(412)
# ax2a.plot(eList, sum(prob[1], 0))
# ax2a.set_ylabel(r'zenith angle $%d^\circ$' % (zList[1] / pi * 180))
# # ax2a.set_xlim([1, 20])
# ax3a = figa.add_subplot(413)
# ax3a.plot(eList, sum(prob[2], 0))
# ax3a.set_ylabel(r'zenith angle $%d^\circ$' % (zList[2] / pi * 180))
# # ax3a.set_xlim([1, 20])
# ax4a = figa.add_subplot(414)
# ax4a.plot(eList, sum(prob[3], 0))
# ax4a.set_ylabel(r'zenith angle $%d^\circ$' % (zList[3] / pi * 180))
# # ax4a.set_xlim([1, 20])
# ax4a.set_xlabel("neutrino energy / GeV")

# savefig("./pics/example-standAlone2.png", dpi=160)

print(" \n Completed without fatal errors! \n ")
