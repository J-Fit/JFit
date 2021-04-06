# -*- coding: utf-8 -*-
'''
author: Jinnan Zhang
date: 2021.03.20
'''
import sys
sys.path.append('../..')
import physics.nu_oscillation.oscprob3nu as oscprob3nu
import physics.nu_oscillation.hamiltonians3nu as hamiltonians3nu
import physics.nu_oscillation.nucraft_trunk.NuCraft as NuCraft
import source.flux_Honda as flux_Honda
import numpy as np
import matplotlib.pyplot as plt


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description="For JUNO NMO analysis.")
    parser.add_argument("--AtmOsci",
                        action="store_true",
                        help="Show atmospheric oscillation parttern")
    parser.set_defaults(AtmOsci=False)
    parser.add_argument("--Eup",
                        type=float,
                        default=1,
                        help="The upper bound of the energy")
    parser.add_argument("--ReaOsci",
                        action="store_true",
                        help="Show Reactor neutrino oscillation parttern")
    parser.set_defaults(ReaOsci=False)

    return parser


def ShowAtmOsciPattern(Eup=1):
    my_flux = flux_Honda.flux_Honda()

    from numpy import arccos, arcsin, sqrt, rollaxis, ones_like, linspace, pi, meshgrid, array
    # number of energy bins
    eBins = 2
    zBins = 100
    # zenith angles for the four plots
    # zList = arccos([-1., -0.8, -0.4, -0.1])
    zList = arccos(linspace(-1, 0, zBins))
    # energy range in GeV
    eList = linspace(0.5, Eup, eBins)

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
    AkhmedovOsci = NuCraft.NuCraft(
        (1., DM21, DM31 - DM21), [(1, 2, theta12), (1, 3, theta13, 0),
                                  (2, 3, theta23)],
        earthModel=NuCraft.EarthModel("prem", y=(0.5, 0.5, 0.5)),
        detectorDepth=0.7,
        atmHeight=0.)
    AkhmedovOsci_IO = NuCraft.NuCraft(
        (1., DM21, DM31_IO - DM21), [(1, 2, theta12), (1, 3, theta13, 0),
                                     (2, 3, theta23)],
        earthModel=NuCraft.EarthModel("prem", y=(0.5, 0.5, 0.5)),
        detectorDepth=0.7,
        atmHeight=0.)
    atmosphereMode = 3  # default: efficiently calculate eight path lenghts per neutrino and take the average
    numPrec = 5e-4
    # 12, -12:  NuE, NuEBar
    # 14, -14:  NuMu, NuMuBar
    # 16, -16:  NuTau, NuTauBar
    pType = 14

    zListLong, eListLong = meshgrid(zList, eList)
    zListLong = zListLong.flatten()
    eListLong = eListLong.flatten()
    tListLong = ones_like(eListLong) * pType
    prob = AkhmedovOsci.CalcWeights((tListLong, eListLong, zListLong),
                                    numPrec=numPrec,
                                    atmMode=atmosphereMode)
    prob_IO = AkhmedovOsci_IO.CalcWeights((tListLong, eListLong, zListLong),
                                          numPrec=numPrec,
                                          atmMode=atmosphereMode)
    probe = AkhmedovOsci.CalcWeights(
        (tListLong / pType * 12, eListLong, zListLong),
        numPrec=numPrec,
        atmMode=atmosphereMode)
    probe_IO = AkhmedovOsci_IO.CalcWeights(
        (tListLong / pType * 12, eListLong, zListLong),
        numPrec=numPrec,
        atmMode=atmosphereMode)
    prob = rollaxis(array(prob).reshape(len(eList), len(zList), -1), 0, 3)
    prob_IO = rollaxis(
        array(prob_IO).reshape(len(eList), len(zList), -1), 0, 3)

    probe = rollaxis(array(probe).reshape(len(eList), len(zList), -1), 0, 3)
    probe_IO = rollaxis(
        array(probe_IO).reshape(len(eList), len(zList), -1), 0, 3)

    # numu
    fig, ax1G = plt.subplots()
    i_e = 1
    ax1G.set_title(r"$\frac{N_{\nu_\mu}}{N_{\nu_\mu}^0}$ @ %.1f GeV" %
                   (eList[i_e]))
    ax1G.set_xlabel(r"Zenith angle [degree]")

    pmu2mu = prob[:, 1, i_e]
    pmu2mu_IO = prob_IO[:, 1, i_e]
    Ne2Nmu = my_flux.get_flavor_ratio(eList[i_e], flavor_a=12, flavor_b=14)
    pe2mu = probe[:, 1, i_e]
    pe2mu_IO = probe_IO[:, 1, i_e]

    survival_mu2mu = pmu2mu + Ne2Nmu * pe2mu
    survival_mu2mu_IO = pmu2mu_IO + Ne2Nmu * pe2mu_IO

    ax1G.plot(zList / pi * 180, survival_mu2mu, label='NO')
    ax1G.plot(zList / pi * 180, survival_mu2mu_IO, label='IO')
    ax1G.legend()
    # plt.show()
    fig.savefig('./pics/Nu2NuAngle_%.1fGeV.png' % (eList[i_e]))

    # nue
    fig, ax1G = plt.subplots()
    i_e = 1
    ax1G.set_title(r"$\frac{N_{\nu_e}}{N_{\nu_e}^0}$ @ %.1f GeV" %
                   (eList[i_e]))
    ax1G.set_xlabel(r"Zenith angle [degree]")

    pmu2e = prob[:, 0, i_e]
    pmu2e_IO = prob_IO[:, 0, i_e]
    Nmu2Ne = my_flux.get_flavor_ratio(eList[i_e], flavor_a=14, flavor_b=12)
    pe2e = probe[:, 0, i_e]
    pe2e_IO = probe_IO[:, 0, i_e]

    survival_e2e = pe2e + Nmu2Ne * pmu2e
    survival_e2e_IO = pe2e_IO + Nmu2Ne * pmu2e_IO

    ax1G.plot(zList / pi * 180, survival_e2e, label='NO')
    ax1G.plot(zList / pi * 180, survival_e2e_IO, label='IO')
    ax1G.legend()
    # plt.show()
    fig.savefig('./pics/Nu2eAngle_%.1fGeV.png' % (eList[i_e]))


def ShowReaOsciPattern():
    pass


if __name__ == "__main__":
    # reactor_average_energy()
    plt.style.use('lib/Paper.mplstyle')
    import sys
    parser = get_parser()
    args = parser.parse_args()
    if args.AtmOsci:
        ShowAtmOsciPattern(args.Eup)
    if args.ReaOsci:
        ShowReaOsciPattern()

    # from physics.nu_oscillation.Prob_e2e import Prob_e2e
    # a= Prob_e2e()
    # a.out()
