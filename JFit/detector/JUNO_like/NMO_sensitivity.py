# -*- coding: utf-8 -*-
'''
author: Jinnan Zhang
date: 2021.03.20
'''
import sys
sys.path.append('../..')
import physics.nu_oscillation.oscprob3nu as oscprob3nu
import physics.nu_oscillation.hamiltonians3nu as hamiltonians3nu
import numpy as np
import matplotlib.pyplot as plt


def reactor_average_energy(E_low=1.81, E_up=12, N=1000):
    from source.flux_HM import flux_HM, xsec_VB_DYB
    N_proton = 1e33
    Es = np.linspace(E_low, E_up, N)
    E_av = 0.0
    E_all = 0.0
    for E in Es:
        E_av += (E * flux_HM(E) * xsec_VB_DYB(E) * N_proton)
        E_all += (flux_HM(E) * xsec_VB_DYB(E) * N_proton)

    E_av /= E_all
    print('Average Energy: %e' % (E_av))  # 4.254 MeV

    # import matplotlib.pyplot as plt
    # plt.plot(Es, flux_HM(Es)*xsec_VB_DYB(Es)*N_proton)
    # plt.show()
    return E_av


def prob_baseline(baseline_low=0, baseline_up=10, N_b=400, E_av=4.253804):
    energy = E_av * 1e6
    h_vacuum_energy_indep = hamiltonians3nu.hamiltonian_3nu_vacuum_energy_independent( \
                                                                                S12_NO_BF, S23_NO_BF,
                                                                                S13_NO_BF, DCP_NO_BF,
                                                                                D21_NO_BF, D31_NO_BF)
    h_vacuum = np.multiply(1. / energy, h_vacuum_energy_indep)

    baselines = np.linspace(baseline_low, baseline_up, N_b)
    # Each element of prob: [Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt]
    prob = np.asarray([
        oscprob3nu.probabilities_3nu(h_vacuum, CONV_KM_TO_INV_EV * l)
        for l in baselines
    ])
    prob_ee = prob[:, 0]
    fig, ax = plt.subplots()
    ax.plot(baselines, prob_ee, label=r'$P_{\nu_e\to\nu_e}$')
    ax.set_xlabel('Baseline [km]')
    # ax.set_ylabel('')
    ax.legend()
    plt.show()


def prob_baseline_nd_Enu(baselines=(0, 10), Enu=(1.807, 10), N_grid=100):
    h_vacuum_energy_indep = hamiltonians3nu.hamiltonian_3nu_vacuum_energy_independent( \
                                                                                S12_NO_BF, S23_NO_BF,
                                                                                S13_NO_BF, DCP_NO_BF,
                                                                                D21_NO_BF, D31_NO_BF)

    return 0


if __name__ == "__main__":
    # reactor_average_energy()
    plt.style.use('lib/Paper.mplstyle')
    prob_baseline()
    # from physics.nu_oscillation.Prob_e2e import Prob_e2e
    # a= Prob_e2e()
    # a.out()
