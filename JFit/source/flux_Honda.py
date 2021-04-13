# -*- coding: utf-8 -*-
''' atmospheric neutrino flux
author: Jinnan Zhang
zhangjinnan@ihep.ac.cn
date: 2021.03.23
'''


class flux_Honda:
    def __init__(self, exp_site='juno'):
        import os
        import numpy as np
        curPath = os.path.dirname(os.path.realpath(__file__))
        self.all_diret_solmin = np.loadtxt(curPath + '/data/' + exp_site +
                                           '-ally-01-01-solmin.d',
                                           skiprows=2)
        from scipy import interpolate
        self.f_flux = {}
        self.f_flux[14] = interpolate.InterpolatedUnivariateSpline(
            self.all_diret_solmin[:, 0], self.all_diret_solmin[:, 1])
        self.f_flux[-14] = interpolate.InterpolatedUnivariateSpline(
            self.all_diret_solmin[:, 0], self.all_diret_solmin[:, 2])
        self.f_flux[12] = interpolate.InterpolatedUnivariateSpline(
            self.all_diret_solmin[:, 0], self.all_diret_solmin[:, 3])
        self.f_flux[-12] = interpolate.InterpolatedUnivariateSpline(
            self.all_diret_solmin[:, 0], self.all_diret_solmin[:, 4])
        self.particle_list = {12, -12, 14, -14}

    def get_flux(self, Enu, flavor_ID=12):
        import numpy as np
        if flavor_ID in self.particle_list:
            return self.f_flux[flavor_ID](Enu)
        else:
            print("WRONG PDGID!")
            return np.zeros_like(Enu)

    def get_flavor_ratio(self, Enu, flavor_a=12, flavor_b=14):
        '''
        Enu: neutrino energy in GeV.

        Flavor: PDGID

        '''
        if {flavor_a, flavor_b}.issubset(self.particle_list):
            return self.f_flux[flavor_a](Enu) / self.f_flux[flavor_b](Enu)
        else:
            print("WRONG PDGID!")
            return np.zeros_like(Enu)


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description="For JUNO NMO analysis.")
    parser.add_argument("--JUNOFlux",
                        action="store_true",
                        default=False,
                        help="Show atmospheric Flux at JUNO.")
    parser.add_argument("--INO",
                        action="store_true",
                        default=False,
                        help="Show atmospheric Flux at INO.")
    parser.add_argument("--JUNOINO",
                        action="store_true",
                        default=False,
                        help="Show atmospheric Flux at diff of INO and JUNO.")

    return parser


def ShowJUNOFlux():
    my_juno_flux = flux_Honda()
    Enu = np.linspace(1, 20,100)

    phi_mu = my_juno_flux.get_flux(Enu, flavor_ID=14)
    phi_mu_bar = my_juno_flux.get_flux(Enu, flavor_ID=-14)
    phi_e = my_juno_flux.get_flux(Enu, flavor_ID=12)
    phi_e_bar = my_juno_flux.get_flux(Enu, flavor_ID=-12)
    plt.plot(Enu, phi_mu / phi_mu_bar, label=r'$\nu_{\mu}$/$\bar{\nu}_{\mu}$')
    plt.plot(Enu, phi_e / phi_e_bar, label=r'$\nu_{e}$/$\bar{\nu}_{e}$')
    plt.plot(Enu, (phi_mu + phi_mu_bar) / (phi_e + phi_e_bar), label=r'($\nu_{\mu}$+$\bar{\nu}_{\mu}$)/($\nu_{e}$+$\bar{\nu}_{e}$)')
    # plt.plot(Enu,
    #          my_juno_flux.get_flux(Enu, flavor_ID=14),
    #          label=r'$\nu_{\mu}$')
    # plt.plot(Enu,
    #          my_juno_flux.get_flux(Enu, flavor_ID=-14),
    #          label=r'$\bar{\nu}_{\mu}$')
    # plt.plot(Enu, my_juno_flux.get_flux(Enu, flavor_ID=12), label=r'$\nu_{e}$')
    # plt.plot(Enu,
    #          my_juno_flux.get_flux(Enu, flavor_ID=-12),
    #          label=r'$\bar{\nu}_{e}$')
    # plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Neutrino Energy [GeV]')
    plt.ylabel(r'$(m^2\cdot sec\cdot sr\cdot GeV)^{-1}$')
    plt.ylabel(r'Flavour ratio')
    plt.legend()
    plt.show()


def ShowJUNO_INOFLux():
    my_juno_flux = flux_Honda()
    my_ino_flux = flux_Honda(exp_site='ino')
    Enu = np.linspace(1, 20, 100)

    # plt.plot(Enu, my_juno_flux.get_flavor_ratio(Enu=Enu, flavor_a=12, flavor_b=14))
    plt.plot(Enu,
             my_juno_flux.get_flux(Enu, flavor_ID=14) /
             my_ino_flux.get_flux(Enu, flavor_ID=14),
             label=r'$\nu_{\mu}$')
    plt.plot(Enu,
             my_juno_flux.get_flux(Enu, flavor_ID=-14) /
             my_ino_flux.get_flux(Enu, flavor_ID=-14),
             label=r'$\bar{\nu}_{\mu}$')
    plt.plot(Enu,
             my_juno_flux.get_flux(Enu, flavor_ID=12) /
             my_ino_flux.get_flux(Enu, flavor_ID=12),
             label=r'$\nu_{e}$')
    plt.plot(Enu,
             my_juno_flux.get_flux(Enu, flavor_ID=-12) /
             my_ino_flux.get_flux(Enu, flavor_ID=-12),
             label=r'$\bar{\nu}_{e}$')
    # plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Neutrino Energy [GeV]')
    # plt.ylabel(r'$(m^2\cdot sec\cdot sr\cdot GeV)^{-1}$')
    plt.ylabel(r'JUNO/INO(YB)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('../Style/Paper.mplstyle')
    parser = get_parser()
    args = parser.parse_args()
    if args.JUNOFlux:
        ShowJUNOFlux()
    if args.JUNOINO:
        ShowJUNO_INOFLux()