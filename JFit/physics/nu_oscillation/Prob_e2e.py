# -*- coding: utf-8 -*-
''' pure nue -> nue
author: Jinnan Zhang
zhangjinnan@ihep.ac.cn
date: 2021.03.08
'''
import numpy as np


class Prob_e2e:
    def __init__(self, NMO=1, ME=True):
        self.NameSpace = 'PDG2020'
        if NMO == 1:
            self.NMO = 'normal'  # mass ordering, 1 for normal, others for invert
        else:
            self.NMO = 'invert'
        self.ME = ME  # in Matter or not
        import os
        import yaml
        curPath = os.path.dirname(os.path.realpath(__file__))
        OsciPar_yamlPath = curPath + "/data/OscillationParameters.yaml"
        f_osci_par = open(OsciPar_yamlPath)
        self.OsciPar = yaml.load(f_osci_par.read(), Loader=yaml.Loader)
        self.Sin_sqTheta12 = self.OsciPar[self.NameSpace][self.NMO]['sinsq12']
        self.DeltaM21_sq = self.OsciPar[self.NameSpace][self.NMO]['dmsq21']
        self.DeltaM31_sq = self.OsciPar[self.NameSpace][self.NMO]['dmsq31']
        self.DeltaM32_sq = self.OsciPar[self.NameSpace][self.NMO]['dmsq32']
        self.Sin_sqTheta13 = self.OsciPar[self.NameSpace][self.NMO]['sinsq13']

        self.matter_density = self.OsciPar['MatterDensity']
        self.cal_matter_potential()

    def out(self):
        print(self.A_MatPoten_0)

    def cal_matter_potential(self):
        M_unified_atomic_kg = 1.6605390666e-27
        N_e = self.matter_density / M_unified_atomic_kg / 2.0
        hbar_C = 197.3269804  # MeV.fm
        G_F = 1.1663787e-5
        self.A_MatPoten_0 = -2 * np.sqrt(
            2) * G_F * N_e * hbar_C * hbar_C * hbar_C * 1e-39

    def get_prob_e2e(self, Enu, baseline, ME=True):
        E = Enu
        BaseLine = baseline * 1e-3  # cm to m
        prob = 0
        Sinsq2Theta12 = (4 * self.Sin_sqTheta12 * (1 - self.Sin_sqTheta12))
        Sinsq2Theta13 = (4 * self.Sin_sqTheta13 * (1 - self.Sin_sqTheta13))
        if ME:
            A_MatPoten = E * self.A_MatPoten_0
            eta_12 = (1 - 2 * self.Sin_sqTheta12 -
                      A_MatPoten / self.DeltaM21_sq) * (
                          1 - 2 * self.Sin_sqTheta12 -
                          A_MatPoten / self.DeltaM21_sq) + Sinsq2Theta12
            eta_13 = (1 - 2 * self.Sin_sqTheta13 -
                      A_MatPoten / self.DeltaM31_sq) * (
                          1 - 2 * self.Sin_sqTheta13 -
                          A_MatPoten / self.DeltaM31_sq) + Sinsq2Theta13
            Sinsq2Theta12_M = Sinsq2Theta12 / eta_12
            Sinsq2Theta13_M = Sinsq2Theta13 / eta_13
            Sin_sqTheta12_M = (1 - np.sqrt(1 - Sinsq2Theta12_M)) / 2.
            Sin_sqTheta13_M = (1 - np.sqrt(1 - Sinsq2Theta13_M)) / 2.
            DeltaM21_sq_M = self.DeltaM21_sq * np.sqrt(eta_12)
            DeltaM31_sq_M = self.DeltaM31_sq * np.sqrt(eta_13)
            DeltaM32_sq_M = DeltaM31_sq_M - DeltaM21_sq_M
            Delta21 = 1.266932679815373 * DeltaM21_sq_M * BaseLine / E
            Delta31 = 1.266932679815373 * DeltaM31_sq_M * BaseLine / E
            Delta32 = 1.266932679815373 * DeltaM32_sq_M * BaseLine / E
            prob = 1. - Sinsq2Theta13_M * (
                (1 - Sin_sqTheta12_M) * np.sin(Delta31)**2. +
                Sin_sqTheta12_M * np.sin(Delta32)**2.) - (
                    (1 - Sin_sqTheta13_M)**
                    2.) * Sinsq2Theta12_M * np.sin(Delta21)**2.
        else:
            Delta21 = 1.266932679815373 * self.DeltaM21_sq * BaseLine / E
            Delta31 = 1.266932679815373 * self.DeltaM31_sq * BaseLine / E
            Delta32 = 1.266932679815373 * self.DeltaM32_sq * BaseLine / E
            prob = 1. - Sinsq2Theta13 * (
                (1 - self.Sin_sqTheta12) * np.sin(Delta31)**2. +
                self.Sin_sqTheta12 * np.sin(Delta32)**2.) - (
                    (1 - self.Sin_sqTheta13)**
                    2.) * Sinsq2Theta12 * np.sin(Delta21)**2.
        return prob


if __name__ == "__main__":
    a = Prob_e2e()
    a.out()