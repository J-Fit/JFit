# -*- coding: utf-8 -*-
''' pure nue -> nue
author: Jinnan Zhang
zhangjinnan@ihep.ac.cn
date: 2021.03.08
'''
import numpy as np


class Prob_e2e:
    def __init__(self, NMO=1, ME=True, NameSpace='PDG2020'):
        self.NameSpace = NameSpace
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
        #- sign for antineutrinos
        self.A_MatPoten_0 = -2 * np.sqrt(
            2) * G_F * N_e * hbar_C * hbar_C * hbar_C * 1e-39

    def get_prob_e2e_Amir(self, Enu, baseline, ME=True):
        '''
        Enu: MeV, baseline: cm

        Based on: arXiv:1910.12900, from Amir N. Khan, Hiroshi Nunokawa,...
        '''
        Sin_sqTheta12 = self.Sin_sqTheta12
        DeltaM21_sq = self.DeltaM21_sq
        DeltaM31_sq = self.DeltaM31_sq
        DeltaM32_sq = self.DeltaM32_sq
        Sin_sqTheta13 = self.Sin_sqTheta13

        E = Enu
        BaseLine = baseline * 1e-2  # cm to m
        prob = 0
        Sinsq2Theta12 = (4 * self.Sin_sqTheta12 * (1 - self.Sin_sqTheta12))
        Sinsq2Theta13 = (4 * self.Sin_sqTheta13 * (1 - self.Sin_sqTheta13))
        if ME:
            # reverse the relation
            A_MatPoten = E * self.A_MatPoten_0
            # eq. 6
            DeltaMsq_ee = DeltaM31_sq * (1 - Sin_sqTheta12) + DeltaM32_sq * Sin_sqTheta12  
            
            Cos_2Theta_12 = 1 - 2 * Sin_sqTheta12
            Cos_2Theta_13 = 1 - 2 * Sin_sqTheta13
            # eq. 8
            DeltaMsq_ee_M = DeltaMsq_ee*np.sqrt((Cos_2Theta_13-A_MatPoten/DeltaMsq_ee)**2+Sinsq2Theta13)
            # eq. 7 
            Cos_2Theta_13_M=(DeltaMsq_ee*Cos_2Theta_13-A_MatPoten)/DeltaMsq_ee_M
            # eq. 11
            A_MatPoten_prime=0.5*(A_MatPoten+DeltaMsq_ee-DeltaMsq_ee_M)
            # eq.12
            Cos_sq_theta13M_minus_theta13=(DeltaMsq_ee_M+DeltaMsq_ee-A_MatPoten*Cos_2Theta_13)*0.5/DeltaMsq_ee_M
            # eq. 10 
            DeltaM21_sq_M= DeltaM21_sq*np.sqrt((Cos_2Theta_12-A_MatPoten_prime/DeltaM21_sq)**2+Cos_sq_theta13M_minus_theta13*Sinsq2Theta12)
            # eq. 9
            Cos_2Theta_12_M=(DeltaM21_sq*Cos_2Theta_12-A_MatPoten_prime)/DeltaM21_sq_M


            Sin_sqTheta13_M=(1-Cos_2Theta_13_M)/2
            Sinsq2Theta13_M = 1-Cos_2Theta_13_M*Cos_2Theta_13_M

            Sin_sqTheta12_M = (1-Cos_2Theta_12_M)/2
            Sinsq2Theta12_M=1-Cos_2Theta_12_M*Cos_2Theta_12_M

            DeltaM31_sq_M = DeltaMsq_ee_M+Sin_sqTheta12_M*DeltaM21_sq_M
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

    def get_prob_e2e_Yufeng(self, Enu, baseline, ME=True):
        '''
        Enu: MeV, baseline: cm

        Based on: https://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=6859
        '''
        Sin_sqTheta12 = self.Sin_sqTheta12
        DeltaM21_sq = self.DeltaM21_sq
        DeltaM31_sq = self.DeltaM31_sq
        DeltaM32_sq = self.DeltaM32_sq
        Sin_sqTheta13 = self.Sin_sqTheta13

        E = Enu
        BaseLine = baseline * 1e-2  # cm to m
        prob = 0
        Sinsq2Theta12 = (4 * self.Sin_sqTheta12 * (1 - self.Sin_sqTheta12))
        Sinsq2Theta13 = (4 * self.Sin_sqTheta13 * (1 - self.Sin_sqTheta13))
        if ME:
            # reverse the relation
            A_MatPoten = -E * self.A_MatPoten_0
            Delta_c = DeltaM31_sq * (1 - Sin_sqTheta12) + DeltaM32_sq * Sin_sqTheta12  # eq. 8
            alpha_c = DeltaM21_sq / Delta_c  # eq .8
            A_star = A_MatPoten * (1 - Sin_sqTheta13) / DeltaM21_sq  # eq .9
            A_c = A_MatPoten / Delta_c  # eq. 9
            Cos_2Theta_12 = 1 - 2 * Sin_sqTheta12
            Cos_2Theta_13 = 1 - 2 * Sin_sqTheta13
            # C_hat_12 = np.sqrt(1 - 2.0 * A_star * Cos_2Theta_12 +A_star * A_star)
            # C_hat_13 = np.sqrt(1 - 2.0 * A_c * Cos_2Theta_13 +A_c * A_c)
            C_hat_12_prime = np.sqrt(1 + 2.0 * A_star * Cos_2Theta_12 +A_star * A_star)
            C_hat_13_prime = np.sqrt(1 + 2.0 * A_c * Cos_2Theta_13 + A_c * A_c)

            Cos_sq_Theta12_tilde = 0.5*(1+(A_star+Cos_2Theta_12)/C_hat_12_prime)
            Cos_sq_Theta13_tilde = 0.5*(1+(A_c+Cos_2Theta_13)/C_hat_13_prime)
            Sin_sqTheta13_M=1-Cos_sq_Theta13_tilde
            Sinsq2Theta13_M = 4*Sin_sqTheta13_M*Cos_sq_Theta13_tilde
            Sin_sqTheta12_M = 1-Cos_sq_Theta12_tilde
            Sinsq2Theta12_M=4*Sin_sqTheta12_M*Cos_sq_Theta12_tilde

            DeltaM21_sq_M = Delta_c*(0.5*(1-A_c-C_hat_13_prime)+alpha_c*(C_hat_12_prime+A_star))
            DeltaM31_sq_M = Delta_c*(0.5*(1-A_c+C_hat_13_prime)+alpha_c*0.5*(C_hat_12_prime+A_star-Cos_2Theta_12))
        
            DeltaM32_sq_M = DeltaM31_sq_M - DeltaM21_sq_M
            Delta21 = 1.266932679815373 * DeltaM21_sq_M * BaseLine / E
            Delta31 = 1.266932679815373 * DeltaM31_sq_M * BaseLine / E
            Delta32 = 1.266932679815373 * DeltaM32_sq_M * BaseLine / E
            prob = 1. - Sinsq2Theta13_M * (
                (1 - Sin_sqTheta12_M) * np.sin(Delta31)**2. +
                Sin_sqTheta12_M * np.sin(Delta32)**2.) - (
                    (1 - Sin_sqTheta13_M)**
                    2.) * Sinsq2Theta12_M * np.sin(Delta21)**2.
            # print()
            
        else:
            Delta21 = 1.266932679815373 * self.DeltaM21_sq * BaseLine / E
            Delta31 = 1.266932679815373 * self.DeltaM31_sq * BaseLine / E
            Delta32 = 1.266932679815373 * self.DeltaM32_sq * BaseLine / E
            prob = 1. - Sinsq2Theta13 * (
                (1 - self.Sin_sqTheta12) * np.sin(Delta31)**2. +
                self.Sin_sqTheta12 * np.sin(Delta32)**2.) - (
                    (1 - self.Sin_sqTheta13)**
                    2.) * Sinsq2Theta12 * np.sin(Delta21)**2.
        # print("Yufeng: ",self.DeltaM31_sq)
        
        return prob

    def get_prob_e2e_YB(self, Enu, baseline, ME=True):
        '''
        Enu: MeV, baseline: cm
        '''
        E = Enu
        BaseLine = baseline * 1e-2  # cm to m
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
        # print("YB: ",self.DeltaM31_sq)
        return prob


def Check_YB_Hermitian(E_low=0.1, E_up=15., N=1000, BaseLine=52.5e5):
    def GetAsy(a, b):
        return 2 * (a - b) / (a + b)

    Es = np.linspace(E_low, E_up, N)

    # JUNO Yellow formula
    P_e2e_YB = Prob_e2e()
    y_YB = P_e2e_YB.get_prob_e2e_YB(Es, baseline=BaseLine)
    y_Yufeng=P_e2e_YB.get_prob_e2e_Yufeng(Es, baseline=BaseLine)
    y_Amir=P_e2e_YB.get_prob_e2e_Amir(Es, baseline=BaseLine)
    # Hermitian approach
    import sys
    sys.path.append('../..')
    from physics.nu_oscillation import oscprob3nu, hamiltonians3nu
    from physics.nu_oscillation.globaldefs import CONV_CM_TO_INV_EV, VCC_EARTH_CRUST, S23_NO_BF, DCP_NO_BF
    S12_NO_BF = np.sqrt(P_e2e_YB.Sin_sqTheta12)
    S13_NO_BF = np.sqrt(P_e2e_YB.Sin_sqTheta13)
    D21_NO_BF = P_e2e_YB.DeltaM21_sq
    D31_NO_BF = P_e2e_YB.DeltaM31_sq
    h_vacuum_energy_indep = hamiltonians3nu.hamiltonian_3nu_vacuum_energy_independent(
        S12_NO_BF, S23_NO_BF, S13_NO_BF, -DCP_NO_BF, D21_NO_BF,
        D31_NO_BF)  # sign - DCP_NO_BF for antineutrinos
    y_Het = np.zeros(N)
    for i, energy in enumerate(Es):
        # sign - for antineutrinos
        h_matter = hamiltonians3nu.hamiltonian_3nu_matter(h_vacuum_energy_indep, energy * 1e6,-VCC_EARTH_CRUST)  
        # h_matter = np.multiply(1/(energy*1e6),h_vacuum_energy_indep)  
        Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt = oscprob3nu.probabilities_3nu(
            h_matter, BaseLine * CONV_CM_TO_INV_EV)
        y_Het[i] = Pee
    import matplotlib.pyplot as plt
    plt.style.use('../../detector/DYB_like/lib/Paper.mplstyle')
    fig, ax = plt.subplots()
    ax.set_ylabel(r'$\frac{2\cdot(A-B)}{(A+B)}$')
    ax.set_xlabel('Neutrino Energy [MeV]')
    # ax.plot(Es, y_Het, label='Hamiltonian approach')
    # ax.plot(Es, y_YB, label='Yellow Book Approach')
    # ax.plot(Es, y_Yufeng, label='Yufeng Approach')
    # ax.plot(Es, y_Amir, label='Amir Approach')
    # ax.plot(Es, GetAsy(y_YB, y_Het), label='YB/Hamiltonian')
    ax.plot(Es, GetAsy(y_Amir,y_Yufeng), label='Amir/Yufeng')
    ax.legend()
    fig.savefig('./results/Yufeng_Amir.png')    
    ax.plot(Es, GetAsy(y_YB, y_Yufeng), label='YB/Yufeng')
    ax.plot(Es, GetAsy(y_Yufeng,y_Het), label='Yufeng/Hamiltonian')
    ax.plot(Es, GetAsy(y_Amir,y_Het), label='Amir/Hamiltonian')

    ax.legend()
    fig.savefig('./results/four_model.png')
    plt.show()
    

if __name__ == "__main__":
    Check_YB_Hermitian()
    # pass
