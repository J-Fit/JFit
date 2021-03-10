# -*- coding: utf-8 -*-
from scipy.interpolate import interp1d
import numpy as np
import ROOT
import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_input = ROOT.TFile(base_dir + "/source/data/JUNOInputs2021_03_05.root")
h_flux = file_input.Get("HuberMuellerFlux_DYBbumpFixed")
h_xsec = file_input.Get("IBDXsec_VogelBeacom_DYB")
x_flux = np.asarray(
    [h_flux.GetBinCenter(i + 1) for i in range(h_flux.GetNbinsX())])
y_flux = np.asarray(h_flux)[1:-1]
x_xsec = np.asarray(
    [h_xsec.GetBinCenter(i + 1) for i in range(h_xsec.GetNbinsX())])
y_xsec = np.asarray(h_xsec)[1:-1]

Enu_average_reactor = 4.253804  # MeV
GW_per_fission = 1.e9 / 1.e6 / 1.602176634e-19 / 205.8371
sec_pre_day = 24. * 3600.
# The Huber+Mueller neutrino flux, antineutrino/MeV/fission, Average fission energy: 205.8371 MeV
# here unit is antineutrino/MeV/GW
flux_HM = interp1d(x=x_flux, y=y_flux * GW_per_fission * sec_pre_day)

xsec_VB_DYB = interp1d(x=x_xsec, y=y_xsec)
