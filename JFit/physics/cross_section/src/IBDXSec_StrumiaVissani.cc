/*
The Inverse Beta Decay interation cross section class.

@author: Jinnan Zhang: zhangjinnan@ihep.ac.cn
@date:2020.03.01
Based on the formulae by Alessandro Strumia and Francesco Vissani in 2003, 
arXiv:astro-ph/0302055v2 29 Apr 2003
*/

#include "IBDXSec_StrumiaVissani.hh"
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "omp.h"

using namespace std;

IBDXSec_StrumiaVissani::IBDXSec_StrumiaVissani()
{
   M_proton = 938.27208816 / 1e3;  //in GeV
   M_neutron = 939.56542052 / 1e3; //in GeV
   M_electron = 0.51099895 / 1e3;  //in GeV
   M_Pion = 139.57061 / 1e3;       //in GeV
   M_Nuc = (M_proton + M_neutron) / 2.;
   M_Nuc_sq = M_Nuc * M_Nuc;
   M_e_sq = M_electron * M_electron;
   M_p_sq = M_proton * M_proton;
   M_n_sq = M_neutron * M_neutron;
   M_pi_sq = M_Pion * M_Pion;
   Delta_np = M_neutron - M_proton;
   Delta_np_sq = Delta_np * Delta_np;

   //PDG2020
   // G_F_sq = 1.1663787e-5 * 1.1663787e-5;//from PDG2020
   // Cos_Cabibbo = 0.97401117;
   // Cos_Cabibbo_sq = 0.97401117 * 0.97401117; // Cos_Cabibbo * Cos_Cabibbo; new from PDG2020
   // fg1of0 = -1.2762;                 //new from PDG2020:https://pdg.lbl.gov/2020/reviews/rpp2020-rev-vud-vus.pdf: (66.6)
   // fMa2 = 1.026 * 1.026; //new from PDG2020
   // fMv2 = 0.840 * 0.840; //new from PDG2020

   //from paper 2003
   G_F_sq = 1.1663710e-5 * 1.1663710e-5; //from paper2003
   Cos_Cabibbo_sq = 0.9746 * 0.9746;     // Cos_Cabibbo * Cos_Cabibbo;
   fg1of0 = -1.270;
   fMa2 = 1;
   fMv2 = 0.71;

   // double mup, mun;
   // mup = 2.7928473;
   // mun = -1.913042;
   // fNucleonMMDiff = mup - 1 - mun; //new from PDG2019

   //paper 2003
   fNucleonMMDiff = 3.706;
   alpha_EM = 7.2973525698e-3;                              //1/137.035999074
   delta_mr = (M_n_sq - M_p_sq - M_e_sq) / (2. * M_proton); //=0.0012940846
}
IBDXSec_StrumiaVissani::~IBDXSec_StrumiaVissani()
{
}

//compute the differetian cross section dsigma/dE_e(E_nu,E_e)
//seems simply, however the range of E_e is not trivial
//both energies in MeV
double IBDXSec_StrumiaVissani::DSigmaDEe(double E_neu, double E_elec)
{
   double xsec(0);
   if (E_neu > E_Lower) //threshold
   {
      double E_e1 = GetE_eLow_SV(E_neu);
      double E_e2 = GetE_eHigh_SV(E_neu);
      if (E_elec < E_e1 || E_elec > E_e2)
         return 0;
      double E_nu(E_neu / 1e3); //to GeV
      double E_e(E_elec / 1e3); //to GeV

      // double P_nu(E_nu);                    //in GeV
      double P_e(sqrt(E_e * E_e - M_e_sq)); //in GeV

      double q_sq(M_n_sq - M_p_sq - 2. * M_proton * (E_nu - E_e)); //momentum transfer, which is t
      // q_sq = M_n_sq - M_p_sq - 2. * M_proton * (E_nu - E_e);
      double sMinusU(2. * M_proton * (E_nu + E_e) - M_e_sq);
      // sMinusU = 2. * M_proton * (E_nu + E_e) - M_e_sq;
      double sMinusMpsq(2. * M_proton * E_nu); //s-m_p^2
      // sMinusMpsq = 2. * M_proton * E_nu;

      double f1, f2, g1, g2;
      double t4Msq(q_sq / 4. / M_Nuc_sq);
      double fdnom((1 - t4Msq) * pow(1 - q_sq / fMv2, 2));
      f1 = (1 - (1 + fNucleonMMDiff) * t4Msq) / fdnom;
      f2 = fNucleonMMDiff / fdnom;
      g1 = fg1of0 / pow(1 - q_sq / fMa2, 2);
      g2 = 2. * M_Nuc_sq * g1 / (M_pi_sq - q_sq);

      double A_t, B_t, C_t;
      A_t = MAterm(q_sq, f1, f2, g1, g2);
      B_t = MBterm(q_sq, f1, f2, g1, g2);
      C_t = MCterm(q_sq, f1, f2, g1);
      xsec = A_t - (sMinusU)*B_t + pow(sMinusU, 2) * C_t; //Matrix square
      xsec = xsec * G_F_sq * Cos_Cabibbo_sq /
             (2. * PI * pow(sMinusMpsq, 2)); //dsigma/dt

      xsec = xsec * 2. * M_proton; //to dSig/dE_e

      // apply correction factors
      double rdcf = RadiativeCorr(E_e);
      xsec *= rdcf;
   }
   xsec = xsec * hbar_C * hbar_C * 1e-35; //GeV to cm^2/MeV
   return xsec;
}

// compute the differential cross section ds/dcos(E_nu, cos(theta))
//E_neu in MeV
//result unit: cm^2
//Strumia and Vissani
double IBDXSec_StrumiaVissani::DSigmaDcos(double E_neu,
                                          double Cos_theta)
{
   double xsec(0);
   if (E_neu > E_Lower) //threshold
   {
      double E_nu = E_neu / 1e3; //to GeV
      // double delta_mr;     //delta as (12) in ref :arXiv:astro-ph/0302055v2
      double epsilon_nu2p(E_nu / M_proton);                                         //ratio E_nu/m_P (8) in ref
      double kappa_ec(pow(1 + epsilon_nu2p, 2) - pow(epsilon_nu2p * Cos_theta, 2)); //(21) in ref
      // delta_mr = (M_n_sq - M_p_sq - M_e_sq) / (2. * M_proton);
      double E_nuMD(E_nu - delta_mr); //E_nu -delta
      double E_e((E_nuMD * (1. + epsilon_nu2p) +
                  epsilon_nu2p * Cos_theta * sqrt(E_nuMD * E_nuMD - M_e_sq * kappa_ec)) /
                 kappa_ec);

      // double P_nu(E_nu);                    //in GeV
      double P_e(sqrt(E_e * E_e - M_e_sq)); //in GeV

      double q_sq(M_n_sq - M_p_sq - 2. * M_proton * (E_nu - E_e)); //momentum transfer, which is t (11) in ref
      double sMinusU(2. * M_proton * (E_nu + E_e) - M_e_sq);
      double sMinusMpsq(2. * M_proton * E_nu); //s-m_p^2

      double f1, f2, g1, g2;
      double t4Msq(q_sq / 4. / M_Nuc_sq);
      double fdnom((1 - t4Msq) * pow(1 - q_sq / fMv2, 2));
      f1 = (1 - (1 + fNucleonMMDiff) * t4Msq) / fdnom;
      f2 = fNucleonMMDiff / fdnom;
      g1 = fg1of0 / pow(1 - q_sq / fMa2, 2);
      g2 = 2. * M_Nuc_sq * g1 / (M_pi_sq - q_sq);

      double A_t, B_t, C_t;
      A_t = MAterm(q_sq, f1, f2, g1, g2);
      B_t = MBterm(q_sq, f1, f2, g1, g2);
      C_t = MCterm(q_sq, f1, f2, g1);
      xsec = A_t - (sMinusU)*B_t + pow(sMinusU, 2) * C_t; //Matrix square
      xsec = xsec * G_F_sq * Cos_Cabibbo_sq /
             (2. * PI * pow(sMinusMpsq, 2)); //dsigma/dt

      xsec = xsec * 2. * M_proton; //to dSig/dE
      double Coeff_t2T(P_e * epsilon_nu2p /
                       (1 + epsilon_nu2p * (1 - E_e / P_e * Cos_theta)));
      // Coeff_t2T = P_e * epsilon_nu2p /
      //             (1 + epsilon_nu2p * (1 - E_e / P_e * Cos_theta));
      // printf("Coeff: %.16f \n", Coeff_t2T);
      xsec = xsec * Coeff_t2T; //to dSig/dcos

      // apply correction factors
      double rdcf = RadiativeCorr(E_e);
      // printf("rad: %f\n",rdcf);
      xsec *= rdcf;
   }
   xsec = xsec * hbar_C * hbar_C * 1e-32; //GeV^2 to cm^2
   return xsec;
}

//____________________________________________________________________________
//return total cross section
double IBDXSec_StrumiaVissani::TotalSigma_SV(double E_nu)
{
   double tsigma = 0;
   const int CalNUM = 1000;
   double denC = 2. / CalNUM;
#pragma omp parallel for reduction(+ \
                                   : tsigma)
   for (int i = 0; i < CalNUM; i++)
   {
      double xc = 0;
      xc = -1. + (i + 0.5) * denC;
      tsigma += DSigmaDcos(E_nu, xc) * denC;
   }
   return tsigma;
}

// return A(t) in the matrix element calculation
// from eqn 6 (without the factor 16)
//
// for speed purposes, no check that |t|^2 = t2 is performed
double IBDXSec_StrumiaVissani::MAterm(double t,
                                      double f1,
                                      double f2,
                                      double g1,
                                      double g2)
{
   double tmesq = t - M_e_sq;     //t-m_e^2
   double M_sq4 = 4. * M_Nuc_sq;  //4*M^2
   double t_o_Msq = t / M_Nuc_sq; //t/M^2
   double f1_sq = f1 * f1;
   double f2_sq = f2 * f2;
   double g1_sq = g1 * g1;
   double g2_sq = g2 * g2;

   double r1;
   r1 = 4. * f1_sq * (M_sq4 + t + M_e_sq);
   r1 += 4. * g1_sq * (t + M_e_sq - M_sq4);
   r1 += f2_sq * (t * t_o_Msq + 4. * t + 4. * M_e_sq);
   r1 += 4. * M_e_sq * g2_sq * t_o_Msq;
   r1 += 8. * f1 * f2 * (2. * t + M_e_sq);
   r1 += 16. * M_e_sq * g1 * g2;
   r1 *= tmesq;

   double r2;
   r2 = (4. * f1_sq + f2_sq * t_o_Msq) * (M_sq4 + tmesq);
   r2 += 4. * g1_sq * (M_sq4 - tmesq);
   r2 += 4. * M_e_sq * g2_sq * tmesq / M_Nuc_sq;
   r2 += 8. * f1 * f2 * (2. * t - M_e_sq);
   r2 += 16. * M_e_sq * g1 * g2;
   r2 *= Delta_np_sq;

   double r3;
   r3 = 32. * M_e_sq * M_Nuc * Delta_np * g1 * (f1 + f2);
   return (r1 - r2 - r3) / 16.;
}

// return B(t) in the matrix element calculation
// from eqn 6 (without the factor 16)
double IBDXSec_StrumiaVissani::MBterm(double t,
                                      double f1,
                                      double f2,
                                      double g1,
                                      double g2)
{
   double bterm = 0;
   double f2_sq = f2 * f2;
   bterm = 16. * t * (g1 * (f1 + f2));
   bterm += 4. * M_e_sq * Delta_np * (f2_sq + f1 * f2 + 2. * g1 * g2) / M_Nuc;
   bterm = bterm / 16.;
   return bterm;
}
// return C(t) in the matrix element calculation
// from eqn 6 (without the factor 16)
double IBDXSec_StrumiaVissani::MCterm(double t,
                                      double f1,
                                      double f2,
                                      double g1)
{
   double f1_sq = f1 * f1;
   double f2_sq = f2 * f2;
   double g1_sq = g1 * g1;
   double t_o_Msq = t / M_Nuc_sq; //t/M^2
   double cterm;
   cterm = 4. * (f1_sq + g1_sq);
   cterm -= f2_sq * t_o_Msq;
   cterm = cterm / 16.;
   return cterm;
}
// radiative correction to the cross section. eqn 14 from the reference
// only valid for Ev<<m_p!
double IBDXSec_StrumiaVissani::RadiativeCorr(const double Ee)
{
   double rc = 6.000 + (1.500 * log(M_proton / (2.000 * Ee)));
   rc += 1.200 * pow((M_electron / Ee), 1.500);
   rc *= alpha_EM / PI;
   rc += 1.000;
   return rc;
}

//eq(15) in ref, not necessary for proton scattering
double IBDXSec_StrumiaVissani::FinalStateCorr(const double Ee)
{
   double eta(2 * PI * alpha_EM /
              sqrt(1 - M_e_sq / Ee / Ee));
   double FEe(eta / (1 - exp(-eta)));
   return FEe;
}

//Calculate the neutrino energy from the positron energy
//input: MeV, output MeV
double *IBDXSec_StrumiaVissani::GetE_nu_SV(double E_e, double Cos_theta)
{
   E_e = E_e / 1e3; //to GeV
   double E_e_sq(E_e * E_e);
   double Cos_theta_sq(Cos_theta * Cos_theta);
   double two_a(2. * (Cos_theta_sq * E_e_sq - E_e_sq + 2 * M_proton * E_e - M_p_sq - Cos_theta_sq * M_e_sq));
   double b(M_proton * (2 * delta_mr * M_proton - 2 * delta_mr * E_e + 2 * M_proton * E_e - 2 * E_e_sq));
   double b_sq(b * b);
   double four_a_c(-4. * M_p_sq * (delta_mr * delta_mr + 2. * delta_mr * E_e + E_e_sq) * (two_a / 2.));
   double *E_nu = new double[2];
   E_nu[0] = (-1. * b + sqrt(b_sq - four_a_c)) / two_a * 1e3;
   E_nu[1] = (-1. * b - sqrt(b_sq - four_a_c)) / two_a * 1e3;
   return E_nu;
}

//wrapper for positron energy, result in MeV
//*@ param x: x[0]: Neutrino energy in MeV
//*@ param x: x[1]: Scattering angle cos(theta)
double IBDXSec_StrumiaVissani::Wrap_GetE_e(double *x, double *par)
{
   return GetE_e(x[0], x[1]);
}

//calculate the energy of positron e^+
//E_nu in MeV, result in MeV
//from Strumia and Vissani
double IBDXSec_StrumiaVissani::GetE_e(double E_nu, double Cos_theta)
{
   double E_e(0);
   if (E_nu > E_Lower)
   {
      E_nu = E_nu / 1e3;                    //to GeV
      double epsilon_nu2p(E_nu / M_proton); //ratio E_nu/m_P
      double kappa_ec;                      //(21) in ref
      // epsilon_nu2p = E_nu / M_proton;

      kappa_ec = pow(1 + epsilon_nu2p, 2) - pow(epsilon_nu2p * Cos_theta, 2);

      double E_nuMD(E_nu - delta_mr); //E_nu -delta
      E_e = (E_nuMD * (1. + epsilon_nu2p) +
             epsilon_nu2p * Cos_theta * sqrt(E_nuMD * E_nuMD - M_e_sq * kappa_ec)) /
            kappa_ec;
      // printf("E_e?: %e\n", E_e);
      E_e = E_e * 1e3; //back to MeV
   }
   return E_e;
}

//calculate neutrino energy E_nu from average positron energy E_e
//input: MeV, outout: MeV
//there are two solutions
//first is what we need
double *IBDXSec_StrumiaVissani::GetEnu_SV(double *E_e, double *par)
{
   if (E_e[0] < 0)
      return 0;
   double E_positron(E_e[0] / 1e3); //to GeV
   double b(M_e_sq - M_n_sq - 4 * E_positron * M_proton + 3 * M_p_sq);
   double b_sq(b * b);
   double four_a_c(8 * M_proton * (M_e_sq * M_proton - M_n_sq * M_proton - 2 * E_positron * M_p_sq + M_p_sq * M_proton));
   double sqrt_b_sq_minus_four_a_c(sqrt(b_sq - four_a_c));
   double two_a(4 * M_proton);
   double *E_nus = new double[2];
   E_nus[0] = (-1 * b + sqrt_b_sq_minus_four_a_c) / two_a * 1e3; //to MeV
   E_nus[1] = (-1 * b - sqrt_b_sq_minus_four_a_c) / two_a * 1e3; //to MeV

   return E_nus;
}

//calculate neutrino energy E_nu from average positron energy E_e
//input: MeV, outout: MeV
//there are two solutions
//first is what we need
double *IBDXSec_StrumiaVissani::GetEnu_from_EeAverageSV(double E_e)
{
   if (E_e < 0)
      return 0;
   double E_positron(E_e / 1e3); //to GeV
   double b(M_e_sq - M_n_sq - 4 * E_positron * M_proton + 3 * M_p_sq);
   double b_sq(b * b);
   double four_a_c(8 * M_proton * (M_e_sq * M_proton - M_n_sq * M_proton - 2 * E_positron * M_p_sq + M_p_sq * M_proton));
   double sqrt_b_sq_minus_four_a_c(sqrt(b_sq - four_a_c));
   double two_a(4 * M_proton);
   double *E_nus = new double[2];
   E_nus[0] = (-1 * b + sqrt_b_sq_minus_four_a_c) / two_a * 1e3; //to MeV
   E_nus[1] = (-1 * b - sqrt_b_sq_minus_four_a_c) / two_a * 1e3; //to MeV

   return E_nus;
}

//Calculate Jacobian dEnu/dEe
//in and out unit: MeV
double *IBDXSec_StrumiaVissani::GetDEnu_DEe(double Ee_average)
{
   if (Ee_average < 0)
      return 0;
   double h = 1e-5;
   double *Enu_plus = GetEnu_from_EeAverageSV(Ee_average + h);
   double *Enu = GetEnu_from_EeAverageSV(Ee_average);
   double *derivatives = new double[2];
   derivatives[0] = (Enu_plus[0] - Enu[0]) / h;
   derivatives[1] = (Enu_plus[1] - Enu[1]) / h;
   return derivatives;
}

//Calculate the average positron energy
//in MeV, wrapper for plots
//* @param E_nu, neutrino energy
double IBDXSec_StrumiaVissani::GetE_eAverage_SV(double *E_nu, double *par)
{
   double E_neu(E_nu[0] / 1e3); //to GeV
   double s(M_p_sq + 2 * M_proton * E_neu);
   double TwoSqrt_s(2 * sqrt(s));
   double E_nu_CM((s - M_p_sq) / TwoSqrt_s);
   double E_e_CM((s - M_n_sq + M_e_sq) / TwoSqrt_s);
   double E_e_Avg(E_neu - delta_mr - (E_nu_CM * E_e_CM / M_proton));
   return E_e_Avg * 1e3;
}
//Calculate the average positron energy
//in MeV
//* @param E_nu, neutrino energy
double IBDXSec_StrumiaVissani::GetEe_Average_SV(double E_nu)
{
   double E_neu(E_nu / 1e3); //to GeV
   double s(M_p_sq + 2 * M_proton * E_neu);
   double TwoSqrt_s(2 * sqrt(s));
   double E_nu_CM((s - M_p_sq) / TwoSqrt_s);
   double E_e_CM((s - M_n_sq + M_e_sq) / TwoSqrt_s);
   double E_e_Avg(E_neu - delta_mr - (E_nu_CM * E_e_CM / M_proton));
   return E_e_Avg * 1e3;
}

//return the lower limit of positron energy for a given neutrino energy
//MeV to MeV
double IBDXSec_StrumiaVissani::GetE_eLow_SV(double E_nu)
{
   double E_neu(E_nu / 1e3);
   double s(M_p_sq + 2 * M_proton * E_neu);
   double TwoSqrt_s(2 * sqrt(s));
   double E_nu_CM((s - M_p_sq) / TwoSqrt_s);
   double E_e_CM((s - M_n_sq + M_e_sq) / TwoSqrt_s);
   double p_e_CM(sqrt((s - pow(M_neutron - M_electron, 2)) *
                      (s - pow(M_neutron + M_electron, 2))) /
                 TwoSqrt_s);
   double E_1(E_neu - delta_mr - 1 / M_proton * E_nu_CM * (E_e_CM + p_e_CM));
   return E_1 * 1e3; //to MeV
}

//return higher limit of positron energy for a given neutrino energy
//MeV to MeV
double IBDXSec_StrumiaVissani::GetE_eHigh_SV(double E_nu)
{
   double E_neu(E_nu / 1e3);
   double s(M_p_sq + 2 * M_proton * E_neu);
   double TwoSqrt_s(2 * sqrt(s));
   double E_nu_CM((s - M_p_sq) / TwoSqrt_s);
   double E_e_CM((s - M_n_sq + M_e_sq) / TwoSqrt_s);
   double p_e_CM(sqrt((s - pow(M_neutron - M_electron, 2)) *
                      (s - pow(M_neutron + M_electron, 2))) /
                 TwoSqrt_s);
   double E_2(E_neu - delta_mr - 1 / M_proton * E_nu_CM * (E_e_CM - p_e_CM));
   return E_2 * 1e3; //to MeV
}

//Wrapper for TF2 to plot.
//* @param x: x[0]:neutrino energy Enu in MeV
//* @param x: x[1]:cos(theta)
double IBDXSec_StrumiaVissani::Wdsigdcos_SV(double *x, double *par)
{
   return DSigmaDcos(x[0], x[1]);
}

//return the numerical partial derivative of \partial E_e/\partial E_nu
//from doi:10.1090/S0025-5718-1988-0935077-0
double IBDXSec_StrumiaVissani::GetPE_ePE_nu(double E_nu, double Cos_T)
{
   double f_p(0);
   double i_h[2]; //i and -i
   if (E_nu > E_Lower)
   {
      for (int i = 0; i < PrecisionOrder; i++)
      {
         i_h[0] = E_nu + (i + 1) * h_x;
         i_h[1] = E_nu - (i + 1) * h_x;
         if (i_h[1] < E_Lower)
            i_h[1] = E_Lower;

         // printf("Part?: %e \n", GetE_e(i_h[1], Cos_T));

         f_p += (Coek_i[i] * GetE_e(i_h[0], Cos_T) -
                 Coek_i[i] * GetE_e(i_h[1], Cos_T));
      }
      f_p = f_p / h_x;
   }
   return f_p;
}
//return the numerical partial derivative of \partial E_e/\partial cos\theta
//from doi:10.1090/S0025-5718-1988-0935077-0
double IBDXSec_StrumiaVissani::GetPE_ePcos(double E_nu, double Cos_T)
{
   double f_p(0);
   double i_h[2]; //i and -i
   if (E_nu > E_Lower)
   {
      for (int i = 0; i < PrecisionOrder; i++)
      {
         i_h[0] = Cos_T + (i + 1) * h_x;
         i_h[1] = Cos_T - (i + 1) * h_x;
         if (i_h[0] > 1.)
            i_h[0] = 1;
         if (i_h[1] < -1.)
            i_h[1] = -1.;
         f_p += (Coek_i[i] * GetE_e(E_nu, i_h[0]) -
                 Coek_i[i] * GetE_e(E_nu, i_h[1]));
      }
      f_p = f_p / h_x;
   }
   return f_p;
}
