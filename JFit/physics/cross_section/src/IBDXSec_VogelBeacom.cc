#include "IBDXSec_VogelBeacom.hh"
#include "JunoARCon.hh"
#include <TMath.h>
#include <TF1.h>
#include <TH1.h>
#include <iostream>
#include <string>
#include <cmath>
#include <math.h>
#include <fstream>
#include <vector>

// IBDXSec_VogelBeacom *gjunoIBDCrossSection = (IBDXSec_VogelBeacom *)0;

IBDXSec_VogelBeacom::IBDXSec_VogelBeacom()
{
  fF = 1;
  fG = 1.26;
  // fG = 1.2762; // PDG2020
  fF2 = 3.706;
  fDelta = gkMassNeutron - gkMassProton;
  fMassEsq = gkMassElectron * gkMassElectron;
  fYsq = (fDelta * fDelta - fMassEsq) / 2;
  fF2Plus3G2 = fF * fF + 3 * fG * fG;
  fF2MinusG2 = fF * fF - fG * fG;
  fSigma0 = 0.0952 / (fF2Plus3G2)*1E-42; // *10^{-42} cm^2, eqn 9

  // fDifferentialXSection = new TF1("fDifferentialXSection", gDSigmaDCosTheta,
  //                                 -1, 1, 1);
  // fXSectionFcn = new TF1("fXSectionFcn", gXSection, 0, 12, 0);
}

IBDXSec_VogelBeacom::IBDXSec_VogelBeacom(const char *filename)
{
  ;
}

IBDXSec_VogelBeacom::~IBDXSec_VogelBeacom()
{
  // delete fXSectionFcn;
  // delete fDifferentialXSection;
}

double IBDXSec_VogelBeacom::Ee1(double aEnu, double aCosTheta)
{
  double answer;
  fE0 = Ee0(aEnu);
  if (fE0 <= gkMassElectron)
  {
    fE0 = 0;
    fP0 = 0;
    answer = 0;
  }
  else
  {
    fP0 = sqrt(fE0 * fE0 - fMassEsq);
    fV0 = fP0 / fE0;
    double sqBracket = 1 - aEnu * (1 - fV0 * aCosTheta) / gkMassProton;
    answer = fE0 * sqBracket - fYsq / gkMassProton;
  }
  if (answer < gkMassElectron)
    return 0;
  return answer;
}

//return dsigma/dcos
//E_nu in MeV, result unit: cm^2
double IBDXSec_VogelBeacom::DSigDCosTh(double aEnu, double aCosTheta)
{
  double answer;
  fE1 = Ee1(aEnu, aCosTheta);
  if (fE1 < gkMassElectron)
    return 0;
  else
  {
    double pe1 = sqrt(fE1 * fE1 - fMassEsq);
    double ve1 = pe1 / fE1;
    double firstLine = (fSigma0 / 2) * (fF2Plus3G2 + fF2MinusG2 * ve1 * aCosTheta) * fE1 * pe1;
    double secondLine = fSigma0 * GammaTerm(aCosTheta) * fE0 * fP0 / (2 * gkMassProton);
    answer = firstLine - secondLine;
  }
  return answer;
}

double IBDXSec_VogelBeacom::GammaTerm(double aCosTheta)
{
  double firstLine = (2 * fE0 + fDelta) * (1 - fV0 * aCosTheta) - fMassEsq / fE0;
  firstLine *= (2 * (fF + fF2) * fG);
  double secondLine = fDelta * (1 + fV0 * aCosTheta) + fMassEsq / fE0;
  secondLine *= (fF * fF + fG * fG);
  double thirdLine = fF2Plus3G2 * ((fE0 + fDelta) * (1 - aCosTheta / fV0) - fDelta);
  double fourthLine = (fF * fF - fG * fG) * fV0 * aCosTheta;
  fourthLine *= ((fE0 + fDelta) * (1 - aCosTheta / fV0) - fDelta);

  double answer = firstLine + secondLine + thirdLine + fourthLine;
  return answer;
}

//total cross section?
double IBDXSec_VogelBeacom::XSection(double E_nu)
{
  double tsigma = 0;
  const int CalNUM = 1e4;
  double denC = 2. / CalNUM;
#pragma omp parallel for reduction(+ \
                                   : tsigma)
  for (int i = 0; i < CalNUM; i++)
  {
    double xc = 0;
    xc = -1. + i * denC;
    tsigma += DSigDCosTh(E_nu, xc) * denC;
  }
  return tsigma;
}

//wrapper for TF2 plot.
//x:E_nu(MeV), cos(theta)
double IBDXSec_VogelBeacom::Wdsigmadcos(double *x, double *par)
{
  return DSigDCosTh(x[0], x[1]);
}

//get average positron energy from neutrino energy,
//from Vogel, Beacom
// unit MeV
double IBDXSec_VogelBeacom::GetEeAverage_VB(double Enu)
{

  double M_diff = fDelta - Enu + (fDelta * fDelta - gkMassElectron * gkMassElectron) / 2. / gkMassProton;
  if (M_diff > 0)
    return 0;
  double E_pos = (sqrt(gkMassNeutron * gkMassNeutron - 4.0 * gkMassProton * M_diff) - gkMassNeutron) / 2.;
  if (E_pos < gkMassElectron)
    return 0;
  return E_pos;
}