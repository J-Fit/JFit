#include "IBDXSec_VogelBeacom_DYB.hh"
#include "TMath.h"
#include "stdio.h"
#include <iostream>

using namespace std;
using namespace TMath;

IBDXSec_VogelBeacom_DYB *gInverseBeta;

IBDXSec_VogelBeacom_DYB::IBDXSec_VogelBeacom_DYB()
{
  fF = 1;
  //fG = 1.26;
  fG = 1.2701; //g_A/g_V: PDG 2010
  fF2 = 3.706;
  PhaseFactor = 1.71465;
  NucleonMass = (gkMassProton + gkMassNeutron) / 2.;
  //PDG 2011
  //NeutronLifeTime = 881.5 / (1.E-6*Hbar()/Qe()); // MeV^-1
  //PDG 2014
  NeutronLifeTime = 880.3 / (1.E-6 * Hbar() / Qe()); // MeV^-1

  fDelta = gkMassNeutron - gkMassProton;
  fMassEsq = gkMassElectron * gkMassElectron;
  fYsq = (fDelta * fDelta - fMassEsq) / 2;
  fF2Plus3G2 = fF * fF + 3 * fG * fG;
  fF2MinusG2 = fF * fF - fG * fG;
  fSigma0 = 0.0952 / (fF2Plus3G2); // *1e{-42} cm^2, eqn 9

  // fDifferential = new TF1("fDifferential", gKRLDSigmaByDCosTheta, -1, 1, 1);
  // fDifferential->SetParameter(0, 3); // default enu = 3 MeV
  // fHTotalCrossSection = NULL;
}

/// in the calculation of the cross section, this must be called first.
// used in DYB, Vogel & Beacom
Double_t IBDXSec_VogelBeacom_DYB::Ee1(Double_t aEnu, Double_t aCosTheta)
{
  Double_t answer;
  fE0 = Ee0(aEnu);
  if (fE0 <= gkMassElectron)
  { // below threshold
    fE0 = 0;
    fP0 = 0;
    answer = 0;
  }
  else
  {
    // fP0 = TMath::Sqrt(fE0 * fE0 - fMassEsq);
    fP0 = sqrt(fE0 * fE0 - fMassEsq);
    fV0 = fP0 / fE0;
    Double_t sqBracket = 1 - aEnu * (1 - fV0 * aCosTheta) / NucleonMass;
    answer = fE0 * sqBracket - fYsq / NucleonMass;
  }
  if (answer < gkMassElectron)
    return 0;
  return answer;
}

//wrapper for TF2, get the positron energy, in MeV
//* @param x: x[0]:neutrino energy Enu in MeV
//* @param x: x[1]:cos(theta)
double IBDXSec_VogelBeacom_DYB::Wrap_Ee1_DYB(double *x, double *par)
{
  return Ee1(x[0], x[1]);
}

//unit: cm^2
Double_t IBDXSec_VogelBeacom_DYB::Sigma0()
{
  Double_t MeV2J, J2MeV;
  MeV2J = 1.E6 * Qe();
  J2MeV = 1. / MeV2J;
  Double_t Sigma0Total = 2. * Power(Pi() * Hbar() * C() * J2MeV, 2) * 1.E4 / (Power(gkMassElectron, 5) * PhaseFactor * NeutronLifeTime); // MeV^{-2}-->cm^2
  Double_t answer = Sigma0Total / fF2Plus3G2;
  //cout<<"--------"<<Sigma0Total<<endl;
  // return answer * 1e42;
  return answer;
}

Double_t IBDXSec_VogelBeacom_DYB::DSigDCosTh_DYB(Double_t aEnu, Double_t aCosTheta)
{
  fE1 = Ee1(aEnu, aCosTheta);
  Double_t answer;
  if (fE1 < gkMassElectron)
  {
    return 0;
  }
  else
  {
    // Double_t pe1 = TMath::Sqrt(fE1 * fE1 - fMassEsq);
    Double_t pe1 = sqrt(fE1 * fE1 - fMassEsq);
    Double_t ve1 = pe1 / fE1;
    Double_t firstLine = (Sigma0() / 2) * (fF2Plus3G2 + fF2MinusG2 * ve1 * aCosTheta) * fE1 * pe1;
    Double_t secondLine = Sigma0() * GammaTerm(aCosTheta) * fE0 * fP0 / (2 * NucleonMass);
    //Double_t firstLine = (fSigma0/2) * (fF2Plus3G2 + fF2MinusG2*ve1*aCosTheta)* fE1*pe1;
    //Double_t secondLine = fSigma0 * GammaTerm(aCosTheta) * fE0 * fP0 / (2* NucleonMass);
    answer = firstLine - secondLine;
  }
  return answer;
}

//Wrapper for TF2 to plot.
//* @param x: x[0]:neutrino energy Enu in MeV
//* @param x: x[1]:cos(theta)
double IBDXSec_VogelBeacom_DYB::Wrap_DSigDCosTh_DYB(double *x, double *par)
{
  return DSigDCosTh_DYB(x[0], x[1]);
}

Double_t IBDXSec_VogelBeacom_DYB::GammaTerm(Double_t aCosTheta)
{
  Double_t firstLine = (2 * fE0 + fDelta) * (1 - fV0 * aCosTheta) - fMassEsq / fE0;
  firstLine *= (2 * (fF + fF2) * fG);
  Double_t secondLine = fDelta * (1 + fV0 * aCosTheta) + fMassEsq / fE0;
  secondLine *= (fF * fF + fG * fG);
  Double_t thirdLine = fF2Plus3G2 * ((fE0 + fDelta) * (1 - aCosTheta / fV0) - fDelta);
  Double_t fourthLine = (fF * fF - fG * fG) * fV0 * aCosTheta;
  fourthLine *= ((fE0 + fDelta) * (1 - aCosTheta / fV0) - fDelta);

  Double_t answer = firstLine + secondLine + thirdLine + fourthLine;
  return answer;
}

//in cm^2
// Double_t IBDXSec_VogelBeacom_DYB::SigmaTot(Double_t aEnu)
// {
//   //  fDifferential->SetParameter(0, aEnu);
//   Double_t answer;
//   if (aEnu < 0)
//   {
//     Warning("SigmaTot(Double_t aEnu)", "Tried to calculate cross section for -ve nu energy");
//     return 0;
//   }
//   if (aEnu > 15)
//   {
//     // table on precalculated to 15 MeV
//     fDifferential->SetParameter(0, aEnu); // default enu = 3 MeV

//     answer = fDifferential->Integral(-1.0, 1.0, 1e-9);
//     return answer;
//   }
//   if (fHTotalCrossSection == NULL)
//     setupTotalCrossSection();
//   int bin = fHTotalCrossSection->FindBin(aEnu);
//   Double_t answer1 = fHTotalCrossSection->GetBinContent(bin);
//   Double_t e1 = fHTotalCrossSection->GetBinCenter(bin);
//   Double_t answer2, e2;
//   if (bin != 1000)
//   {
//     answer2 = fHTotalCrossSection->GetBinContent(bin + 1);
//     e2 = fHTotalCrossSection->GetBinCenter(bin + 1);
//   }
//   else
//   {
//     answer2 = fHTotalCrossSection->GetBinContent(bin - 1);
//     e2 = fHTotalCrossSection->GetBinCenter(bin - 1);
//   }
//   answer = answer1 + (answer2 - answer1) * (aEnu - e1) / (e2 - e1); // answer1 + slope correction
//   if (answer < 0)
//     answer = 0;
//   return answer;
// }

void IBDXSec_VogelBeacom_DYB::setupTotalCrossSection()
{
  // fHTotalCrossSection = new TH1D("htotalCross", "Total #nu (p,n) e Cross Section", 1000, 0, 15);
  // Double_t enu;
  // for (int i = 1; i <= 1000; i++)
  // {
  //   enu = fHTotalCrossSection->GetBinCenter(i);
  //   fDifferential->SetParameter(0, enu); // default enu = 3 MeV
  //   fHTotalCrossSection->SetBinContent(i, fDifferential->Integral(-1.0, 1.0, 1e-9));
  // }
}

/// takes given prompt energy and turns it into a neutrino energy
/// inverts eqn 11
Double_t IBDXSec_VogelBeacom_DYB::PromptEnergyToNeutrinoEnergy(Double_t aE_prompt)
{
  Double_t ePos = aE_prompt - gkMassElectron; // paper uses total E so only
  // subtract one mass
  Double_t a = -1 / (gkMassProton);
  Double_t b = 1 + fDelta / gkMassProton;
  Double_t c = -fYsq / gkMassProton - ePos - fDelta;
  Double_t answer1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
  Double_t answer2 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
  //  printf("%e\t%e\t%e\t%e\t%e\n", aE_prompt, ePos, answer1, answer2, ePos+fDelta);
  return answer1;
}

// in MeV
Double_t *IBDXSec_VogelBeacom_DYB::GetEnuFromEpos_DYB(Double_t Epos)
{
  Double_t ePos = Epos;
  Double_t a = -1 / (gkMassProton);
  Double_t b = 1 + fDelta / gkMassProton;
  Double_t c = -fYsq / gkMassProton - ePos - fDelta;
  Double_t answer1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
  Double_t answer2 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
  //  printf("%e\t%e\t%e\t%e\t%e\n", aE_prompt, ePos, answer1, answer2, ePos+fDelta);
  double *Enus = new double[2];
  Enus[0] = answer1;
  Enus[1] = answer2;
  return Enus;
}

//Calculate Jacobian dEnu/dEe
//in and out unit: MeV
double *IBDXSec_VogelBeacom_DYB::GetDEnu_DEe_DYB(double Ee)
{
  if (Ee < gkMassElectron)
    return 0;
  double h = 1e-5;
  double *Enu_plus = GetEnuFromEpos_DYB(Ee + h);
  double *Enu = GetEnuFromEpos_DYB(Ee);
  double *derivatives = new double[2];
  derivatives[0] = (Enu_plus[0] - Enu[0]) / h;
  derivatives[1] = (Enu_plus[1] - Enu[1]) / h;
  return derivatives;
}

// ///global function to allow calls from a TF1.
// ///This is necessary to use the gaussian quadrature method
// ///built into TF1
// /// dsig/dcos(theta)
// Double_t gKRLDSigmaByDCosTheta(Double_t *x, Double_t *a)
// {
//   Double_t cosTheta = x[0];
//   Double_t enu = a[0];
//   if (gInverseBeta == NULL)
//     gInverseBeta = new IBDXSec_VogelBeacom_DYB();
//   return gInverseBeta->DSigDCosTh_DYB(enu, cosTheta);
// }

// //a global function to allow the total cross section call be put into a TF1
// // a redirect to IBDXSec_VogelBeacom_DYB::SigmaTot()
// Double_t gKRLSigmaTotal(Double_t *x, Double_t *a)
// {
//   // a not used
//   // x[0] = neutrino energy (MeV)
//   if (gInverseBeta == NULL)
//     gInverseBeta = new IBDXSec_VogelBeacom_DYB();
//   return gInverseBeta->SigmaTot(x[0]);
// }
