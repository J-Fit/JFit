// Author: Chris Jillings 10/18/2004

/********************************************************************
 * Cross section for nu_e bar + p -> n + e^+
 * Formulae taken from Vogel and Beacom, PRD 60, 053003
 ********************************************************************/

#ifndef __KRLINVERSEBETA__H__
#define __KRLINVERSEBETA__H__

#include "TObject.h"
#include "TF1.h"
#include "TH1.h"

class IBDXSec_VogelBeacom_DYB : public TObject
{
private:
  const Double_t gkMassProton = 938.27203;    // MeV
  const Double_t gkMassNeutron = 939.56536;   // MeV
  const Double_t gkMassElectron = 0.51099892; // MeV
  Double_t fEpos;                             // positron energy (MeV)
  Double_t fEnu;                              // anti neutrino energy (MeV)
  Double_t fSigma0;

  //utility constants set in constructor...
  Double_t fF;       // hadronic weak interaction constants
  Double_t fG;       // hadronic weak interaction constants
  Double_t fF2;      // hadronic weak interaction constants
  Double_t fCosCab;  // cosine of Caibibo angle
  Double_t fDelta;   // M_n - M_p (MeV)
  Double_t fYsq;     // y defined in text below eqn 11
  Double_t fMassEsq; // electron mass squared (MeV^2)
  Double_t fF2Plus3G2;
  Double_t fF2MinusG2;

  Double_t NeutronLifeTime;
  Double_t PhaseFactor;
  Double_t NucleonMass;

  // variables that are here to save calculation
  Double_t fE0; //set in Ee1()
  Double_t fP0; //set in Ee1();
  Double_t fV0; //set in Ee1();
  Double_t fE1; //set in DSigDCosTh_DYB

  TH1D *fHTotalCrossSection;

public:
  IBDXSec_VogelBeacom_DYB();
  virtual ~IBDXSec_VogelBeacom_DYB() { ; }

  Double_t Ee0(Double_t aEnu) { return (aEnu - fDelta); } // eqn 6 in Vogel/Beacom
  Double_t Ee1(Double_t aEnu, Double_t aCosTheta);        // eqn 11
  double Wrap_Ee1_DYB(double *x, double *par);
  Double_t Sigma0();                                          // eqn 9,10
  Double_t DSigDCosTh_DYB(Double_t aEnu, Double_t aCosTheta); // eqn 12
  double Wrap_DSigDCosTh_DYB(double *x, double *par);
  Double_t GammaTerm(Double_t aCosTheta); // eqn 13
  // Double_t SigmaTot(Double_t aEnu); // integration of eqn 12 by Gaussian quadrature
  Double_t PromptEnergyToNeutrinoEnergy(Double_t aEprompt); // incles 2*.511
  Double_t *GetEnuFromEpos_DYB(Double_t Epos);
  double *GetDEnu_DEe_DYB(double Ee);
  // TF1* fDifferential; //! here for easy integration of diff cross section

private:
  void setupTotalCrossSection();
};

// ///global function to allow calls from a TF1.
// ///This is necessary to use the gaussian quadrature method
// ///built into TF1
// /// dsig/dcos(theta)
// Double_t gKRLDSigmaByDCosTheta(Double_t* x, Double_t*a);

// ///global function to allow calls from a TF1.
// /// a redirect to IBDXSec_VogelBeacom_DYB::SigmaTot()
// Double_t gKRLSigmaTotal(Double_t* x, Double_t*a);

// R__EXTERN IBDXSec_VogelBeacom_DYB* gInverseBeta;

#endif
