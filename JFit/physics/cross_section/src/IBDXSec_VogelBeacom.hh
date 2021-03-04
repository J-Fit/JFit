#ifndef IBDXSec_VogelBeacom_hh
#define IBDXSec_VogelBeacom_hh
//based on arXiv:hep-ph/9903554v1
//from zhanl@ihep.ac.cn

#include <vector>

// using namespace std;

class TF1;
class TH1D;

class IBDXSec_VogelBeacom
{
public:
  IBDXSec_VogelBeacom();
  IBDXSec_VogelBeacom(const char *filename);
  ~IBDXSec_VogelBeacom();
  double DSigDCosTh(double aEnu, double aCosTheta);
  double XSection(double aEnu);
  double Wdsigmadcos(double *x, double *par);
  double GetEeAverage_VB(double Enu);

private:
  // equations in Vogel formula, stolen from InverseBeta generator.
  double Ee0(double aEnu) { return (aEnu - fDelta); }
  double Ee1(double aEnu, double aCosTheta);
  double GammaTerm(double aCosTheta);
  const double gkMassProton = 938.27208816;    // MeV
  const double gkMassNeutron = 939.56542052;   // MeV
  const double gkMassElectron = 0.51099895; // MeV
  // utility constants in Vogel formula, stolen from InverseBeta generator.
  double fF, fG, fF2;
  double fCosCab;
  double fDelta;
  double fYsq;
  double fMassEsq;
  double fSigma0;
  double fF2Plus3G2, fF2MinusG2;
  int Npoints;

  // variables that are here to save calculation
  double fE0; //set in Ee1()
  double fP0; //set in Ee1();
  double fV0; //set in Ee1();
  double fE1; //set in DSigDCosTh
};

#endif
