#ifndef IBDXSec_StrumiaVissani_hh
#define IBDXSec_StrumiaVissani_hh
#include <string>

// using namespace std;

class IBDXSec_StrumiaVissani
{

public:
  IBDXSec_StrumiaVissani();
  ~IBDXSec_StrumiaVissani();

  double DSigmaDEe(double E_nu, double E_e);
  double DSigmaDcos(double E_nu, double Cos_theta);
  double TotalSigma_SV(double E_nu);
  double GetE_e(double E_nu, double Cos_T);
  double Wrap_GetE_e(double *x,double *par);
  double* GetE_nu_SV(double E_e,double Cos_theta);
  double GetE_eAverage_SV(double *E_nu, double *par);
  double GetEe_Average_SV(double E_nu);
  double* GetEnu_SV(double *E_e_average,double *par);
  double *GetEnu_from_EeAverageSV(double E_e_average);
  double *GetDEnu_DEe(double Ee_average);
  double GetE_eLow_SV(double E_nu);
  double GetE_eHigh_SV(double E_nu);
  double Wdsigdcos_SV(double *x, double *par);
  double GetPE_ePE_nu(double E_nu, double Cos_T);
  double GetPE_ePcos(double E_nu, double Cos_T);
  double Wrapper_dsigma_dcos(double *cos_T, double *E_nu)
  {
    return DSigmaDcos(E_nu[0], cos_T[0]);
    // return (DSigmaDcos(E_nu[0], cos_T[0])*cos_T[0]/TotalSigma_SV(E_nu[0]));
  }

private:
  // double GetPE_ePE_nu(double E_nu, double Cos_T);
  // double GetPE_ePcos(double E_nu, double Cos_T);
  double h_x = 1e-5;
  //Central finite difference coefficients from: doi:10.1090/S0025-5718-1988-0935077-0
  //error: to the h_x^8 or to the h_x^4 is enough
  static const int PrecisionOrder = 4 / 2;
  double Coek_i[PrecisionOrder] = {2 / 3., -1 / 12.}; //1,2
  // double Coek_i[PrecisionOrder]={4/5.,-1/5.,4/105.,-1/280.};
  double delta_mr; //delta as (12) in ref :arXiv:astro-ph/0302055v2

  double MAterm(double t,
                double f1,
                double f2,
                double g1,
                double g2);
  double MBterm(double t,
                double f1,
                double f2,
                double g1,
                double g2);
  double MCterm(double t,
                double f1,
                double f2,
                double g1);
  double RadiativeCorr(const double Ee);
  double FinalStateCorr(const double Ee);

  // variables
  // double Cos_Cabibbo;    //  cos(cabibbo)
  double Cos_Cabibbo_sq; //cos^2(cabibbo)
  double fg1of0;         //  axial form factor at q2=0
  double fMa2;           //  axial mass squared
  double fMv2;           //  vector mass squared
  double fNucleonMMDiff; //  nucleon magnetic moment difference

  //my variables:
  double M_electron;  //in GeV
  double M_proton;    //in GeV
  double M_neutron;   //in GeV
  double M_Pion;      //in GeV
  double Delta_np;    //in GeV (m_n - m_p)
  double Delta_np_sq; //in GeV^2 (m_n - m_p)^2
  // double G_F;        //Feimi constant GeV^-2
  double G_F_sq; //Feimi constant  square GeV^-4

  double M_Nuc;                //in GeV^2 (m_P+m_n)/2
  double M_Nuc_sq;             //in GeV^2 ((m_p+m_n)/2)^2
  double M_e_sq;               //in GeV^2 M_e^2
  double M_p_sq;               //in GeV^2 M_p^2
  double M_n_sq;               //in GeV^2 M_n^2
  double M_pi_sq;              //in GeV^2 M_pion^2
  double alpha_EM;             //fine structure constant
  double hbar_C = 197.3269804; //MeV.fm
};

#endif // _SV_QUASIELASTIC_NU_NUCLEON_XSEC_H_
