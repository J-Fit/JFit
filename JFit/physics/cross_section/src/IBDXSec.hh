#ifndef IBDXSec_hh
#define IBDXSec_hh
#include "IBDXSec_StrumiaVissani.hh"
#include "IBDXSec_VogelBeacom.hh"
#include "IBDXSec_VogelBeacom_DYB.hh"
#include <string>
#include <math.h>
#include "omp.h"

// using namespace std;

//two options:
//0 for VogelBeacom, 1 for StrumiaVissani, 2 for VogelBeacom_DYB
class IBDXSec : public IBDXSec_StrumiaVissani, public IBDXSec_VogelBeacom, public IBDXSec_VogelBeacom_DYB
{
public:
    //0 for VogelBeacom, 1 for StrumiaVissani, 2 for VogelBeacom_DYB
    IBDXSec(int XSecOp = 1)
    {
        this->XSecOp = XSecOp;
        fX = 0;
        fW = 0;
        CalGLSamPts();
    }
    //0 for VogelBeacom, 1 for StrumiaVissani, 2 for VogelBeacom_DYB
    void SetXsecModel(int XSecOp = 1) { this->XSecOp = XSecOp; }
    ~IBDXSec()
    {
        delete[] fX;
        delete[] fW;
    }
    void SetSamplePtNUM(int SamPtNUM)
    {
        fNum = SamPtNUM;
        CalGLSamPts();
    }

    //E_nu in MeV
    double Sigma_tot(double E_nu)
    {
        const double a0(0); //(a+b)/2
        const double b0(1); //(b-a)/2
        double St(0);
        if (E_nu > 1.8)
        {
            switch (XSecOp)
            {
            case 0: //Vogel and Beacom Model
            {

#pragma omp parallel for reduction(+ \
                                   : St)
                for (int i = 0; i < fNum; i++)
                {
                    double x = a0 + b0 * fX[i];
                    St += fW[i] * DSigDCosTh(E_nu, x);
                }

                St = St * b0;
                return St;
                break;
            }
            case 1: //Strumia and Vissani model
            {

#pragma omp parallel for reduction(+ \
                                   : St)
                for (int i = 0; i < fNum; i++)
                {
                    double x = a0 + b0 * fX[i];
                    St += fW[i] * DSigmaDcos(E_nu, x);
                }

                St = St * b0;
                return St;
                break;
            }
            case 2: // for VogelBeacom_DYB
            {

#pragma omp parallel for reduction(+ \
                                   : St)
                for (int i = 0; i < fNum; i++)
                {
                    double x = a0 + b0 * fX[i];
                    St += fW[i] * DSigDCosTh_DYB(E_nu, x);
                }

                St = St * b0;
                return St;
                break;
            }
            }
        }
        return St;
    }

private:
    int XSecOp;
    double fEpsRel = 1e-7; // Relative error.
    int fNum = 2000;       // Number of points used in the estimation of the integral.
    double *fX;            // Abscisa of the points used.
    double *fW;            // Weights of the points used.
    void CalGLSamPts()
    {
        // Given the number of sampling points this routine fills the
        // arrays x and w.

        if (fNum <= 0 || fEpsRel <= 0)
            return;

        if (fX == 0)
            delete[] fX;

        if (fW == 0)
            delete[] fW;

        fX = new double[fNum];
        fW = new double[fNum];

        // The roots of symmetric is the interval, so we only have to find half of them
        const unsigned int m = (fNum + 1) / 2;

        double z, pp, p1, p2, p3;

        // Loop over the desired roots
        for (unsigned int i = 0; i < m; i++)
        {
            z = cos(3.14159265358979323846 * (i + 0.75) / (fNum + 0.5));

            // Starting with the above approximation to the i-th root, we enter
            // the main loop of refinement by Newton's method
            do
            {
                p1 = 1.0;
                p2 = 0.0;

                // Loop up the recurrence relation to get the Legendre
                // polynomial evaluated at z
                for (int j = 0; j < fNum; j++)
                {
                    p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
                }
                // p1 is now the desired Legendre polynomial. We next compute pp, its
                // derivative, by a standard relation involving also p2, the polynomial
                // of one lower order
                pp = fNum * (z * p1 - p2) / (z * z - 1.0);
                // Newton's method
                z -= p1 / pp;

            } while (fabs(p1 / pp) > fEpsRel);

            // Put root and its symmetric counterpart
            fX[i] = -z;
            fX[fNum - i - 1] = z;

            // Compute the weight and put its symmetric counterpart
            fW[i] = 2.0 / ((1.0 - z * z) * pp * pp);
            fW[fNum - i - 1] = fW[i];
        }
    }
};

#endif