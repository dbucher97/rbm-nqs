#pragma once

#include <complex>

typedef std::complex<double> cmplx;

extern "C" {

extern void zminresqlp(int *n, void (*Aprod)(int *, cmplx *, cmplx *),
                       const cmplx *b, double *shift,
                       void (*Msolve)(int *, cmplx *, cmplx *), bool *disable,
                       int *nout, int *itnlim, double *rtol, double *maxnorm,
                       double *trancond, double *Acondlim, cmplx *x, int *istop,
                       int *itn, double *rnorm, double *Arnorm, double *xnorm,
                       double *Anorm, double *Acond);
extern void minresqlp(int *n, void (*Aprod)(int *, double *, double *),
                      const double *b, double *shift,
                      void (*Msolve)(int *, double *, double *), bool *disable,
                      int *nout, int *itnlim, double *rtol, double *maxnorm,
                      double *trancond, double *Acondlim, double *x, int *istop,
                      int *itn, double *rnorm, double *Arnorm, double *xnorm,
                      double *Anorm, double *Acond);
}

inline int minresqlp(int n, void (*Aprod)(int *, cmplx *, cmplx *),
                     const cmplx *b, cmplx *x, double *shift,
                     void (*Msolve)(int *, cmplx *, cmplx *), bool *disable,
                     int *nout, int *itnlim, double *rtol, double *maxnorm,
                     double *trancond, double *Acondlim, int *itn,
                     double *rnorm, double *Arnorm, double *xnorm,
                     double *Anorm, double *Acond) {
    int istop;
    zminresqlp(&n, Aprod, b, shift, Msolve, disable, nout, itnlim, rtol,
               maxnorm, trancond, Acondlim, x, &istop, itn, rnorm, Arnorm,
               xnorm, Anorm, Acond);
    return istop;
}

inline int minresqlp(int n, void (*Aprod)(int *, double *, double *),
                     const double *b, double *x, double *shift,
                     void (*Msolve)(int *, double *, double *), bool *disable,
                     int *nout, int *itnlim, double *rtol, double *maxnorm,
                     double *trancond, double *Acondlim, int *itn,
                     double *rnorm, double *Arnorm, double *xnorm,
                     double *Anorm, double *Acond) {
    int istop;
    minresqlp(&n, Aprod, b, shift, Msolve, disable, nout, itnlim, rtol, maxnorm,
              trancond, Acondlim, x, &istop, itn, rnorm, Arnorm, xnorm, Anorm,
              Acond);
    return istop;
}
