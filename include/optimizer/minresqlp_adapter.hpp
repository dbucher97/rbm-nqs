/*
 * Copyright (C) 2021  David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <Eigen/Dense>
#include <complex>
//

namespace optimizer {

const std::complex<double> g_one = 1., g_zero = 0., g_mone = -1.;

extern const std::complex<double>* g_mat;
extern const std::complex<double>* g_vec;
extern std::complex<double>* g_tmp;
extern std::complex<double>* g_dot;
extern const std::complex<double>* g_diag;
extern double g_norm;
extern double* g_reg;
extern int g_mat_dim2, g_nn;

extern void Aprod(int* n, std::complex<double>* x, std::complex<double>* y);
extern void Msolve(int* n, std::complex<double>* x, std::complex<double>* y);

class minresqlp_adapter {
    int n;
    int istop;
    int itn;
    double rnorm;
    double Arnorm;
    double xnorm;
    double Anorm;
    double Acond;

    std::complex<double> dot;

    const Eigen::MatrixXcd& mat;
    const Eigen::MatrixXcd& vec;
    const Eigen::VectorXcd& diag;
    Eigen::VectorXcd& tmp;
    double reg[3];

   public:
    double shift = 0.;
    bool disable = false;
    int itnlim = 0;
    double rtol = 1e-16;
    double maxnorm = 1e7;
    double trancond = 1e7;
    double Acondlim = 1e15;
    bool precond = false;

    minresqlp_adapter(const Eigen::MatrixXcd& mat, const Eigen::MatrixXcd& vec,
                      const double e1, const double e2, const double de,
                      const double norm, const int nn,
                      const Eigen::VectorXcd& diag, Eigen::VectorXcd& tmp);

    int apply(const Eigen::VectorXcd& b, Eigen::VectorXcd& x);

    const int getIstop() { return istop; }
    const int getItn() { return itn; }
    const double getRnorm() { return rnorm; }
    const double getArnorm() { return Arnorm; }
    const double getXnorm() { return xnorm; }
    const double getAnorm() { return Anorm; }
    const double getAcond() { return Acond; }
};

}  // namespace optimizer
