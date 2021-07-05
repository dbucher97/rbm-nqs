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

extern std::complex<double>* g_mat;
extern std::complex<double>* g_vec;
extern std::complex<double>* g_tmp;
extern std::complex<double>* g_dot;
extern std::complex<double>* g_diag;
extern double g_norm;
extern double* g_reg;
extern int g_mat_dim2, g_nn;

extern void Aprod(int* n, std::complex<double>* x, std::complex<double>* y);
extern void Msolve(int* n, std::complex<double>* x, std::complex<double>* y);

class minresqlp_adapter {
    using VectorXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
    int n;
    int istop;
    int itn;
    double rnorm;
    double Arnorm;
    double xnorm;
    double Anorm;
    double Acond;

    std::complex<double> dot;

    Eigen::MatrixXcd& mat;
    Eigen::MatrixXcd& vec;
    VectorXcd diag;
    VectorXcd tmp;
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

    minresqlp_adapter(Eigen::MatrixXcd& mat, Eigen::MatrixXcd& vec, double e1,
                      double e2, double de, double norm, int nn);

    int apply(VectorXcd& b, Eigen::MatrixXcd& x);

    const int getIstop() { return istop; }
    const int getItn() { return itn; }
    const double getRnorm() { return rnorm; }
    const double getArnorm() { return Arnorm; }
    const double getXnorm() { return xnorm; }
    const double getAnorm() { return Anorm; }
    const double getAcond() { return Acond; }
};

}  // namespace optimizer
