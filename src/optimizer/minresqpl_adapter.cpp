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

#ifdef ITERATIVE_USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif
#include <omp.h>
//
#include <iostream>
#include <minresqlp.hpp>
#include <optimizer/minres_adapter.hpp>

namespace optimizer {

const std::complex<double>* g_mat = 0;
const std::complex<double>* g_vec = 0;
std::complex<double>* g_tmp = 0;
std::complex<double>* g_dot = 0;
const std::complex<double>* g_diag = 0;
double g_norm = 0;
double* g_reg = 0;
int g_mat_dim2 = 0, g_nn = 0;

void Aprod(int* n, std::complex<double>* x, std::complex<double>* y) {
    // Missing * factor
    cblas_zgemv(CblasColMajor, CblasTrans, *n, g_mat_dim2, &g_norm, g_mat, *n,
                x, 1, &g_zero, g_tmp, 1);
    // cblas_zgemv(CblasColMajor, CblasConjNoTrans, *n, g_mat_dim2, &g_one,
    // g_mat,
    //             *n, g_tmp, 1, &g_zero, y, 1);
    cblas_zgemv(CblasRowMajor, CblasConjTrans, g_mat_dim2, *n, &g_one, g_mat,
                *n, g_tmp, 1, &g_zero, y, 1);

    cblas_zdotc_sub(*n, g_vec, 1, x, 1, g_dot);
    (*g_dot) = -(*g_dot);
    cblas_zaxpy(*n, g_dot, g_vec, 1, y, 1);

    // diagonal scaling
#pragma omp parallel for
    for (int i = 0; i < g_nn; i++) {
        y[i] += g_reg[0] * g_diag[i] * x[i];
    }
#pragma omp parallel for
    for (int i = g_nn; i < *n; i++) {
        y[i] += g_reg[1] * g_diag[i] * x[i];
    }
}

void Msolve(int* n, std::complex<double>* x, std::complex<double>* y) {
#pragma omp parallel for
    for (int i = 0; i < *n; i++) {
        y[i] = x[i] / g_diag[i];
    }
}
}  // namespace optimizer

using namespace optimizer;

minresqlp_adapter::minresqlp_adapter(const Eigen::MatrixXcd& mat,
                                     const Eigen::MatrixXcd& vec,
                                     const double e1, const double e2,
                                     const double de, const double norm,
                                     const int nn, const Eigen::VectorXcd& diag,
                                     Eigen::VectorXcd& tmp)
    : n{static_cast<int>(mat.rows())},
      mat{mat},
      vec{vec},
      diag{diag},
      tmp{tmp} {
    itnlim = 4 * n;
    // diag = mat.cwiseAbs2().rowwise().sum() / norm - vec.cwiseAbs2();
    reg[0] = e1;
    reg[1] = e1 + de;
    g_diag = diag.data();
    g_mat = mat.data();
    g_vec = vec.data();
    g_tmp = tmp.data();
    g_mat_dim2 = mat.cols();
    g_nn = nn;
    g_dot = &dot;
    g_reg = reg;
    g_norm = 1 / norm;
    shift -= e2 * diag.real().maxCoeff();
}

int minresqlp_adapter::apply(const VectorXcd& b, Eigen::MatrixXcd& x) {
    istop =
        minresqlp(n, Aprod, b.data(), x.data(), &shift, precond ? Msolve : 0,
                  &disable, 0, &itnlim, &rtol, &maxnorm, &trancond, &Acondlim,
                  &itn, &rnorm, &Arnorm, &xnorm, &Anorm, &Acond);
    std::cout << itn << ", " << rnorm << ", " << Arnorm << ", " << xnorm << ", "
              << Anorm << ", " << Acond << std::endl;
    return istop;
}
