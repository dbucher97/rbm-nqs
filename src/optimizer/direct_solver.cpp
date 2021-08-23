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
 * */

#include <mpi.h>
#include <omp.h>

#include <iostream>
//
#include <optimizer/direct_solver.hpp>
#include <tools/ini.hpp>
#include <tools/mpi.hpp>

using namespace optimizer;

direct_solver::direct_solver(size_t n, size_t n_neural)
    : Base{n, n_neural}, S(n, n) {}
void direct_solver::solve(const Eigen::MatrixXcd& mat,
                          const Eigen::VectorXcd& d, const double norm,
                          const Eigen::VectorXcd& b, Eigen::MatrixXcd& x,
                          const double r1, const double r2, const double rd,
                          const Eigen::VectorXd& diag) {
    S = mat.conjugate() * mat.transpose() / norm;
    if (mpi::master)
        MPI_Reduce(MPI_IN_PLACE, S.data(), S.size(), MPI_DOUBLE_COMPLEX,
                   MPI_SUM, 0, MPI_COMM_WORLD);
    else
        MPI_Reduce(S.data(), S.data(), S.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, 0,
                   MPI_COMM_WORLD);
    if (mpi::master) {
        int nt = omp_get_num_threads();
        omp_set_num_threads(ini::n_threads);
        S -= d.conjugate() * d.transpose();
        // Add regularization.
        S.diagonal().topRows(n_neural_) *= 1 + r1;
        if (n_ - n_neural_)
            S.diagonal().bottomRows(n_ - n_neural_) *= 1 + r1 + rd;

        S += r2 * diag.cwiseAbs().maxCoeff() *
             Eigen::MatrixXcd::Identity(S.rows(), S.cols());

        x = S.completeOrthogonalDecomposition().solve(b);
        omp_set_num_threads(nt);
    }
    MPI_Bcast(x.data(), x.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}
