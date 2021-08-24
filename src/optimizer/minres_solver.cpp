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
#include <optimizer/minres_adapter.hpp>
#include <optimizer/minres_solver.hpp>
#include <tools/ini.hpp>
#include <tools/mpi.hpp>

using namespace optimizer;

minres_solver::minres_solver(size_t n, size_t m, int mloc, size_t n_neural,
                             size_t max_iterations, double rtol)
    : Base{n, n_neural}, max_iterations_{max_iterations}, rtol_{rtol} {
    if (mpi::master) {
        mat_ = Eigen::MatrixXcd(n, m);
        tmp_ = Eigen::VectorXcd(m);
    }
    n_samples = new int[mpi::n_proc];
    n_offsets = new int[mpi::n_proc];
    mloc *= n;
    MPI_Allgather(&mloc, 1, MPI_INT, n_samples, 1, MPI_INT, MPI_COMM_WORLD);
    size_t start = 0;
    for (int i = 0; i < mpi::n_proc; i++) {
        n_offsets[i] = start;
        start += n_samples[i];
    }
}
minres_solver::~minres_solver() {
    delete[] n_samples;
    delete[] n_offsets;
}

void minres_solver::solve(const Eigen::MatrixXcd& mat,
                          const Eigen::VectorXcd& d, const double norm,
                          const Eigen::VectorXcd& b, Eigen::MatrixXcd& x,
                          const double r1, const double r2, const double rd,
                          const Eigen::VectorXd& diag) {
    MPI_Gatherv(mat.data(), mat.size(), MPI_DOUBLE_COMPLEX, mat_.data(),
                n_samples, n_offsets, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    if (mpi::master) {
        // int nt = omp_get_num_threads();
        // omp_set_num_threads(ini::n_threads);
        Eigen::MatrixXcd A(mat_.rows(), mat_.rows());
        A = mat_.conjugate() * mat_.transpose() / norm;
        A -= d.conjugate() * d.transpose();
        Eigen::VectorXcd diag_ = diag.cast<std::complex<double>>();
        std::cout << std::endl
                  << (A.adjoint() - A).cwiseAbs().mean() << std::endl;
        std::cout << (A.diagonal() - diag_).norm() << std::endl;
        minresqlp_adapter min(mat_, d, r1, r2, rd, norm, n_neural_, diag_,
                              tmp_);
        // if (max_iterations_) min.itnlim = max_iterations_;
        // if (rtol_ > 0.0) min.rtol = rtol_;
        min.apply(b, x);
        // omp_set_num_threads(nt);
        std::cout << "Acond: " << min.getAcond() << std::endl;
        std::cout << "Rnorm: " << min.getRnorm() << std::endl;
        std::cout << "Rtol: " << min.rtol << std::endl;
        std::cout << "Itn: " << min.getItn() << std::endl;
        std::cout << "Istop: " << min.getIstop() << std::endl;
    }

    MPI_Bcast(x.data(), x.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}
