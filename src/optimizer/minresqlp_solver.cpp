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
#include <optimizer/minresqlp_adapter.hpp>
#include <optimizer/minresqlp_solver.hpp>
#include <tools/ini.hpp>
#include <tools/mpi.hpp>

using namespace optimizer;

minresqlp_solver::minresqlp_solver(size_t n, size_t m, int mloc,
                                   size_t n_neural, size_t max_iterations,
                                   double rtol)
    : Base{n, n_neural}, max_iterations_{max_iterations}, rtol_{rtol} {
    if (mpi::master) {
        mat_ = Eigen::MatrixXcd(n, m);
        tmp_ = Eigen::VectorXcd(m);
        diag_ = Eigen::VectorXcd(n);
        vec_ = Eigen::MatrixXcd(n, 1);
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
minresqlp_solver::~minresqlp_solver() {
    delete[] n_samples;
    delete[] n_offsets;
}

void minresqlp_solver::solve(const Eigen::MatrixXcd& mat,
                             const Eigen::VectorXcd& d, const double norm,
                             const Eigen::VectorXcd& b, Eigen::VectorXcd& x,
                             const double r1, const double r2, const double rd,
                             const Eigen::VectorXd& diag) {
    MPI_Gatherv(mat.data(), mat.size(), MPI_DOUBLE_COMPLEX, mat_.data(),
                n_samples, n_offsets, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    if (mpi::master) {
        int nt = omp_get_max_threads();
        omp_set_num_threads(ini::n_threads);

        Eigen::MatrixXcd S = mat.conjugate() * mat.transpose() / norm -
                             d.conjugate() * d.transpose();

        diag_ = diag.cast<std::complex<double>>();
        vec_ = d.conjugate();

        minresqlp_adapter min(mat_, vec_, r1, r2, rd, norm, n_neural_, diag_,
                              tmp_);

        Eigen::VectorXcd tr = Eigen::VectorXcd::Ones(mat.rows());
        Eigen::VectorXcd res(mat.rows());
        int n = mat.rows();
        Aprod(&n, tr.data(), res.data());
        std::cout << (res - S * tr).norm() << std::endl;

        if (max_iterations_) min.itnlim = max_iterations_;
        if (rtol_ > 0.0) min.rtol = rtol_;
        min.apply(b, x);

        std::cout << "Istop: " << min.getIstop() << " It: " << min.getItn()
                  << " Rtol: " << min.getRnorm() << " Acond: " << min.getAcond()
                  << std::endl;

        omp_set_num_threads(nt);
    }

    MPI_Bcast(x.data(), x.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}
