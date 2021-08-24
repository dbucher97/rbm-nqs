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

//
#include <optimizer/abstract_solver.hpp>

namespace optimizer {

class minres_solver : public abstract_solver {
    using Base = abstract_solver;

    size_t max_iterations_;
    double rtol_;
    Eigen::MatrixXcd mat_;
    Eigen::VectorXcd tmp_;
    Eigen::VectorXcd diag_;
    Eigen::MatrixXcd vec_;
    int* n_samples;
    int* n_offsets;

   public:
    minres_solver(size_t n, size_t m, int mloc, size_t n_neural,
                  size_t max_iterations = 0, double r_tol = 0);
    ~minres_solver();

    void solve(const Eigen::MatrixXcd& mat, const Eigen::VectorXcd& d,
               const double norm, const Eigen::VectorXcd& b,
               Eigen::VectorXcd& x, const double r1, const double r2,
               const double rd, const Eigen::VectorXd& diag) override;
};

}  // namespace optimizer
