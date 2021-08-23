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

namespace optimizer {

class abstract_solver {
   protected:
    size_t n_, n_neural_;

    abstract_solver(size_t n, size_t n_neural) : n_{n}, n_neural_{n_neural} {};

   public:
    virtual void solve(const Eigen::MatrixXcd& mat, const Eigen::VectorXcd& d,
                       const double norm, const Eigen::VectorXcd& b,
                       Eigen::MatrixXcd& x, const double r1, const double r2,
                       const double rd, const Eigen::VectorXd& diag) = 0;
};

}  // namespace optimizer
