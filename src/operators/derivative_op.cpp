/**
 * src/operators/derivative_op.cpp
 * Copyright (c) 2021 David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <Eigen/Dense>
#include <functional>
#include <vector>
//
#include <machine/rbm.hpp>
#include <operators/derivative_op.hpp>

using namespace operators;

derivative_op::derivative_op(size_t n_alpha, size_t n_sym)
    : Base{1 + n_alpha * (1 + n_sym)} {}

void derivative_op::evaluate(machine::rbm& rbm, const Eigen::MatrixXcd& state,
                             const Eigen::MatrixXcd& thetas) {
    auto& result = get_result_();
    result.setZero();
    result(0) = state.sum();
    Eigen::MatrixXcd tanh = thetas.array().tanh();
    result.block(1, 0, rbm.n_alpha, 1) = tanh.rowwise().sum();
    auto& symm = rbm.get_symmetry();
    size_t n_tot = rbm.n_visible * rbm.n_alpha;
    for (size_t s = 0; s < symm.size(); s++) {
        Eigen::MatrixXcd x = (symm[s] * state) * tanh.col(s).transpose();
        result.block(1 + rbm.n_alpha, 0, n_tot, 1) +=
            Eigen::Map<Eigen::MatrixXcd>(x.data(), n_tot, 1);
    }
}
