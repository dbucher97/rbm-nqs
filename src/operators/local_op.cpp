/*
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
//
#include <machine/rbm_base.hpp>
#include <operators/local_op.hpp>

using namespace operators;

local_op::local_op(const std::vector<size_t>& acts_on, Eigen::MatrixXcd& op)
    : Base{}, acts_on_{acts_on}, op_{op} {}

void local_op::evaluate(machine::rbm_base& rbm, const Eigen::MatrixXcd& state,
                        const Eigen::MatrixXcd& thetas) {
    // Get the result of the current thread
    auto& result = get_result_();
    result.setZero();

    // The next two lines are basicelly Op |\psi>, since |\psi> has only one
    // non-zero element, which is one.
    size_t loc = get_local_psi(state);
    Eigen::MatrixXcd res = op_.col(loc);

    // Initialize the flips vector.
    std::vector<size_t> flips;

    for (size_t i = 0; i < static_cast<size_t>(res.size()); i++) {
        if (std::abs(res(i)) > 1e-12) {
            if (i == loc) {
                // Diagonal operator contribution
                result(0) += res(i);
            } else {
                // Off diagonal elements.
                // Get the flips to get from `loc` to `i` and calculate the
                // `psi_over_psi` local weight.
                get_flips(i ^ loc, flips);
                result(0) += res(i) * rbm.psi_over_psi(state, flips, thetas);
            }
        }
    }
}

size_t local_op::get_local_psi(const Eigen::MatrixXcd& s) {
    size_t loc = 0;
    for (size_t i = 0; i < acts_on_.size(); i++) {
        // set bit `i` of `loc` to one if acted on site `i` is -1.
        loc += ((s(acts_on_[i]) < 0) << i);
    }
    return loc;
}

void local_op::get_flips(size_t loc, std::vector<size_t>& flips) {
    flips.clear();
    for (size_t i = 0; i < acts_on_.size(); i++) {
        // if bit `i` of `loc` is 1, flip acted on site `i`.
        if ((loc >> i & 1)) {
            flips.push_back(acts_on_[i]);
        }
    }
}
