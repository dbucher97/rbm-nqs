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
#include <tools/state.hpp>

using namespace operators;

local_op::local_op(const std::vector<size_t>& acts_on, Eigen::MatrixXcd& op)
    : Base{}, acts_on_{acts_on}, op_{op} {}

void local_op::evaluate(machine::rbm_base& rbm, const Eigen::MatrixXcd& state,
                        const Eigen::MatrixXcd& thetas) {
    // Get the result of the current thread
    auto& result = get_result_();
    result.setZero();

    // The next two lines are basicelly <\psi| Op, since |\psi> has only one
    // non-zero element, which is one.
    size_t loc = tools::state_to_num_loc(state, acts_on_);
    Eigen::MatrixXcd res = op_.row(loc);

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
                tools::get_flips(i ^ loc, flips, acts_on_);
                result(0) +=
#ifndef ALT_POP
                    res(i) * rbm.psi_over_psi(state, flips, thetas);
#else
                    res(i) * rbm.psi_over_psi_alt(state, flips, thetas);
#endif
            }
        }
    }
}
