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
#include <operators/local_op.hpp>
#include <tools/state.hpp>

using namespace operators;

local_op::local_op(const std::vector<size_t>& acts_on, const SparseXcd& op)
    : Base{}, acts_on_{acts_on}, op_{op} {}

void local_op::evaluate(machine::abstract_machine& rbm,
                        const Eigen::MatrixXcd& state,
                        machine::rbm_context& context) {
    typedef Eigen::SparseVector<std::complex<double>> SpVec;
    // Get the result of the current thread
    auto& result = get_result_();
    result.setZero();

    // The next two lines are basicelly <\psi| Op, since |\psi> has only one
    // non-zero element, which is one.
    size_t loc = tools::state_to_num_loc(state, acts_on_);
    SpVec res = op_.row(loc);

    // Initialize the flips vector.
    std::vector<size_t> flips;

    for (SpVec::InnerIterator it(res); it; ++it) {
        if (loc == (size_t)it.index()) {
            // Diagonal operator contribution
            result(0) += it.value();
        } else {
            // Off diagonal elements.
            // Get the flips to get from `loc` to `i` and calculate the
            // `psi_over_psi` local weight.
            tools::get_flips(it.index() ^ loc, flips, acts_on_);
            result(0) += it.value() * rbm.psi_over_psi(state, flips, context);
        }
    }
}
