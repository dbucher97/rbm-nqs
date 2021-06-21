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

#include <Eigen/Dense>
#include <functional>
#include <vector>
//
#include <machine/rbm_base.hpp>
#include <operators/derivative_op.hpp>

using namespace operators;

derivative_op::derivative_op(size_t n_params) : Base{n_params} {}

void derivative_op::evaluate(machine::rbm_base& rbm,
                             const Eigen::MatrixXcd& state,
                             const machine::rbm_context& context) {
    auto& result = get_result_();
    // Get the result from `rbm.derivative`
    // This is done to more easily allow for different RBM types
    result = rbm.derivative(state, context);
}
