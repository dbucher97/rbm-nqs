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

#pragma once

#include <Eigen/Dense>
#include <vector>
//
#include <operators/base_op.hpp>

namespace operators {

/**
 * @brief Derivative operator calculates the derivative of the RBM with respect
 * to the parameters.
 */
class derivative_op : public base_op {
    using Base = base_op;

   public:
    /**
     * @brief Derivative operator constructor
     *
     * @param n_params Number of parameters needed for initialization.
     */
    derivative_op(size_t n_params);

    virtual void evaluate(machine::abstract_machine&, const Eigen::MatrixXcd&,
                          machine::rbm_context&) override;
};

}  // namespace operators
