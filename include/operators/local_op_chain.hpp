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
#include <operators/local_op.hpp>

namespace operators {

/**
 * @brief A sum of local operators.
 */
class local_op_chain : public base_op {
    using Base = base_op;
    std::vector<local_op> ops_;  ///< The list of local opertors

   public:
    /**
     * @brief Local Operator Chain constructor
     *
     * @param ops A vector of local operators (Default: {}).
     */
    local_op_chain(const std::vector<local_op>& ops = std::vector<local_op>{});

    virtual void evaluate(machine::abstract_machine&, const Eigen::MatrixXcd&,
                          const machine::rbm_context&) override;

    /**
     * @brief Push back another local operator.
     *
     * @param local_op The operator to add to the chain.
     */
    void push_back(local_op);

    /**
     * @brief Pop last local operator.
     */
    void pop_back();
};
}  // namespace operators
