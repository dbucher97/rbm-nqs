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
#include <lattice/bravais.hpp>
#include <operators/local_op_chain.hpp>

namespace operators {

/**
 * @brief Bond operator, is a operator only acting on a list of `lattice::bonds`
 * between two sites. Each bond can be associated with an operator by the bond
 * type attribute. Bond type is an index to the operators list.
 */
class bond_op : public local_op_chain {
    using Base = local_op_chain;
    std::vector<Eigen::MatrixXcd> bops_;  ///< List of operators.

   public:
    /**
     * @brief Bond operator constructor.
     *
     * @param bonds List of `lattice::bonds`.
     * @param ops List of bond type operators.
     */
    bond_op(const std::vector<lattice::bond>& bonds,
            const std::vector<Eigen::MatrixXcd>& ops);
};
}  // namespace operators
