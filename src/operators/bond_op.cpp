/**
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
#include <vector>
//
#include <lattice/bravais.hpp>
#include <operators/bond_op.hpp>
#include <operators/local_op.hpp>

using namespace operators;

bond_op::bond_op(const std::vector<lattice::bond>& bonds,
                 const std::vector<Eigen::MatrixXcd>& ops)
    : Base{}, bops_{} {
    // Copy operators
    for (auto& op : ops) {
        // ADDED TRANSPOSITION HERE
        bops_.push_back(op);
        // bops_.push_back(op);
    }
    // Push every bond with corresponding operator into the local operator
    // chain.
    size_t c = 0;
    for (auto& bond : bonds) {
        // if (c >= 6 && c < 7) {
        push_back({{bond.a, bond.b}, bops_[bond.type]});
        // }
        c++;
    }
}
