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

#include <omp.h>

#include <Eigen/Dense>
#include <vector>
//
#include <operators/local_op.hpp>
#include <operators/local_op_chain.hpp>

using namespace operators;

local_op_chain::local_op_chain(const std::vector<local_op>& ops)
    : Base{}, ops_{ops} {}

void local_op_chain::evaluate(machine::abstract_machine& rbm,
                              const machine::spin_state& state,
                              machine::rbm_context& context) {}

void local_op_chain::push_back(local_op op) {
    // Push baack wrapper
    ops_.push_back(op);
}

void local_op_chain::pop_back() {
    // Push baack wrapper
    ops_.pop_back();
}

std::vector<base_op*> local_op_chain::get_ops() {
    std::vector<base_op*> ret;
    for (auto& op : ops_) {
        ret.push_back(&op);
    }
    return ret;
}

void local_op_chain::finailize() {
    auto& result = get_result_();
    result.setZero();
    for (auto& op : ops_) {
        result += op.get_result();
    }
}
