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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
//
#include <machine/rbm.hpp>
#include <operators/local_op.hpp>

using namespace operators;

local_op::local_op(size_t n_total, const std::vector<size_t>& acts_on,
                   Eigen::MatrixXcd& op)
    : Base{}, acts_on_{acts_on}, op_{op} {}

void local_op::evaluate(machine::rbm& rbm, const Eigen::MatrixXcd& state,
                        const Eigen::MatrixXcd& thetas) {
    result_.setZero();
    size_t loc = get_local_psi(state);

    Eigen::MatrixXcd res = op_.col(loc);

    // std::cout << state.transpose() << " ";
    // for (auto& a : acts_on_) {
    //     std::cout << a << ",";
    // }
    // std::cout << " " << loc << " " << res.transpose() << " ";

    for (size_t i = 0; i < static_cast<size_t>(res.size()); i++) {
        if (std::abs(res(i)) > 1e-12) {
            // std::cout << ((i == loc) ? "true" : "false") << " ";
            if (i == loc) {
                result_(0) += res(i);
            } else {
                // std::cout << i << ";" << loc << "=" << (size_t)(i ^ loc)
                //           << " ...";
                get_flips(i ^ loc);
                // for (auto& f : flips_) {
                //     std::cout << f << ",";
                // }
                result_(0) += res(i) * rbm.psi_over_psi(state, flips_, thetas);
            }
        }
    }
    // std::cout << std::endl;
}

size_t local_op::get_local_psi(const Eigen::MatrixXcd& s) {
    size_t loc = 0;
    for (size_t i = 0; i < acts_on_.size(); i++) {
        loc += ((std::abs(s(acts_on_[i]) - 1.) > 1e-12) << i);
    }
    return loc;
}

void local_op::get_flips(size_t loc) {
    flips_.clear();
    for (size_t i = 0; i < acts_on_.size(); i++) {
        if ((loc >> i & 1)) {
            flips_.push_back(acts_on_[i]);
        }
    }
}
