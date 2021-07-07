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
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
//
#include <operators/store_state.hpp>

using namespace operators;

store_state::store_state(const std::string& filename) : Base{}, file_{} {
    file_.open(filename);
}

void store_state::evaluate(machine::abstract_machine& rbm,
                           const Eigen::MatrixXcd& state,
                           const machine::rbm_context& context) {
    // print state vecotr as 1 <=> +1, 0 <=> -1.
#pragma omp critical
    for (size_t i = 0; i < rbm.n_visible; i++) {
        file_ << (std::real(state(i)) > 0 ? '1' : '0');
    }
    file_ << "\n";
}
