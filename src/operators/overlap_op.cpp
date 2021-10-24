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
#include <cmath>
#include <fstream>
#include <string>
//
#include <operators/overlap_op.hpp>
#include <tools/state.hpp>

using namespace operators;

overlap_op::overlap_op(const std::string& file, size_t n_visible)
    : Base{2},
      state_vec_(static_cast<size_t>(1 << n_visible), 1),
      n_vis_{n_visible} {
    fill_vec(file);
}

void overlap_op::fill_vec(const std::string& file) {
    std::complex<double> x;
    // Open the infile.
    std::ifstream filestream(file);
    for (size_t i = 0; i < static_cast<size_t>(1 << n_vis_); i++) {
        // Read the complex value from infile
        filestream >> x;
        // Store that value in the state vector.
        state_vec_(i) = x;
    }
}

void overlap_op::evaluate(machine::abstract_machine& rbm,
                          const machine::spin_state& state,
                          machine::rbm_context& context) {
    auto& result = get_result_();
    // Get the loaded psi for the state.
    std::complex<double> psi_gs = get_psi(state);
    // Get the RBM psi for the state.
    std::complex<double> psi = rbm.psi(state, context);
    // Calculate the overlap of psi with the state.
    result(0) = psi_gs / psi;
    // evaluate 1/p for normalization of the overlap.
    result(1) = 1 / std::pow(std::abs(psi), 2);
}

std::complex<double> overlap_op::get_psi(const machine::spin_state& s) {
    return state_vec_(s.to_num());
}
