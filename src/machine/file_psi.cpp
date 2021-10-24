/*
 * Copyright (C) 2021  David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <omp.h>

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <stdexcept>
//
#include <machine/file_psi.hpp>
#include <tools/state.hpp>

using namespace machine;

file_psi::file_psi(lattice::bravais& lattice, const std::string& filename)
    : Base{lattice, 1},
      state_vec_{new Eigen::MatrixXcd((size_t)(1 << lattice.n_total), 1)},
      created_{true} {
    if (lattice.n_total > 64) {
        throw std::runtime_error("File psi not possible for N > 64.");
    }
    std::cout << state_vec_->rows() << ", " << state_vec_->cols() << std::endl;
    std::ifstream file{filename};
    std::complex<double> line;
    size_t c = 0;
    while (file >> line) {
        (*state_vec_)(c) = line;
        c++;
    }
    if (c != (size_t)state_vec_->size())
        throw std::runtime_error("File " + filename + " has wrong size!");
    std::cout << "Loaded state vec with norm: "
              << (state_vec_->transpose().conjugate() * *state_vec_).real()
              << std::endl;
}

file_psi::~file_psi() {
    if (created_) delete state_vec_;
}

file_psi::file_psi(lattice::bravais& lattice, Eigen::MatrixXcd& state)
    : Base{lattice, 1}, state_vec_(&state), created_(false) {}

std::complex<double> file_psi::psi(const spin_state& state, rbm_context&) {
    return (*state_vec_)(state.to_num());
}

std::complex<double> file_psi::psi_over_psi(const spin_state& state,
                                            const std::vector<size_t>& flips,
                                            rbm_context& co, rbm_context&,
                                            bool discard, bool* didupdate) {
    std::complex<double> ps1 = psi(state, co);
    spin_state state2 = state;
    state2.flip(flips);
    return psi(state2, co) / ps1;
}
