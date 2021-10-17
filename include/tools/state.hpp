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

// ATTENTION: Deprecated since spin_state

#include <Eigen/Dense>
#include <vector>

namespace tools {

inline size_t state_to_num(const Eigen::MatrixXcd& state) {
    size_t res = 0;
    for (size_t i = 0; i < static_cast<size_t>(state.size()); i++) {
        if (std::real(state(i)) < 0) res += (1 << i);
    }
    return res;
}

/**
 * @brief Convert the state in number (bit) representation into a Eigen
 * Matrix used as input for RBM. 1 -> -1. 0 -> 1..
 *
 * @param state_num The state in bit form.
 * @param state The Eigen Matrix reference, which will be filled.
 */
inline void num_to_state(size_t num, Eigen::MatrixXcd& state) {
    for (size_t i = 0; i < static_cast<size_t>(state.size()); i++) {
        state(i) = ((num >> i) & 1) ? -1 : 1;
    }
}

/**
 * @brief Returns the local quantum state of the selected sites. Since a
 * local psi derived from a z-basis state has only one non-zero entry at
 * index `loc`, returning only `loc` is sufficient.
 *
 * @param state Input state.
 *
 * @return The non-zero index `loc`.
 */
inline size_t state_to_num_loc(const Eigen::MatrixXcd& state,
                               const std::vector<size_t>& acts_on) {
    size_t res = 0;
    for (size_t i = 0; i < acts_on.size(); i++) {
        if (std::real(state(acts_on[i])) < 0) res += (1 << i);
    }
    return res;
}

/**
 * @brief Fills the vector flips with indices where bits of `loc` are 1.
 *
 * @param loc A integer where the sites to flip are 1.
 * @param flips A vector of site indices will be cleared and refilled.
 */
inline void get_flips(size_t num, std::vector<size_t>& flips,
                      std::vector<size_t>& acts_on) {
    flips.clear();
    for (size_t i = 0; i < acts_on.size(); i++) {
        if ((num >> i) & 1) flips.push_back(acts_on[i]);
    }
}

}  // namespace tools
