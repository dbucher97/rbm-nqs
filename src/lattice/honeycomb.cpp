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
#include <iostream>
#include <string>
#include <vector>
//
#include <lattice/honeycomb.hpp>

using namespace lattice;

// The Honeycomb lattice is a 2 dimensional bravais lattice with 2 basis sites
// and a coordination of three.
honeycomb::honeycomb(size_t n_uc) : Base{n_uc, 2, 2, 3} {}

std::vector<size_t> honeycomb::nns(size_t i) const {
    // Get the Honeycomb NNs. different positions for the two basis indices.
    // Order: x-bond, y-bond, z-bond.
    size_t uc = uc_idx(i);
    if (b_idx(i) == 0) {
        return {idx(uc, 1), idx(down(uc), 1), idx(down(uc, 1), 1)};
    } else {
        return {idx(uc, 0), idx(up(uc), 0), idx(up(uc, 1), 0)};
    }
}

std::vector<bond> honeycomb::get_bonds() const {
    std::vector<bond> vec;
    // Get all bonds by iterating ove the unitcells and get all NNs of basis
    // index one. Assing each bond the type 0 for x, 1 for y and 2 for z.
    for (size_t i = 0; i < n_total_uc; i++) {
        auto nn = nns(idx(i, 0));
        for (size_t c = 0; c < n_coordination; c++)
            vec.push_back({idx(i, 0), nn[c], c});
    }
    return vec;
}

std::vector<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>
honeycomb::construct_symmetry() const {
    // Define `p_mat`
    typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> p_mat;
    std::vector<p_mat> ret(n_total);

    // Permutation function, permutes the indices of a
    // `Eigen::PermutationMatrix` by a respective amount.
    //
    // The Honeycomb lattice is translationally symmetry by translations about
    // the unitcells and also is symmetric by 180° rotations and shift to the
    // other basis index.
    auto permute = [this](size_t y, size_t x, bool s, p_mat& p) {
        auto& indices = p.indices();

        // Iterate over all indices
        for (size_t i = 0; i < n_total; i++) {
            size_t uc = uc_idx(i);

            // Move the unitcell along their respective directions.
            for (size_t xi = 0; xi < x; xi++) uc = up(uc, 0);
            for (size_t yi = 0; yi < y; yi++) uc = up(uc, 1);

            // If s == true do the 180° rotation. otherwise just return the new
            // site_index
            if (s) {
                indices(i) = n_total - 1 - idx(uc, b_idx(i));
            } else {
                indices(i) = idx(uc, b_idx(i));
            }
        }
    };

    // Iterate over all unitcell positions, i.e. all the symmetry points
    for (size_t i = 0; i < n_uc; i++) {
        for (size_t j = 0; j < n_uc; j++) {
            size_t id = n_uc * i + j;

            // Initialize the permutation matrix and get the permutation for
            // the 0th basis.
            ret[2 * id] = p_mat(n_total);
            permute(i, j, false, ret[2 * id]);

            // Initialize the permutation matrix and get the permutation for
            // the 1st basis with the 180 degree rotation.
            ret[2 * id + 1] = p_mat(n_total);
            permute(i, j, true, ret[2 * id + 1]);
        }
    }
    return ret;
}

void honeycomb::print_lattice(const std::vector<size_t>& el) const {
    // Print the lattice with zero spin in the bottom left, therefore begin
    // with the last row
    for (size_t row = n_uc - 1; row < n_uc; row--) {
        for (size_t i = n_basis - 1; i < n_basis; i--) {
            // Print shift with spaces.
            std::cout << std::string(2 * (row + i), ' ');

            for (size_t col = 0; col < n_uc; col++) {
                size_t oc = count_occurances_(idx({row, col}, i), el);

                // Print the site, `.` if no highlight, number of occurances
                // otherwise.
                if (oc == 0) {
                    std::cout << ".   ";
                } else {
                    std::cout << oc << "   ";
                }
            }
            std::cout << std::endl;
        }
    }
}

size_t honeycomb::count_occurances_(size_t idx,
                                    const std::vector<size_t>& el) const {
    size_t ret = 0;
    // Count number of occurances.
    for (size_t i : el) ret += idx == i;
    return ret;
}

