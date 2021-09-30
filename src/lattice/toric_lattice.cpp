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
#include <lattice/toric_lattice.hpp>

using namespace lattice;

// The toric_lattice is a 2 dimensional bravais lattice with 2 basis sites
// and a coordination of four.
toric_lattice::toric_lattice(size_t n_uc) : Base{n_uc, 2, 2, 4} {
    construct_bonds();
}

std::vector<size_t> toric_lattice::nns(size_t i) const {
    // Get the toric_lattice NNs. different positions for the two basis indices.
    // Order: x-bond, y-bond, z-bond.
    size_t uc = uc_idx(i);
    if (b_idx(i) == 0) {
        return {idx(uc, 1), idx(up(uc), 1), idx(down(uc, 1), 1),
                idx(up(down(uc, 1)), 1)};
    } else {
        return {idx(uc, 0), idx(down(uc), 0), idx(up(uc, 1), 0),
                idx(down(up(uc, 1)), 0)};
    }
}

void toric_lattice::construct_bonds() {
    // Get all bonds by iterating ove the unitcells and get all NNs of basis
    // index one. Assing each bond the type 0 for x, 1 for y and 2 for z.
    for (size_t i = 0; i < n_total_uc; i++) {
        auto nn = nns(idx(i, 0));
        for (size_t c = 0; c < n_coordination; c++)
            bonds_.push_back({idx(i, 0), nn[c], c});
    }
}

std::vector<plaq> toric_lattice::construct_plaqs() const {
    std::vector<plaq> ret;
    for (size_t uc = 0; uc < n_total_uc; uc++) {
        // Plaquette
        ret.push_back(
            {{idx(uc, 0), idx(uc, 1), idx(up(uc, 1), 0), idx(up(uc), 1)}, 0});
        // Vertex
        ret.push_back(
            {{idx(uc, 0), idx(uc, 1), idx(down(uc), 0), idx(down(uc, 1), 1)},
             1});
    }
    return ret;
}

void toric_lattice::print_lattice(const std::vector<size_t>& el) const {
    // Print the lattice with zero spin in the bottom left, therefore begin
    // with the last row
    for (size_t row = n_uc - 1; row < n_uc; row--) {
        for (size_t i = n_basis - 1; i < n_basis; i--) {
            if (i == 0) std::cout << "  ";
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

size_t toric_lattice::count_occurances_(size_t idx,
                                        const std::vector<size_t>& el) const {
    size_t ret = 0;
    // Count number of occurances.
    for (size_t i : el) ret += idx == i;
    return ret;
}

std::vector<toric_lattice::correlator_group> toric_lattice::get_correlators()
    const {
    auto plaqs = construct_plaqs();
    correlator_group plaq_group;
    for (auto& p : plaqs) {
        if (p.type == 0) {
            plaq_group.push_back(
                std::vector<size_t>(p.idxs.begin(), p.idxs.end()));
        }
    }
    return {plaq_group};
}
