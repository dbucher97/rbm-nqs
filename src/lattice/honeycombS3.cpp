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
#include <lattice/honeycombS3.hpp>

using namespace lattice;

// The Honeycomb S3 lattice is a 2 dimensional bravais lattice with 6 basis
// sites
honeycombS3::honeycombS3(size_t n_uc) : Base{n_uc, 2, 6, 4, n_uc * 3, n_uc} {
    construct_bonds();
}

std::vector<size_t> honeycombS3::nns(size_t i) const {
    // Get the Honeycomb NNs. different positions for the two basis indices.
    size_t uc = uc_idx(i);
    size_t b = b_idx(i);
    switch (b) {
        case 0:
            return {idx(down(uc), 3), idx(down(uc, 1), 4), idx(uc, 1)};
        case 1:
            return {idx(uc, 2), idx(down(uc), 5), idx(uc, 0)};
        case 2:
            return {idx(uc, 1), idx(uc, 3), idx(uc, 4)};
        case 3:
            return {idx(up(uc), 0), idx(uc, 2), idx(down(uc, 1), 5)};
        case 4:
            return {idx(uc, 5), idx(up(uc, 1), 0), idx(uc, 2)};
        default:
            return {idx(uc, 4), idx(up(uc), 1), idx(up(uc, 1), 3)};
    }
}

void honeycombS3::construct_bonds() {
    std::vector<size_t> types;
    // Get all bonds by iterating ove the unitcells and get all NNs of basis
    // index one. Assing each bond the type 0 for x, 1 for y and 2 for z.
    for (size_t i = 0; i < n_total_uc; i++) {
        auto nn = nns(idx(i, 0));
        types = {BOND_XX, BOND_Y_XZ, BOND_Z_XY};
        for (size_t c = 0; c < nn.size(); c++)
            bonds_.push_back({idx(i, 0), nn[c], types[c]});
        nn = nns(idx(i, 2));
        types = {BOND_X_ZY, BOND_Y_ZX, BOND_ZZ};
        for (size_t c = 0; c < nn.size(); c++)
            bonds_.push_back({idx(i, 2), nn[c], types[c]});
        nn = nns(idx(i, 5));
        types = {BOND_X_YZ, BOND_YY, BOND_Z_YX};
        for (size_t c = 0; c < nn.size(); c++)
            bonds_.push_back({idx(i, 5), nn[c], types[c]});
    }
}

void honeycombS3::print_lattice(const std::vector<size_t>& el) const {
    for (size_t i = n_uc - 1; i < n_uc; i--) {
        for (size_t j = 0; j < n_uc_b; j++) {
            size_t c = 0;
            for (auto& e : el) {
                if (e == i * n_uc_b + j) c++;
            }
            if (c == 0) {
                std::cout << ".";
            } else {
                std::cout << c;
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<honeycombS3::correlator_group> honeycombS3::get_correlators()
    const {
    correlator_group plaq, xbonds, ybonds, zbonds;
    for (size_t i = 0; i < n_total_uc; i++) {
        plaq.push_back({idx(i, 0), idx(i, 1), idx(i, 2), idx(i, 3),
                        idx(down(i, 1), 5), idx(down(i, 1), 4)});
        xbonds.push_back({idx(i, 0), idx(down(i), 3)});
        ybonds.push_back({idx(i, 1), idx(down(i), 5)});
        zbonds.push_back({idx(i, 2), idx(i, 4)});
    }

    return {plaq, xbonds, ybonds, zbonds};
}
