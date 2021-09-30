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

#include <iostream>
//
#include <lattice/square.hpp>

using namespace lattice;

void square::print_lattice(const std::vector<size_t>& highlights) const {
    for (size_t i = n_uc_b - 1; i < n_uc_b; i--) {
        for (size_t j = 0; j < n_uc; j++) {
            size_t site = idx({i, j}, 0);
            size_t oc = 0;
            for (const auto& h : highlights) {
                oc += h == site;
            }
            if (oc)
                std::cout << oc << " ";
            else
                std::cout << ". ";
        }
        std::cout << std::endl;
    }
}

std::vector<size_t> square::nns(size_t s) const {
    return {down(s), up(s, 1), up(s), down(s, 1)};
}

std::vector<size_t> square::nnns(size_t s) const {
    return {up(down(s), 1), up(up(s, 1)), down(up(s), 1), down(down(s, 1))};
}

void square::construct_bonds() {
    for (size_t i = 0; i < n_total_uc; i++) {
        bonds_.push_back({i, up(i), 0});
        bonds_.push_back({i, up(i, 1), 0});
    }
}
