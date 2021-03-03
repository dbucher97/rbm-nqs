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

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>
//
#include <lattice/honeycomb.hpp>

using namespace lattice;

honeycomb::honeycomb(size_t n_uc) : Base{n_uc, 2, 2, 3} {}

std::vector<size_t> honeycomb::nns(size_t i) const {
    size_t uc = uc_idx(i);
    if (b_idx(i) == 0) {
        return {idx(uc, 1), idx(down(uc), 1), idx(down(uc, 1), 1)};
    } else {
        return {idx(uc, 0), idx(up(uc), 0), idx(up(uc, 1), 0)};
    }
}

std::vector<bond> honeycomb::get_bonds() const {
    std::vector<bond> vec;
    for (size_t i = 0; i < n_total_uc; i++) {
        auto nn = nns(idx(i, 0));
        for (size_t c = 0; c < n_coordination; c++)
            vec.push_back({idx(i, 0), nn[c], c});
    }
    return vec;
}

std::vector<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>
honeycomb::construct_symmetry() const {
    typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> p_mat;
    std::vector<p_mat> ret(n_total);
    auto permute = [this](size_t y, size_t x, bool s, p_mat& p) {
        auto& indices = p.indices();
        for (size_t i = 0; i < n_total; i++) {
            size_t uc = uc_idx(i);
            for (size_t xi = 0; xi < x; xi++) uc = up(uc, 0);
            for (size_t yi = 0; yi < y; yi++) uc = up(uc, 1);
            if (s) {
                indices(i) = n_total - 1 - idx(uc, b_idx(i));
            } else {
                indices(i) = idx(uc, b_idx(i));
            }
        }
    };
    for (size_t i = 0; i < n_uc; i++) {
        for (size_t j = 0; j < n_uc; j++) {
            size_t id = n_uc * i + j;
            ret[2 * id] = p_mat(n_total);
            permute(i, j, false, ret[2 * id]);
            ret[2 * id + 1] = p_mat(n_total);
            permute(i, j, true, ret[2 * id + 1]);
        }
    }
    return ret;
}

void honeycomb::print_lattice(const std::vector<size_t>& el) const {
    for (size_t row = n_uc - 1; row < n_uc; row--) {
        for (size_t i = n_basis - 1; i < n_basis; i--) {
            std::cout << std::string(2 * (row + i), ' ');
            for (size_t col = 0; col < n_uc; col++) {
                size_t oc = _count_occurances(idx({row, col}, i), el);
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

size_t honeycomb::_count_occurances(size_t idx,
                                    const std::vector<size_t>& el) const {
    size_t ret = 0;
    for (size_t i : el) ret += idx == i;
    return ret;
}

