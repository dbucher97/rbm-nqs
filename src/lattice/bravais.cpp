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
#include <cmath>
#include <iostream>
#include <vector>
//
#include <lattice/bravais.hpp>

using namespace lattice;

bravais::bravais(size_t n_uc, size_t n_dim, size_t n_basis,
                 size_t n_coordination, size_t n_uc_b, size_t h_shift)
    : n_uc{n_uc},
      n_dim{n_dim},
      n_basis{n_basis},
      n_coordination{n_coordination},
      n_total_uc{n_uc_b == 0 ? static_cast<size_t>(std::pow(n_uc, n_dim))
                             : n_uc * n_uc_b},
      n_total{n_total_uc * n_basis},
      n_uc_b{n_uc_b == 0 ? n_uc : n_uc_b},
      h_shift{h_shift} {}

size_t bravais::uc_idx(size_t idx) const { return idx / n_basis; }

size_t bravais::uc_idx(std::vector<size_t>&& idxs) const {
    size_t fac = 1;
    // Starting with 0 index.
    size_t idx = 0;
    for (size_t i = idxs.size() - 1; i < idxs.size(); i--) {
        // Move index by `idxs[i]` in the current dimension.
        idx += fac * idxs[i];
        // Multiply by `n_uc` for next dimension
        if (i == 1) {
            fac *= n_uc_b;
        } else {
            fac *= n_uc;
        }
    }
    return idx;
}
size_t bravais::b_idx(size_t idx) const { return idx % n_basis; }
size_t bravais::idx(size_t uc_idx, size_t b_idx) const {
    return uc_idx * n_basis + b_idx;
}
size_t bravais::idx(std::vector<size_t>&& idxs, size_t b_idx) const {
    return idx(uc_idx(std::move(idxs)), b_idx);
}

size_t bravais::up(size_t uc_idx, size_t dir, size_t step) const {
    // Calculate the total index shift.
    size_t shift = static_cast<size_t>(std::pow(n_uc, dir)) * n_uc_b;
    // Get the offset of the last samlle dimension.
    size_t offset = uc_idx / shift;
    // use n_uc_b for first dim.
    size_t n_uc2 = (dir == 0 ? n_uc_b : n_uc);
    // Check if overflow happened for h shift
    size_t overflow = (uc_idx + (step * shift) / n_uc2) / shift;
    // Shift unitcell along a dimension.
    step %= n_uc2;
    uc_idx = (uc_idx + step * shift / n_uc2) % shift;
    // Get the complete unitcell index together with previous dimension.
    size_t ret = shift * offset + uc_idx;
    if (overflow && h_shift && dir == 1) {
        ret = down(ret, 0, h_shift);
    }
    return ret;
}
size_t bravais::down(size_t uc_idx, size_t dir, size_t step) const {
    // Calculate the total index shift.
    size_t shift = static_cast<size_t>(std::pow(n_uc, dir)) * n_uc_b;
    // Get the offset of the last samlle dimension.
    size_t offset = uc_idx / shift;
    // use n_uc_b for first dim.
    size_t n_uc2 = dir == 0 ? n_uc_b : n_uc;
    // Check if overflow happened for h shift
    size_t overflow = (uc_idx + (n_uc2 - step) * shift / n_uc2) / shift;
    // Shift unitcell along a dimension.
    step %= n_uc2;
    uc_idx = (uc_idx + (n_uc2 - step) * shift / n_uc2) % shift;
    // Get the complete unitcell index together with previous dimension.
    size_t ret = shift * offset + uc_idx;
    if (!overflow && h_shift && dir == 1) {
        ret = up(ret, 0, h_shift);
    }
    return ret;
}

std::vector<std::vector<size_t>> bravais::construct_uc_symmetry() const {
    std::vector<std::vector<size_t>> ret(n_uc_b * n_uc);
    for (size_t y = 0; y < n_uc; y++) {
        for (size_t x = 0; x < n_uc_b; x++) {
            for (size_t uc = 0; uc < n_total_uc; uc++) {
                ret[x + y * n_uc_b].push_back(up(up(uc, 0, x), 1, y));
            }
        }
    }
    return ret;
}

std::vector<Eigen::PermutationMatrix<Eigen::Dynamic>>
bravais::construct_symmetry() const {
    using p_mat = Eigen::PermutationMatrix<Eigen::Dynamic>;
    std::vector<p_mat> ret(n_total_uc);

    // Permutation function, permutes the indices of a
    // `Eigen::PermutationMatrix` by a respective amount.
    //
    auto permute = [this](const std::vector<size_t>& ucs, bool s, p_mat& p) {
        auto& indices = p.indices();

        // Iterate over all indices
        for (size_t i = 0; i < n_total; i++) {
            size_t uc = ucs[uc_idx(i)];

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
    auto uc_symm = construct_uc_symmetry();
    for (size_t i = 0; i < n_total_uc; i++) {
        size_t id = i;

        // Initialize the permutation matrix and get the permutation for
        // the 0th basis.
        ret[id] = p_mat(n_total);
        permute(uc_symm[i], false, ret[id]);

        // Initialize the permutation matrix and get the permutation for
        // the 1st basis with the 180 degree rotation.
    }
    return ret;
}
