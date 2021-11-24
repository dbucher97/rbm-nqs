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
#include <lattice/honeycomb_hex.hpp>

using namespace lattice;

honeycomb_hex::honeycomb_hex(size_t n_max, const std::vector<double>& symmetry)
    : Base{n_max, (int)n_max * 3, symmetry},
      n_below_uc_{n_uc * (n_uc - 1) / 2},
      n_lower_uc_{n_uc * (2 * n_uc + 1) - n_below_uc_} {
    bonds_.clear();
    construct_bonds();
}

size_t honeycomb_hex::cols(size_t row) const {
    if (row <= n_uc) {
        return row + n_uc;
    } else {
        return 3 * n_uc - row;
    }
}

void honeycomb_hex::row_col(size_t uc, size_t& row, size_t& col) const {
    col = n_uc;
    row = 0;
    size_t below = 0;
    int delta = 1;

    while (uc >= below) {
        below += col;
        if (col == 2 * n_uc) {
            delta = -1;
        }
        col += delta;
        row++;
    }
    row--;
    col = uc - (below - col + delta);
}

void honeycomb_hex::get_loc(size_t uc, int* loc) const {
    // NOT REAL LOC
    size_t t = 2 * n_uc;
    if (uc < t * n_uc) {
        loc[0] = uc % t;
        loc[1] = uc / t;
    } else {
        size_t ucx = uc - t * n_uc;
        loc[0] = ucx % n_uc;
        loc[1] = ucx / n_uc + n_uc;
    }
}

size_t honeycomb_hex::uc_idx(std::vector<size_t>&& idxs) const {
    if (idxs[0] <= n_uc) {
        return (idxs[0] + n_uc - 1) * (idxs[0] + n_uc) / 2 - n_below_uc_ +
               idxs[1];
    } else {
        return (2 * n_uc - 1) * n_uc -
               (2 * n_uc + 1 - idxs[0] + n_uc) * (3 * n_uc - idxs[0]) / 2 +
               n_lower_uc_ + idxs[1];
    }
}

size_t honeycomb_hex::up(size_t uc, size_t dir, size_t step) const {
    if (step == 0) return uc;
    size_t row, col;
    row_col(uc, row, col);
    col++;
    if (dir == 1) {
        row++;
        if (row > n_uc) {
            col--;
        }
        row %= 2 * n_uc;
    }
    if (col >= cols(row)) {
        if (row >= n_uc) {
            row -= n_uc;
        } else {
            row += n_uc;
        }
        col = 0;
    }
    uc = uc_idx({row, col});
    if (step > 1) {
        return up(uc, dir, step - 1);
    }
    return uc;
}
size_t honeycomb_hex::down(size_t uc, size_t dir, size_t step) const {
    if (step == 0) return uc;
    size_t row, col;
    row_col(uc, row, col);
    col--;
    if (dir == 1) {
        row--;
        if (row >= n_uc) {
            col++;
        }
        row = (row + 2 * n_uc) % (2 * n_uc);
    }
    if (col >= cols(row)) {
        if (row >= n_uc) {
            row -= n_uc;
        } else {
            row += n_uc;
        }
        col = cols(row) - 1;
    }
    uc = uc_idx({row, col});
    if (step > 1) {
        return down(uc, dir, step - 1);
    }
    return uc;
}

void honeycomb_hex::print_lattice(const std::vector<size_t>& highlights) const {
    size_t start = n_uc + 1;
    int delta = +1;
    for (size_t i = 2 * n_uc - 1; i < 2 * n_uc; i--) {
        for (size_t k = 1; k < 2; k--) {
            std::string fill(2 * (2 * n_uc - start + k * (delta * (k == 1))),
                             ' ');
            std::cout << fill;
            if (k == 1 && delta == -1) {
                size_t id = idx(uc_idx({i + n_uc, cols(i + n_uc) - 1}), k);
                size_t oc = count_occurances_(id, highlights);
                if (oc) {
                    std::cout << oc << "   ";
                } else {
                    std::cout << ".   ";
                }
            }
            for (size_t j = 0; j < start - (k == 1 && delta == 1); j++) {
                size_t id = idx(uc_idx({i, j}), k);
                size_t oc = count_occurances_(id, highlights);
                if (oc) {
                    std::cout << oc << "   ";
                } else {
                    std::cout << ".   ";
                }
            }
            std::cout << std::endl;
        }
        if (start == 2 * n_uc) {
            delta = -1;
        }
        start += delta;
    }
}

std::vector<std::vector<size_t>> honeycomb_hex::construct_uc_symmetry(
    const std::vector<double>& symm) const {
    if (symm.size() == 1 && std::abs(symm[0] - 0.6) < 1e-10) {
        std::vector<std::vector<size_t>> ret(n_total_uc / 3);
        size_t n = std::sqrt(n_total_uc / 3);
        for (size_t y = 0; y < n; y++) {
            for (size_t x = 0; x < n; x++) {
                for (size_t uc = 0; uc < n_total_uc; uc++) {
                    size_t u = uc;
                    u = up(up(u, 1, x), 0, x);
                    u = down(up(u, 1, 2 * y), 0, y);
                    ret[x + n * y].push_back(u);
                }
            }
        }
        return ret;
    } else {
        std::vector<std::vector<size_t>> ret(n_total_uc);
        size_t offset = 0;
        for (size_t y = 0; y < 2 * n_uc; y++) {
            for (size_t x = 0; x < cols(y); x++) {
                for (size_t uc = 0; uc < n_total_uc; uc++) {
                    size_t uc2 = up(up(uc, 0, x), 1, y);
                    if (y <= n_uc) {
                        uc2 = down(uc2, 0, y);
                    } else {
                        uc2 = down(uc2, 0, n_uc);
                    }
                    ret[x + offset].push_back(uc2);
                }
            }
            offset += cols(y);
        }
        return ret;
    }
}

std::vector<size_t> honeycomb_hex::construct_symm_basis(
    const std::vector<double>& symm) const {
    if (symm.size() == 1 && std::abs(symm[0] - 0.6) < 1e-10) {
        const size_t uc = 0;
        return {idx(down(uc, 1), 1),
                idx(uc, 0),
                idx(uc, 1),
                idx(up(uc), 0),
                idx(down(up(uc), 1), 1),
                idx(down(up(uc), 1), 0)};
    } else {
        return Base::construct_symm_basis(symm);
    }
}

size_t honeycomb_hex::rot180(size_t i) const {
    size_t row, col;
    size_t uc = uc_idx(i);
    row_col(uc, row, col);
    if (cols(row) == col + 1 && row >= n_uc) {
        row = 3 * n_uc - row - 1;
        col = cols(row) - 1;
    } else {
        col = cols(row) - 1 - col - (row >= n_uc);
        row = 2 * n_uc - 1 - row;
    }
    return idx(uc_idx({row, col}), n_basis - 1 - b_idx(i));
}
