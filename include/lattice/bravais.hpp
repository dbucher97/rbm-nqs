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
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#define DIR_X 0
#define DIR_Y 1
#define DIR_Z 2

namespace lattice {

struct bond {
    const size_t a;
    const size_t b;
    const size_t type;
};

class bravais {
   public:
    const size_t n_uc, n_dim, n_basis, n_coordination, n_total_uc, n_total;

    bravais(size_t, size_t, size_t, size_t);

    size_t uc_idx(size_t) const;
    size_t uc_idx(std::vector<size_t>&&) const;
    size_t b_idx(size_t) const;
    size_t idx(size_t, size_t) const;

    size_t idx(std::vector<size_t>&& idxs, size_t b) const {
        return idx(uc_idx(std::move(idxs)), b);
    }

    size_t up(size_t, size_t = 0) const;
    size_t down(size_t, size_t = 0) const;

    virtual std::vector<size_t> nns(size_t) const = 0;

    virtual std::vector<bond> get_bonds() const = 0;

    virtual std::vector<
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>
    construct_symmetry() const = 0;

    virtual void print_lattice(const std::vector<size_t>&) const = 0;
    void print_lattice() const { print_lattice({}); }

    ~bravais() = default;
};

}  // namespace lattice
