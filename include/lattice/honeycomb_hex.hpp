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
#pragma once

#include <lattice/honeycomb.hpp>

namespace lattice {
class honeycomb_hex : public honeycomb {
    using Base = honeycomb;

    const size_t n_below_uc_;
    const size_t n_lower_uc_;

    size_t cols(size_t row) const;
    void row_col(size_t uc, size_t& row, size_t& col) const;

    size_t rot180(size_t idx) const override;

   public:
    honeycomb_hex(size_t n_max, bool full_symm = true);

    using Base::uc_idx;
    virtual size_t uc_idx(std::vector<size_t>&& idxs) const override;

    virtual size_t up(size_t uc, size_t dir, size_t step = 1) const override;
    virtual size_t down(size_t uc, size_t dir, size_t step = 1) const override;

    virtual void print_lattice(
        const std::vector<size_t>& highlights) const override;

    virtual std::vector<std::vector<size_t>> construct_uc_symmetry()
        const override;
};

}  // namespace lattice
