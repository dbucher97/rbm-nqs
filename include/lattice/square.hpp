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

#include <lattice/bravais.hpp>

namespace lattice {

class square : public bravais {
    using Base = bravais;

    virtual void construct_bonds() override;

   public:
    square(size_t n_uc, size_t n_uc2 = 0) : Base{n_uc, 2, 1, 4, n_uc2} {
        construct_bonds();
    }

    virtual std::vector<size_t> nns(size_t site_idx) const override;

    std::vector<size_t> nnns(size_t site_idx) const;

    virtual void print_lattice(
        const std::vector<size_t>& highlights) const override;
};

}  // namespace lattice
