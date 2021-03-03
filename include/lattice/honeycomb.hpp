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
#include <vector>
//
#include <lattice/bravais.hpp>

#define BOND_X 0
#define BOND_Y 1
#define BOND_Z 2

namespace lattice {

class honeycomb : public bravais {
   private:
    size_t _count_occurances(size_t, const std::vector<size_t>&) const;

   public:
    using Base = bravais;
    honeycomb(size_t);

    virtual std::vector<size_t> nns(size_t) const override;

    virtual std::vector<bond> get_bonds() const override;

    virtual std::vector<
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>
    construct_symmetry() const override;

    using Base::print_lattice;
    virtual void print_lattice(const std::vector<size_t>&) const override;
};

}  // namespace lattice
