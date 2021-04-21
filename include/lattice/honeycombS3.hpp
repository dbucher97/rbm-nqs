/*
 * Copyright (C) 2021  David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <Eigen/Dense>
#include <vector>
//
#include <lattice/bravais.hpp>

#define BOND_XX 0
#define BOND_YY 1
#define BOND_ZZ 2
#define BOND_X_YZ 3
#define BOND_X_ZY 4
#define BOND_Y_XZ 5
#define BOND_Y_ZX 6
#define BOND_Z_XY 7
#define BOND_Z_YX 8

namespace lattice {

/**
 * @brief Honeycomb S3 transformed 6 spin unitcells
 */
class honeycombS3 : public bravais {
    using Base = bravais;

   public:
    /**
     * @brief Constructor of the Honeycomb lattice.
     *
     * @param n_uc Number of unitcells in one direction.
     */
    honeycombS3(size_t n_uc);

    virtual std::vector<size_t> nns(size_t) const override;

    virtual void construct_bonds() override;

    virtual std::vector<
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>
    construct_symmetry() const override;

    virtual bool has_correlators() const override { return true; }

    virtual std::vector<correlator_group> get_correlators() const override;

    using Base::print_lattice;
    virtual void print_lattice(const std::vector<size_t>&) const override;
};

}  // namespace lattice
