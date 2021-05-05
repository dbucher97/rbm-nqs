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

#include <Eigen/Dense>
#include <array>
#include <vector>
//
#include <lattice/bravais.hpp>

namespace lattice {

struct plaq {
    std::array<size_t, 4> idxs;
    size_t type;
};

class toric_lattice : public bravais {
   private:
    /**
     * @brief Counts the number of occurances of one site in the hightlights
     * vector.
     *
     * @todo Move to bravais class, since it must be useful in general for all
     * print lattice methods.
     *
     * @param site_idx Site index.
     * @param highlights Vector of site indices, aka hightlights.
     *
     * @return Number of occurances.
     */
    size_t count_occurances_(size_t site_idx,
                             const std::vector<size_t>& highlights) const;

   public:
    using Base = bravais;

    /**
     * @brief Constructor of the toric_lattice lattice.
     *
     * @param n_uc Number of unitcells in one direction.
     */
    toric_lattice(size_t n_uc);

    virtual std::vector<size_t> nns(size_t) const override;

    virtual void construct_bonds() override;

    std::vector<plaq> construct_plaqs() const;

    virtual std::vector<
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>
    construct_symmetry() const override;

    using Base::print_lattice;
    virtual void print_lattice(const std::vector<size_t>&) const override;

    virtual bool has_correlators() const override { return true; }

    virtual std::vector<correlator_group> get_correlators() const override;
};

}  // namespace lattice
