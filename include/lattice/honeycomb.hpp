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
#include <vector>
//
#include <lattice/bravais.hpp>

#define BOND_X 0
#define BOND_Y 1
#define BOND_Z 2

namespace lattice {

class honeycomb : public bravais {
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
     * @brief Constructor of the Honeycomb lattice.
     *
     * @param n_uc Number of unitcells in one direction.
     */
    honeycomb(size_t n_uc);

    virtual std::vector<size_t> nns(size_t) const override;

    virtual void construct_bonds() override;

    virtual std::vector<
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>
    construct_symmetry() const override;

    using Base::print_lattice;
    virtual void print_lattice(const std::vector<size_t>&) const override;

    virtual bool supports_custom_weight_initialization() const override {
        return true;
    }
    virtual void initialize_vb(const std::string& type,
                               Eigen::MatrixXcd& v_bias) const override;

#ifndef FULL_SYMMETRY
    virtual bool has_correlators() const override { return true; }

    virtual std::vector<correlator_group> get_correlators() const override;
#endif

    std::vector<std::vector<size_t>> get_hexagons() const;
    
};

}  // namespace lattice
