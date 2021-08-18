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
   protected:
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

    bool full_symm_;

    virtual size_t rot180(size_t idx) const;

   public:
    using Base = bravais;

    /**
     * @brief Constructor of the Honeycomb lattice.
     *
     * @param n_uc Number of unitcells in one direction.
     * @param n_uc_b Number of unitcells in another direction.
     */
    honeycomb(size_t n_uc, int n_uc_b = -1, bool full_symmetry = true);

    virtual std::vector<size_t> nns(size_t) const override;

    virtual void construct_bonds() override;

    virtual std::vector<
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>
    construct_symmetry() const override;

    virtual size_t symmetry_size() const override {
        if (full_symm_) {
            return n_total;
        } else {
            return Base::symmetry_size();
        }
    }

    using Base::print_lattice;
    virtual void print_lattice(const std::vector<size_t>&) const override;

    virtual bool supports_custom_weight_initialization() const override {
        return true;
    }
    virtual void initialize_vb(const std::string& type,
                               Eigen::MatrixXcd& v_bias) const override;

    virtual bool has_correlators() const override { return !full_symm_; }

    virtual std::vector<correlator_group> get_correlators() const override;

    std::vector<std::vector<size_t>> get_hexagons() const;
};

}  // namespace lattice
