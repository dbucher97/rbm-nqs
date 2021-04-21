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

// #define FULL_SYMMETRY

#include <Eigen/Dense>
#include <memory>
#include <vector>

#define DIR_X 0
#define DIR_Y 1
#define DIR_Z 2

/**
 * @brief Namespace of Lattice related objects.
 */
namespace lattice {

/**
 * @brief a simple bond struct with properties site a, site b and bond
 * type.
 */
struct bond {
    const size_t a;
    const size_t b;
    const size_t type;
};

/**
 * @brief The abstract Bravais Lattice class, as a base class for lattices.
 * Capable of handling periodic unitcells in multiple dimensions.
 */
class bravais {
   public:
    const size_t n_uc;            ///< Number of unitcells in one dimension.
    const size_t n_dim;           ///< Number of dimensions.
    const size_t n_basis;         ///< Basis size.
    const size_t n_coordination;  ///< Coordination number.
    const size_t n_total_uc;      ///< Number of total unitcells in lattice.
    const size_t n_total;         ///< Number of total spins in lattice.
    const size_t n_uc_b;          ///< Number of unitcells in second direction.
    const size_t h_shift;  ///< Horizontal shift on vertical periodic boundary.

   protected:
    using correlator = std::vector<size_t>;
    using correlator_group = std::vector<correlator>;

    std::vector<bond> bonds_;

    /**
     * @brief Bravais constructor.
     *
     * @param n_uc Number of unitcells in one dimension.
     * @param n_dim Number of spatial dimensions.
     * @param n_basis Basis number.
     * @param n_coordination Coordination numnber.
     * @param n_uc_b Number of unitcells in second direction.
     * @param h_shift horizontal unitcell shift when going over a periodic.
     * boundary
     */
    bravais(size_t n_uc, size_t n_dim, size_t n_basis, size_t n_coordination,
            size_t n_uc_b = 0, size_t h_shift = 0);

    /**
     * @brief Constructs the nearest neighbouring bonds of the lattice
     *
     * @return vector of bonds
     */
    virtual void construct_bonds() = 0;

   public:
    /**
     * @brief Returns the unitcell index for a site index.
     *
     * @param idx Site index.
     *
     * @return Unitcell index
     */
    size_t uc_idx(size_t idx) const;

    /**
     * @brief Returns the unitcell index for a set of unitcell indices in all
     * spatial dimensions.
     *
     * @param idxs rhs vector of uc indices in all spatial dimensions e.g. `{1,
     * 2, 3}`.
     *
     * @return Unitcell index.
     */
    size_t uc_idx(std::vector<size_t>&& idxs) const;

    /**
     * @brief Returns the basis index for a site index
     *
     * @param idx Site index.
     *
     * @return Basis index.
     */
    size_t b_idx(size_t idx) const;

    /**
     * @brief Returns the site index given a unitcell and a basis index.
     *
     * @param uc_idx Unitcell index.
     * @param b_idx Basis index.
     *
     * @return site index.
     */
    size_t idx(size_t uc_idx, size_t b_idx) const;

    /**
     * @brief Returns the site index given a vector of unitcell indices in all
     * spatial dimensions and a basis index.
     *
     * @param idxs rhs vector of uc indices in all spatial dimensions e.g. `{1,
     * 2, 3}`.
     * @param b_idx Basis index.
     *
     * @return site index.
     */
    size_t idx(std::vector<size_t>&& idxs, size_t b_idx) const;

    /**
     * @brief move one unitcell up in given dimension
     *
     * @param uc_idx Current unitcell index
     * @param dir Direction (Dimension of movement: 0 <-> x, 1 <-> y, ...)
     *
     * @return Index of new unitcell.
     */
    size_t up(size_t uc_idx, size_t dir = 0) const;

    /**
     * @brief move one unitcell down in given dimension
     *
     * @param uc_idx Current unitcell index
     * @param dir Direction (Dimension of movement: 0 <-> x, 1 <-> y, ...)
     *
     * @return Index of new unitcell.
     */
    size_t down(size_t ux_idx, size_t dir = 0) const;

    /**
     * @brief Returns the nearest neighbour sites to a given site.
     *
     * @param site_idx Index of reference site.
     *
     * @return vector of site indices of the nearest neightbours.
     */
    virtual std::vector<size_t> nns(size_t site_idx) const = 0;

    /**
     * @brief Retruns the nearest neighbouring bonds of the lattice
     *
     * @return vector of bonds
     */
    virtual const std::vector<bond>& get_bonds() const { return bonds_; }

    /**
     * @brief Constructs the symmetry of the lattice. A symmetry can be
     * expressed as a permutation on the spin site indices.
     *
     * @return vector of permutation matrices corresponding to the symmetries.
     */
    virtual std::vector<
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>
    construct_symmetry() const = 0;

    /**
     * @brief Check if lattice has correlators.
     *
     * @return True if it has correlators.
     */
    virtual bool has_correlators() const { return false; }

    /**
     * @brief Returns a vector of correlator groups
     *
     * @return Vector of correlator_groups
     */
    virtual std::vector<correlator_group> get_correlators() const {
        return {};
    };

    /**
     * @brief Printis the lattice and highlights
     * certain sites inside the lattice. A highlight is marked with the
     * number of occurances in the hightlits vector.
     *
     * @param highlights Vector of site indices to be hightligthed.
     */
    virtual void print_lattice(const std::vector<size_t>& highlights) const = 0;

    /**
     * @brief `print_lattice` wrapper with empty highlights.
     */
    void print_lattice() const { print_lattice({}); }

    /**
     * @brief Default virtual destructor.
     */
    virtual ~bravais() = default;
};

}  // namespace lattice
