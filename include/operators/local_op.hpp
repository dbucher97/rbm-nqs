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
#include <machine/rbm_base.hpp>
#include <operators/base_op.hpp>

namespace operators {

/**
 * @brief Local operator acting on only a few sites.
 */
class local_op : public base_op {
    using Base = base_op;
    std::vector<size_t> acts_on_;  ///< List of site indices operator acts on.
    Eigen::MatrixXcd& op_;  ///< Operator Matrix of size `2**len(acts_on_)`.

    /**
     * @brief Returns the local quantum state of the selected sites. Since a
     * local psi derived from a z-basis state has only one non-zero entry at
     * index `loc`, returning only `loc` is sufficient.
     *
     * @param state Input state.
     *
     * @return The non-zero index `loc`.
     */
    size_t get_local_psi(const Eigen::MatrixXcd& state);

    /**
     * @brief Fills the vector flips with indices where bits of `loc` are 1.
     *
     * @param loc A integer where the sites to flip are 1.
     * @param flips A vector of site indices will be cleared and refilled.
     */
    void get_flips(size_t loc, std::vector<size_t>& flips);

   public:
    /**
     * @brief Local Operator constructor.
     *
     * @param acts_on Vector of site indices, the operator acts on.
     * @param op Operator Matrix of size `2**len(acts_on)`
     */
    local_op(const std::vector<size_t>& acts_on, Eigen::MatrixXcd& op);

    void evaluate(machine::rbm_base&, const Eigen::MatrixXcd&,
                  const Eigen::MatrixXcd&) final;
};

}  // namespace operators

