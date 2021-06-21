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
#include <Eigen/Sparse>
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
    typedef Eigen::SparseMatrix<std::complex<double>> SparseXcd;

    std::vector<size_t> acts_on_;  ///< List of site indices operator acts on.
    SparseXcd op_;  ///< Operator Matrix of size `2**len(acts_on_)`.l

   public:
    /**
     * @brief Local Operator constructor.
     *
     * @param acts_on Vector of site indices, the operator acts on.
     * @param op Operator Matrix of size `2**len(acts_on)`
     */
    local_op(const std::vector<size_t>& acts_on, const SparseXcd& op);

    void evaluate(machine::rbm_base&, const Eigen::MatrixXcd&,
                  const machine::rbm_context&) final;
};

}  // namespace operators

