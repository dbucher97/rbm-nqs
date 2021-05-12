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
//
#include <operators/base_op.hpp>

/**
 * @brief Namespace for all operator related objects
 */
namespace operators {

/**
 * @brief Base class for an aggregator, an object which aggregates the
 * Operator op results of each sample. `agg.result_ = <op>`
 */
class aggregator {
    Eigen::MatrixXcd result_;  ///< The result Matrix

   protected:
    const base_op&
        op_;  ///< Operator, for which the results should be accumulated.

    size_t num_samples_;  ///< Number of samples
    size_t current_;      ///< Current sample

   public:
    /**
     * @brief Aggregator constructor with size same as oprator result size.
     *
     * @param base_op Reference to the operator.
     * @param num_samples Number of samples.
     */
    aggregator(const base_op&);
    /**
     * @brief Default virtual destructor.
     */
    virtual ~aggregator() = default;

    /**
     * @brief Aggregate the current operator(s) result.
     *
     * @param weight Weight.
     */
    void aggregate(double weight = 1.);

    /**
     * @brief Result getter.
     *
     * @return The reference to the result.
     */
    const Eigen::MatrixXcd& get_result() const;

    /**
     * @brief Sets result to zero.
     */
    void init(size_t num_samples);
};

}  // namespace operators
