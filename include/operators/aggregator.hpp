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
#include <optimizer/outer_matrix.hpp>

/**
 * @brief Namespace for all operator related objects
 */
namespace operators {

/**
 * @brief Base class for an aggregator, an object which aggregates the
 * Operator op results of each sample. `agg.result_ = <op>`
 */
class aggregator {
   protected:
    const size_t n_samples_;
    size_t n_bins_ = 50;
    size_t bin_size_;
    size_t cur_n_ = 0;
    size_t cur_n_bin_ = 0;

    bool track_variance_ =
        false;  ///< If is set, track the variance of a observable
    Eigen::MatrixXcd result_;          ///< The result Matrix
    Eigen::MatrixXcd result_binned_;   ///< The result Matrix
    Eigen::MatrixXcd bin_;             ///< The result Matrix
    Eigen::MatrixXd variance_;         ///< The variance Matrix
    Eigen::MatrixXd variance_binned_;  ///< The variance Matrix

    const base_op&
        op_;  ///< Operator, for which the results should be accumulated.

    /**
     * @brief Get the observable from the operator(s), this will be overriden
     * by derived classes.
     *
     * @return Matrix of the specific result.
     */
    virtual Eigen::MatrixXcd aggregate_();

    /**
     * @brief Protected Aggregator constructor for custorm observable size
     * (Matrix returned by `aggregate_`).
     *
     * @param base_op Reference to the operator.
     * @param rows Number of rows.
     * @param cols Number of cols.
     */
    aggregator(const base_op&, size_t samples, size_t, size_t);

   public:
    /**
     * @brief Aggregator constructor with size same as oprator result size.
     *
     * @param base_op Reference to the operator.
     */
    aggregator(const base_op&, size_t samples);
    /**
     * @brief Default virtual destructor.
     */
    virtual ~aggregator() = default;

    /**
     * @brief Aggregate the current operator(s) result.
     *
     * @param weight Weight (Default = 1.).
     */
    virtual void aggregate(double weight = 1.);

    /**
     * @brief Turn on `track_variance_`.
     */
    void track_variance(size_t n_bins = 50);

    /**
     * @brief Result getter.
     *
     * @return The reference to the result.
     */
    Eigen::MatrixXcd& get_result();
    /**
     * @brief Variance getter.
     *
     * @return The reference to the variance.
     */
    Eigen::MatrixXd& get_variance();
    /**
     * @brief Finalize the Aggregation.
     *
     * @param normalization divide by normalization factor.
     */
    virtual void finalize(double normalization);

    /**
     * @brief Sets result to zero.
     */
    virtual void set_zero();

    Eigen::MatrixXd get_stddev() const;
    Eigen::MatrixXd get_tau() const;
};

/**
 * @brief Prodcut aggregator calculates the expectation value of the product of
 * a scalar operator with a matrix sized operator. agg.resut_ = <op_ ob_b^*>
 */
class prod_aggregator : public aggregator {
    using Base = aggregator;

    const base_op& scalar_;  ///< Reference to the scalar operator.

    virtual Eigen::MatrixXcd aggregate_() override;

   public:
    /**
     * @brief Constructor of the product aggregator
     *
     * @param matrix_op Matrix sized operator.
     * @param scalar_op Scalar operator.
     */
    prod_aggregator(const base_op& matrix_op, const base_op& scalar_op,
                    size_t samples);
};

/**
 * @brief Outer aggregator calculates the expectation value of the outer
 * product of a vector sized operator with itself. agg.result_ = <op^* op^T>
 */
class outer_aggregator : public aggregator {
    using Base = aggregator;

    virtual Eigen::MatrixXcd aggregate_() override;

   public:
    /**
     * @brief Outer aggregator constructor.
     *
     * @param base_op Reference to the vector sized operator.
     */
    outer_aggregator(const base_op&, size_t samples);
};

/**
 * @brief Outer aggregator calculates the expectation value of the outer
 * product of a vector sized operator with itself. agg.result_ = <op^* op^T>
 */
class outer_aggregator_lazy : public aggregator {
    using Base = aggregator;

    size_t current_index_ = 0;
    double norm_ = 0;
    Eigen::VectorXd diag_;

   public:
    /**
     * @brief Outer aggregator constructor.
     *
     * @param base_op Reference to the vector sized operator.
     */
    outer_aggregator_lazy(const base_op&, size_t samples);

    virtual void aggregate(double weight = 1.) override;

    virtual void finalize(double val) override;
    virtual void set_zero() override;

    optimizer::OuterMatrix construct_outer_matrix(aggregator& derv, double reg1,
                                                  double reg2);

    double get_norm() { return norm_; }

    void finalize_diag(const Eigen::MatrixXcd&);
    Eigen::VectorXd& get_diag();
};

}  // namespace operators
