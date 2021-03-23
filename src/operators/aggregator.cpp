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

#include <omp.h>

#include <Eigen/Dense>
#include <stdexcept>
//
#include <operators/aggregator.hpp>

using namespace operators;

aggregator::aggregator(const base_op& op, size_t r, size_t c)
    : result_(r, c), variance_(r, c), op_{op} {
    // Initialize result as zero.
    set_zero();
}

aggregator::aggregator(const base_op& op)
    : aggregator{op, op.rows(), op.cols()} {}

void aggregator::set_zero() {
    result_.setZero();
    if (track_variance_) {
        variance_.setZero();
    }
}

void aggregator::finalize(double num) {
    // Normalize result
    result_ /= num;
    if (track_variance_) {
        variance_ /= num;
        // Calculate variance <0^2> - <0>^2
        variance_ -= (Eigen::MatrixXcd)result_.array().pow(2);
    }
}

Eigen::MatrixXcd& aggregator::get_result() { return result_; }
Eigen::MatrixXcd& aggregator::get_variance() { return variance_; }

Eigen::MatrixXcd aggregator::aggregate_() {
    // By default, just forward the operator result.
    return op_.get_result();
}

void aggregator::aggregate(double weight) {
    // Calculate the weight * observable
    Eigen::MatrixXcd x = weight * aggregate_();
    // Safly aggeregte the result
#pragma omp critical
    result_ += x;

    if (track_variance_) {
        // Calculate the resul of the squared observable
        Eigen::MatrixXcd xx = x.array().pow(2) / weight;
        // Safely aggregate teh variance.
#pragma omp critical
        variance_ += xx;
    }
}

prod_aggregator::prod_aggregator(const base_op& op, const base_op& scalar)
    : Base{op}, scalar_{scalar} {
    if (scalar_.rows() != 1 || scalar_.cols() != 1) {
        throw std::runtime_error("scalar operator must have size (1, 1).");
    }
}

Eigen::MatrixXcd prod_aggregator::aggregate_() {
    // Calculate op_a op_b^*
    return scalar_.get_result()(0, 0) * op_.get_result().conjugate();
}

outer_aggregator::outer_aggregator(const base_op& op)
    : Base{op, op.rows(), op.rows()} {
    // Guard operator to be vector.
    if (op_.cols() != 1) {
        throw std::runtime_error(
            "outer_aggregator needs column vector type operators");
    }
}

Eigen::MatrixXcd outer_aggregator::aggregate_() {
    // Calculate op^* op^T
    return op_.get_result().conjugate() * op_.get_result().transpose();
}
