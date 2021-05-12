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

aggregator::aggregator(const base_op& op) : result_(1, 1), op_{op} {}

void aggregator::init(size_t num_samples) {
    current_ = 0;
    if ((size_t)result_.cols() != num_samples ||
        (size_t)result_.rows() != op_.size()) {
        result_.resize(op_.size(), num_samples);
    }
}

const Eigen::MatrixXcd& aggregator::get_result() const { return result_; }

void aggregator::aggregate(double weight) {
    // Calculate the weight * observable
#pragma omp critical
    {
        result_.col(current_) = Eigen::Map<const Eigen::MatrixXcd>(
            op_.get_result().data(), result_.rows(), 1);
        if (weight != 1.) {
            result_.col(current_) *= weight;
        }
        current_++;
    }
}

