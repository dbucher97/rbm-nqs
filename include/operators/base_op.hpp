/**
 * include/operator/base_op.hpp
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

#include <omp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <vector>
//
#include <machine/rbm_base.hpp>

namespace operators {

class base_op {
    size_t r_, c_;
    std::vector<Eigen::MatrixXcd> result_;

   protected:
    inline Eigen::MatrixXcd& get_result_() {
        return result_[omp_get_thread_num()];
    }

   public:
    base_op(size_t r = 1, size_t c = 1)
        : r_{r}, c_{c}, result_(omp_get_max_threads()) {
        for (size_t i = 0; i < result_.size(); i++)
            result_[i] = Eigen::MatrixXcd(r_, c_);
    }
    virtual ~base_op() = default;

    virtual void evaluate(machine::rbm_base&, const Eigen::MatrixXcd&,
                          const Eigen::MatrixXcd&) = 0;

    bool is_scalar() const { return r_ == 1 && c_ == 1; }
    bool is_vector() const { return c_ == 1; }
    size_t rows() const { return r_; }
    size_t cols() const { return c_; }

    const Eigen::MatrixXcd& get_result() const {
        return result_[omp_get_thread_num()];
    }
};
}  // namespace operators
