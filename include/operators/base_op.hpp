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

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <vector>
//
#include <machine/rbm.hpp>

namespace operators {

class base_op {
   protected:
    Eigen::MatrixXcd result_;

   public:
    base_op(size_t r = 1, size_t c = 1) : result_(r, c) {}
    virtual ~base_op() = default;

    virtual void evaluate(machine::rbm&, const Eigen::MatrixXcd&,
                          const Eigen::MatrixXcd&) = 0;

    bool is_scalar() const {
        return result_.cols() == 1 && result_.rows() == 1;
    }
    bool is_vector() const { return result_.cols() == 1; }
    size_t rows() const { return result_.rows(); }
    size_t cols() const { return result_.cols(); }

    const Eigen::MatrixXcd& get_result() const { return result_; }
};
}  // namespace operators
