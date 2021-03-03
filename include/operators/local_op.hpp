/**
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
#include <machine/rbm.hpp>
#include <operators/base_op.hpp>

namespace operators {

class local_op : public base_op {
    using Base = base_op;
    std::vector<size_t> acts_on_;
    Eigen::MatrixXcd& op_;
    std::vector<size_t> flips_;

    size_t get_local_psi(const Eigen::MatrixXcd&);
    void get_flips(size_t);

   public:
    local_op(size_t n_total, const std::vector<size_t>&, Eigen::MatrixXcd&);

    void evaluate(machine::rbm&, const Eigen::MatrixXcd&,
                  const Eigen::MatrixXcd&) final;
};

}  // namespace operators

