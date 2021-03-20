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
#include <complex>
#include <string>
#include <vector>
//
#include <machine/rbm_base.hpp>
#include <operators/base_op.hpp>

namespace operators {

class overlap_op : public base_op {
    using Base = base_op;
    Eigen::MatrixXcd state_vec_;
    size_t n_vis_;

    std::complex<double> get_psi(const Eigen::MatrixXcd& state);

    void fill_vec(const std::string& file);

   public:
    overlap_op(const std::string& file, size_t);

    virtual void evaluate(machine::rbm_base&, const Eigen::MatrixXcd&,
                          const Eigen::MatrixXcd&) override;
};

}  // namespace operators

