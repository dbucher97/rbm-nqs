/**
 * include/optimizer/plugin.hpp
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

namespace optimizer {

class base_plugin {
   public:
    virtual ~base_plugin() = default;

    virtual Eigen::MatrixXcd apply(Eigen::MatrixXcd& dw) = 0;
};

class adam_plugin : public base_plugin {
    using Base = base_plugin;

    double beta1_, beta2_, eps_;
    size_t t_;

    Eigen::MatrixXcd m_;
    Eigen::MatrixXd wr_;
    Eigen::MatrixXd wi_;

   public:
    adam_plugin(size_t l, double beta1 = 0.9, double beta2 = 0.999,
                double eps = 1e-8);

    virtual Eigen::MatrixXcd apply(Eigen::MatrixXcd&) override;
};

class momentum_plugin : public base_plugin {
    using Base = base_plugin;

    double alpha_;

    Eigen::MatrixXcd m_;

   public:
    momentum_plugin(size_t l, double alpha = 0.1);

    virtual Eigen::MatrixXcd apply(Eigen::MatrixXcd&) override;
};

}  // namespace optimizer
