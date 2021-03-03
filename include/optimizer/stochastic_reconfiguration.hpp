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
#include <complex>
#include <vector>
//
#include <machine/rbm.hpp>
#include <machine/sampler.hpp>
#include <operators/aggregator.hpp>
#include <operators/base_op.hpp>
#include <operators/derivative_op.hpp>

namespace optimizer {

class stochastic_reconfiguration {
   public:
    stochastic_reconfiguration(machine::rbm&, machine::sampler&,
                               operators::base_op&, double lr = 0.001,
                               double k0 = 100, double kmin = 1e-4,
                               double m = 0.9);

    void register_observables();

    void optimize(size_t);

   private:
    machine::rbm& rbm_;
    machine::sampler& sampler_;
    operators::base_op& hamiltonian_;
    operators::derivative_op derivative_;

    operators::aggregator a_h_;
    operators::aggregator a_d_;
    operators::prod_aggregator a_dh_;
    operators::outer_aggregator a_dd_;

    size_t n_total_;

    double lr_;
    double m_;
    double kmin_;
    double kp_;

    bool reg_min_ = false;

    std::vector<Eigen::MatrixXcd> dws_ = {};
};
}  // namespace optimizer
