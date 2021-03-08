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
#include <machine/abstract_sampler.hpp>
#include <machine/rbm_base.hpp>
#include <operators/aggregator.hpp>
#include <operators/base_op.hpp>
#include <operators/derivative_op.hpp>
#include <optimizer/plugin.hpp>

namespace optimizer {

class stochastic_reconfiguration {
   public:
    stochastic_reconfiguration(machine::rbm_base&, machine::abstract_sampler&,
                               operators::base_op&, double lr = 0.001,
                               double lrmin = 1e-3, double lrm = 0.99,
                               double k0 = 1, double kmin = 1e-2,
                               double m = 0.95);

    void register_observables();

    void optimize();

    void set_plugin(base_plugin* plug);
    size_t get_n_total();

   private:
    machine::rbm_base& rbm_;
    machine::abstract_sampler& sampler_;
    operators::base_op& hamiltonian_;
    operators::derivative_op derivative_;

    operators::aggregator a_h_;
    operators::aggregator a_d_;
    operators::prod_aggregator a_dh_;
    operators::outer_aggregator a_dd_;

    base_plugin* plug_;

    size_t n_total_;

    double lr_;
    double lrmin_;
    double lrm_;
    double m_;
    double kmin_;
    double kp_;

    bool reg_min_ = false;
    bool lr_min_ = true;

    std::vector<Eigen::MatrixXcd> dws_ = {};
};
}  // namespace optimizer
