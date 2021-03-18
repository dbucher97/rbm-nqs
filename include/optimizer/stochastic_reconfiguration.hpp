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
#include <tools/ini.hpp>

namespace optimizer {

class decay_t {
    const double initial, min, decay;
    double value;
    bool min_reached = false;

   public:
    decay_t(double initial, double min, double decay, size_t = 0);
    decay_t(double initial) : decay_t{initial, initial, 1.} {}
    decay_t(const ini::decay_t& other, size_t = 0);
    double get();
    void reset();
};

class stochastic_reconfiguration {
   public:
    stochastic_reconfiguration(machine::rbm_base&, machine::abstract_sampler&,
                               operators::base_op&, const ini::decay_t&,
                               const ini::decay_t&);

    void register_observables();

    void optimize();

    void set_plugin(base_plugin* plug);

    double get_current_energy();

   private:
    machine::rbm_base& rbm_;
    machine::abstract_sampler& sampler_;
    operators::base_op& hamiltonian_;
    operators::derivative_op derivative_;

    operators::aggregator a_h_;
    operators::aggregator a_d_;
    operators::prod_aggregator a_dh_;
    operators::outer_aggregator a_dd_;

    decay_t lr_;
    decay_t kp_;

    base_plugin* plug_ = nullptr;
};
}  // namespace optimizer
