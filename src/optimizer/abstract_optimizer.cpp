/*
 * Copyright (C) 2021  David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <optimizer/abstract_optimizer.hpp>

using namespace optimizer;

decay_t::decay_t(double initial, double min, double decay, size_t offset)
    : initial{initial},
      min{min},
      decay{decay},
      value{std::pow(decay, offset) * initial} {
    if (decay == 1.) {
        min_reached = true;
    }
}
decay_t::decay_t(const ini::decay_t& other, size_t offset)
    : decay_t{other.initial, other.min, other.decay, offset} {}

double decay_t::get() {
    // Decay current value if is not min.
    if (!min_reached) {
        value *= decay;
        if (value < min) {
            min_reached = true;
            value = min;
        }
    }
    return value;
}

void decay_t::reset() {
    value = initial;
    min_reached = decay == 1.;
}

abstract_optimizer::abstract_optimizer(machine::rbm_base& rbm,
                                       machine::abstract_sampler& sampler,
                                       operators::base_op& hamiltonian,
                                       const ini::decay_t& learning_rate)
    : rbm_{rbm},
      sampler_{sampler},
      hamiltonian_{hamiltonian},
      // Initialze derivative operator.
      derivative_{rbm.get_n_params()},
      // Initialize the aggregators.
      a_h_{hamiltonian_},
      a_d_{derivative_},
      lr_{learning_rate, rbm_.get_n_updates()} {}

void abstract_optimizer::set_plugin(base_plugin* plug) { plug_ = plug; }
void abstract_optimizer::remove_plugin() { plug_ = nullptr; }

double abstract_optimizer::get_current_energy(double norm) {
    return std::real(a_h_.get_result().sum() / norm);
}

void abstract_optimizer::register_observables() {
    // Register operators and aggregators
    sampler_.register_ops({&hamiltonian_, &derivative_});
    sampler_.register_aggs({&a_h_, &a_d_});
}
