/**
 * src/optimizer/stochastic_reconfiguration.cpp
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

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <vector>
//
#include <machine/abstract_sampler.hpp>
#include <machine/rbm_base.hpp>
#include <operators/base_op.hpp>
#include <operators/derivative_op.hpp>
#include <optimizer/plugin.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>
#include <tools/logger.hpp>

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

stochastic_reconfiguration::stochastic_reconfiguration(
    machine::rbm_base& rbm, machine::abstract_sampler& sampler,
    operators::base_op& hamiltonian, const ini::decay_t& lr,
    const ini::decay_t& kp)
    : rbm_{rbm},
      sampler_{sampler},
      hamiltonian_{hamiltonian},
      derivative_{rbm.n_params},
      a_h_{hamiltonian_},
      a_d_{derivative_},
      a_dh_{derivative_, hamiltonian_},
      a_dd_{derivative_},
      lr_{lr, rbm_.get_n_updates()},
      kp_{kp, rbm_.get_n_updates()} {}

void stochastic_reconfiguration::register_observables() {
    sampler_.register_ops({&hamiltonian_, &derivative_});
    a_h_.track_variance();
    sampler_.register_aggs({&a_h_, &a_d_, &a_dh_, &a_dd_});
}

void stochastic_reconfiguration::optimize() {
    auto& h = a_h_.get_result();
    auto& d = a_d_.get_result();
    auto& dh = a_dh_.get_result();
    auto& dd = a_dd_.get_result();

    logger::log(std::real(h(0)) / rbm_.n_visible, "Energy");
    logger::log(std::real(a_h_.get_variance()(0)) / rbm_.n_visible,
                "Energy variance");

    Eigen::MatrixXcd F = dh - d.conjugate() * h(0);
    Eigen::MatrixXcd S = dd - d.conjugate() * d.transpose();

    S += kp_.get() * Eigen::MatrixXcd::Identity(rbm_.n_params, rbm_.n_params);
    Eigen::MatrixXcd dw = S.completeOrthogonalDecomposition().solve(F);

    if (!plug_) {
        dw *= lr_.get();
    } else {
        dw = lr_.get() * plug_->apply(dw);
    }

    rbm_.update_weights(dw);
}

void stochastic_reconfiguration::set_plugin(base_plugin* plug) { plug_ = plug; }

double stochastic_reconfiguration::get_current_energy() {
    return std::real(a_h_.get_result()(0));
}
