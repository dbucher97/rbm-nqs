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

#include <cstdio>
//
#include <machine/rbm_base.hpp>
#include <optimizer/abstract_optimizer.hpp>
#include <sampler/full_sampler.hpp>
#include <tools/mpi.hpp>

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

abstract_optimizer::abstract_optimizer(machine::abstract_machine& rbm,
                                       sampler::abstract_sampler& sampler,
                                       operators::local_op_chain& hamiltonian,
                                       const ini::decay_t& learning_rate,
                                       bool resample, double alpha1,
                                       double alpha2, double alpha3)
    : rbm_{rbm},
      sampler_{sampler},
      hamiltonian_{hamiltonian},
      // Initialze derivative operator.
      derivative_{rbm.get_n_params()},
      // Initialize the aggregators.
      a_h_{hamiltonian_, sampler.get_my_n_samples()},
      a_d_{derivative_, sampler.get_my_n_samples()},
      lr_{learning_rate, rbm_.get_n_updates()},
      resample_{resample},
      alpha1_{alpha1},
      alpha2_{alpha2},
      alpha3_{alpha3},
      dw_(rbm_.get_n_params()) {}

void abstract_optimizer::set_plugin(base_plugin* plug) { plug_ = plug; }
void abstract_optimizer::remove_plugin() { plug_ = nullptr; }

void abstract_optimizer::optimize() {
    double lr = lr_.get();
    // Check resample criteria
    if (resample_ && rbm_.get_n_updates() > 50) {
        std::complex<double> e = a_h_.get_result()(0);
        double d = std::sqrt(a_h_.get_variance()(0));
        int resample = 1;
        size_t rcount = 0;
        while (resample && last_energy_std_ != -1 && rcount < 3) {
            resample = 0;
            resample |= std::abs(std::real(last_energy_ - e) /
                                 rbm_.n_visible) >= alpha1_;
            resample |= (std::abs(std::imag(e)) >= alpha2_ * d) << 1;
            resample |= (d / last_energy_std_ >= alpha3_) << 2;
            resample |= (last_energy_std_ / d > alpha3_) << 3;
            // std::cout << "resample" << std::endl;

            if (resample) {
                mpi::cout << "resample " << resample << ", " << d << " "
                          << last_energy_std_ << mpi::endl;
                rcount++;
                // std::cout << "resample" << std::endl;

                // Undo last update
                rbm_.update_weights_nc(-last_update_);

                // Repeat last step update
                sampler_.sample();
                gradient(false);
                dw_ /= std::pow(2, rcount);
                // Apply plugin if set
                if (plug_) {
                    plug_->apply(dw_, lr);
                } else {
                    dw_ *= lr;
                }
                // Update weights
                rbm_.update_weights_nc(dw_);
                last_update_ = dw_;

                // Resample
                sampler_.sample();
                e = a_h_.get_result()(0);
                d = std::sqrt(a_h_.get_variance()(0));
            }
        }
        last_energy_ = e;
        last_energy_std_ = d;
        total_resamples_ += rcount;
    }

    gradient(true);

    std::complex<double> e = a_h_.get_result()(0) / (double)rbm_.n_visible;
    if (std::real(e - last_energy_) > 0.05 && rbm_.get_n_updates() > 10) {
        // mpi::cout << "whee" << mpi::endl;
        // sampler::full_sampler sa{rbm_, 2};
        // sa.register_op(&hamiltonian_);
        // sa.register_agg(&a_h_);
        // sa.sample(true);
        // mpi::cout << a_h_.get_result() << mpi::endl;
        // MPI_Barrier(MPI_COMM_WORLD);
        /*mpi::cout << a_h_.get_result() / rbm_.n_visible << mpi::endl;
        if (mpi::master) {
            std::rename((ini::name + ".state").c_str(),
                        (ini::name + ".new.state").c_str());
        }
        MPI_Barrier(MPI_COMM_WORLD);
        rbm_.update_weights_nc(-last_update_);
        sa.sample(true);
        mpi::cout << a_h_.get_result() / rbm_.n_visible << mpi::endl;
        rbm_.update_weights_nc(last_update_); */
        // sampler_.sample();
        // gradient(false);
    }
    last_energy_ = e;

    // Apply plugin if set
    if (plug_) {
        plug_->apply(dw_, lr);
    } else {
        dw_ *= lr;
    }

    // Update the weights.
    rbm_.update_weights(dw_);
    // if (resample_)
    last_update_ = dw_;
}

double abstract_optimizer::get_current_energy() {
    return std::real(a_h_.get_result()(0));
}

void abstract_optimizer::register_observables() {
    // Register operators and aggregators
    sampler_.register_op(&hamiltonian_);
    sampler_.register_op(&derivative_);
    a_h_.track_variance(10);
    sampler_.register_aggs({&a_h_, &a_d_});
}
