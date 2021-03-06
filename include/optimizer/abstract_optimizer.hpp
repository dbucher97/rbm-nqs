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

#pragma once

#include <machine/abstract_machine.hpp>
#include <operators/base_op.hpp>
#include <operators/derivative_op.hpp>
#include <optimizer/plugin.hpp>
#include <sampler/abstract_sampler.hpp>
#include <tools/ini.hpp>

namespace optimizer {

/**
 * @brief A decay type used for learning rate and regularization strength.
 * Each optimization step, the decay_t decays by the factor `decay` until
 * it reaches `min`.
 */
class decay_t {
    const double initial,      ///< initial value.
        min,                   ///< min value.
        decay;                 ///< decay rate.
    double value;              ///< current value.
    bool min_reached = false;  ///< flag if `vaule == min`

   public:
    /**
     * @brief decay_t constructor.
     *
     * @param initial Initial value.
     * @param min Minimum value.
     * @param decay Deacy rate.
     * @param n_updates Number of updates offset.
     */
    decay_t(double initial, double min, double decay, size_t n_updates = 0);
    /**
     * @brief decay_t constructor with no decay, just constant value.
     *
     * @param initial Initial value.
     */
    decay_t(double initial) : decay_t{initial, initial, 1.} {}
    /**
     * @brief decay_t constructor from `ini::decay_t`
     *
     * @param other `ini::decay_t`
     * @param n_updates Number of updates offset.
     */
    decay_t(const ini::decay_t& other, size_t n_updates = 0);
    /**
     * @brief Returns the new decayed value.
     *
     * @return Current value.
     */
    double get();
    /**
     * @brief Reset the state to initial.
     */
    void reset();
};

/**
 * @brief optimizer base class
 */
class abstract_optimizer {
   protected:
    machine::abstract_machine& rbm_;      ///< RBM reference
    sampler::abstract_sampler& sampler_;  ///< Sampler reference
    operators::local_op_chain&
        hamiltonian_;                      ///< Hamiltonian operator reference
    operators::derivative_op derivative_;  ///< Derivative oprator

    operators::aggregator a_h_;  ///< Energy aggregator
    operators::aggregator a_d_;  ///< Derivative aggregator

    decay_t lr_;                   ///< Learing rate
    base_plugin* plug_ = nullptr;  ///< Pointer to the Plugin

    bool resample_;
    double alpha1_, alpha2_, alpha3_;

    std::complex<double> last_energy_ = 0;
    double last_energy_std_ = -1;
    Eigen::MatrixXcd last_update_;
    size_t total_resamples_ = 0;

    Eigen::VectorXcd dw_;
    /**
     * @brief Protected Abstract optimizer constructor.
     *
     * @param rbm RBM reference
     * @param sampler Sampler reference
     * @param hamiltonian Hamiltonian reference
     * @param learning_rate Learing rate `ini::decay_t`
     */
    abstract_optimizer(machine::abstract_machine& rbm,
                       sampler::abstract_sampler& sampler,
                       operators::local_op_chain& hamiltonian,
                       const ini::decay_t& learning_rate, bool resample = false,
                       double alpha1 = 2, double alpha2 = 5, double alpha3 = 6);

   public:
    /**
     * @brief Default abstract destructor.
     */
    virtual ~abstract_optimizer() = default;

    /**
     * @brief Registers the observables (operators and aggregators)
     */
    virtual void register_observables();

    /**
     * @brief Do the optimization step described in the class description.
     */
    virtual Eigen::VectorXcd& gradient(bool log = false) = 0;

    /**
     * @brief Do the optimization step described in the class description.
     */
    virtual bool optimize();

    /**
     * @brief Add a plugin to the optimization.
     *
     * @param plug Pointer to the plugin.
     */
    void set_plugin(base_plugin* plug);

    /**
     * @brief Removes the plugin.
     */
    void remove_plugin();

    /**
     * @brief Returns the current energy
     *
     * @return current energy of `a_h_`.
     */
    double get_current_energy();

    size_t get_total_resamples() { return total_resamples_; }
};
}  // namespace optimizer
