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
 * @brief The Stochastic Reconfiguration opetimizer. It registers the
 * derivative operator and the hamiltonian and four aggregators of the
 * observables. The Covaraiance matrix `S` and the gradient descent vector `F`
 * are constructed and the RBM parameters are updated by `inv(S)*F`. S needs to
 * be regularized for stability by `S -> S + kp*I`. Also the pseudo inverse is
 * used.
 */
class stochastic_reconfiguration {
   public:
    /**
     * @brief Stochastic Reconfiguration Constructor
     *
     * @param rbm Reference to the RBM.
     * @param sampler Reference to the Sampler.
     * @param hamiltonian Reference to the Hamiltonian operator.
     * @param learning_rate Learing rate `ini::decay_t`.
     * @param regularization Regularization `ini::decay_t`.
     */
    stochastic_reconfiguration(machine::rbm_base& rbm,
                               machine::abstract_sampler& sampler,
                               operators::base_op& hamiltonian,
                               const ini::decay_t& learning_rate,
                               const ini::decay_t& regularization);

    /**
     * @brief Registers the observables (operators and aggregators)
     */
    void register_observables();

    /**
     * @brief Do the optimization step described in the class description.
     */
    void optimize();

    /**
     * @brief Add a plugin to the optimization.
     *
     * @param plug Pointer to the plugin.
     */
    void set_plugin(base_plugin* plug);

    /**
     * @brief Returns the current energy
     *
     * @return current energy of `a_h_`.
     */
    double get_current_energy();

   private:
    machine::rbm_base& rbm_;               ///< RBM reference
    machine::abstract_sampler& sampler_;   ///< Sampler reference
    operators::base_op& hamiltonian_;      ///< Hamiltonian operator reference
    operators::derivative_op derivative_;  ///< Derivative oprator

    operators::aggregator a_h_;  ///< Energy aggregator
    operators::aggregator a_d_;  ///< Derivative aggregator
    operators::prod_aggregator
        a_dh_;  ///< Derivative Hamiltonian aggregator <D^* H>
    operators::outer_aggregator
        a_dd_;  ///< Outer product derivative aggregator <D^* D^T>

    decay_t lr_;  ///< Learing rate
    decay_t kp_;  ///< Regularization

    base_plugin* plug_ = nullptr;  ///< Pointer to the Plugin
};
}  // namespace optimizer
