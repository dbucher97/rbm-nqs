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
#include <optimizer/abstract_optimizer.hpp>
#include <optimizer/plugin.hpp>
#include <tools/ini.hpp>

namespace optimizer {

/**
 * @brief The Stochastic Reconfiguration opetimizer. It registers the
 * derivative operator and the hamiltonian and four aggregators of the
 * observables. The Covaraiance matrix `S` and the gradient descent vector `F`
 * are constructed and the RBM parameters are updated by `inv(S)*F`. S needs to
 * be regularized for stability by `S -> S + kp*I`. Also the pseudo inverse is
 * used.
 */
class stochastic_reconfiguration : public abstract_optimizer {
    using Base = abstract_optimizer;

   public:
    /**
     * @brief Stochastic Reconfiguration Constructor
     *
     * @param rbm Reference to the RBM.
     * @param sampler Reference to the Sampler.
     * @param hamiltonian Reference to the Hamiltonian operator.
     * @param learning_rate Learing rate `ini::decay_t`.
     * @param regularization Regularization `ini::decay_t`.
     * @param use_gmres Use GMRES for Matrix inversion flag (default true).
     */
    stochastic_reconfiguration(machine::rbm_base& rbm,
                               machine::abstract_sampler& sampler,
                               operators::base_op& hamiltonian,
                               const ini::decay_t& learning_rate,
                               const ini::decay_t& regularization,
                               bool use_gmres = true);

    virtual void register_observables() override;

    virtual void optimize() override;

   private:
    bool use_gmres_; ///< Use GMRES flag

    operators::prod_aggregator
        a_dh_;  ///< Derivative Hamiltonian aggregator <D^* H>
    std::unique_ptr<operators::aggregator>
        a_dd_;  ///< Outer product derivative aggregator <D^* D^T>

    decay_t kp_;  ///< Regularization

};
}  // namespace optimizer
