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
#include <operators/aggregator.hpp>
#include <optimizer/abstract_optimizer.hpp>
#include <optimizer/abstract_solver.hpp>
#include <optimizer/plugin.hpp>

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
     * @param regularization1 Regularization scale `ini::decay_t`.
     * @param regularization2 Regularization shift `ini::decay_t`.
     * @param iterative Use ConjugateGradient for Matrix inversion flag (default
     * true).
     * @param max_iterations Number of max iterations for iterative scheme
     */
    stochastic_reconfiguration(machine::abstract_machine& rbm,
                               sampler::abstract_sampler& sampler,
                               operators::base_op& hamiltonian,
                               const ini::decay_t& learning_rate,
                               const ini::decay_t& regularization1,
                               const ini::decay_t& regularization2,
                               const ini::decay_t& regularization1delta,
                               std::string method = "minresqlp",
                               size_t max_iterations = 0, double rtol = 0.0,
                               bool resample = false, double alpha1 = 2,
                               double alpha2 = 5, double alpha3 = 6);

    virtual void register_observables() override;

    virtual Eigen::MatrixXcd gradient(bool log = false) override;

   private:
    std::string method_;     ///< Use ConjugateGradient flag
    size_t max_iterations_;  ///< Number of maximum iterations
    double rtol_;            ///< Residue tolerance for iterative solution

    operators::prod_aggregator
        a_dh_;  ///< Derivative Hamiltonian aggregator <D^* H>
    operators::outer_aggregator_lazy
        a_dd_;  ///< Outer product derivative aggregator <D^* D^T>

    decay_t kp1_;   ///< Regularization scale
    decay_t kp2_;   ///< Regularization shift
    decay_t kp1d_;  ///< Regularization scale offset pfaffian parameters

    std::unique_ptr<abstract_solver> solver_;

    Eigen::VectorXcd F_;
    Eigen::MatrixXcd dw_;
};
}  // namespace optimizer
