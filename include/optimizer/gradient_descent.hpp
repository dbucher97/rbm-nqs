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
 * @brief Basic gradient descent.
 */
class gradient_descent : public abstract_optimizer {
    using Base = abstract_optimizer;

   public:
    /**
     * @brief Gradient Descent Constructor
     *
     * @param rbm Reference to the RBM.
     * @param sampler Reference to the Sampler.
     * @param hamiltonian Reference to the Hamiltonian operator.
     * @param learning_rate Learing rate `ini::decay_t`.
     * @param regularization Regularization `ini::decay_t`.
     */
    gradient_descent(machine::rbm_base& rbm, machine::abstract_sampler& sampler,
                     operators::base_op& hamiltonian,
                     const ini::decay_t& learning_rate,
                     double real_factor = 1.);

    virtual void register_observables() override;

    virtual void optimize() override;

   private:
    operators::prod_aggregator
        a_dh_;  ///< Derivative Hamiltonian aggregator <D^* H>

    double real_factor_;
};
}  // namespace optimizer
