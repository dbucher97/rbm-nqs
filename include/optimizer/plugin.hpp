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
//
#include <machine/abstract_machine.hpp>
#include <sampler/abstract_sampler.hpp>

/**
 * @brief Namespace for all optimizer related classes.
 */
namespace optimizer {

/**
 * @brief Abstract base_plugin for adaptive learning rates.
 */
class base_plugin {
   protected:
    Eigen::MatrixXcd* met1_ = NULL;
    Eigen::MatrixXcd* met2_ = NULL;

   public:
    /**
     * @brief Default virtual constructor
     */
    virtual ~base_plugin() = default;

    /**
     * @brief apply the learning rate modification
     *
     * @param dw The proposed update of weights
     *
     * @return The adaptive update of weights (without learning rate.)
     */
    virtual void apply(Eigen::VectorXcd& dw, double lr) = 0;

    virtual void add_metric(Eigen::MatrixXcd* met1, Eigen::MatrixXcd* met2) {
        met1_ = met1;
        met2_ = met2;
    }
};

/**
 * @brief A optimizer plugin, which does the adam optimization.
 */
class adam_plugin : public base_plugin {
    using Base = base_plugin;

    double beta1_,  ///< \beta_1 param
        beta2_,     ///< \beta_2 param
        eps_;       ///< \eps param
    size_t t_;      ///< Time attribute.

    Eigen::MatrixXcd m_;  ///< The first order momentum of t-1
    Eigen::MatrixXd wr_;  ///< The second order real momentum of t-1
    Eigen::MatrixXd wi_;  ///< The second order imag momentum of t-1

   public:
    /**
     * @brief The Adam Plugin optimizer
     *
     * @param l Number of Parameters.
     * @param beta1 \beta_1 param (Default = 0.9)
     * @param beta2 \beta_2 param (Default = 0.999)
     * @param eps \eps param (Default = 1e-8)
     */
    adam_plugin(size_t l, double beta1 = 0.9, double beta2 = 0.999,
                double eps = 1e-8);

    virtual void apply(Eigen::VectorXcd&, double lr) override;
};

/**
 * @brief The momentum plugin, using a first order momentum term for adaptive
 * optimization.
 */
class momentum_plugin : public base_plugin {
    using Base = base_plugin;

    double alpha_;  ///< \alpha param

    Eigen::MatrixXcd m_;  ///< The first order moment of last iteration.

   public:
    /**
     * @brief Momentum Plugin constructor.
     *
     * @param l Number of Parameters.
     * @param alpha \alpha param (Default = 0.1)
     */
    momentum_plugin(size_t l, double alpha = 0.1);

    virtual void apply(Eigen::VectorXcd&, double lr) override;
};

/**
 * @brief The Heun Plugin uses the two step Runge Kutta method based on Heuns
 * method. It needs to evaluate the gradient with a half step to compare it to
 * the full step gradient. This helps to estimate the curvature in the parameter
 * space.
 */
class heun_plugin : public base_plugin {
    using Base = base_plugin;

    const std::function<Eigen::VectorXcd&(void)>& gradient_;
    machine::abstract_machine& rbm_;
    sampler::abstract_sampler& sampler_;
    double eps_;

   public:
    heun_plugin(const std::function<Eigen::VectorXcd&(void)>& gradient,
                machine::abstract_machine& rbm,
                sampler::abstract_sampler& sampler, double eps_);

    virtual void apply(Eigen::VectorXcd&, double lr) override;
};

}  // namespace optimizer
