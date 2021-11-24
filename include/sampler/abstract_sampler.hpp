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
#include <random>
#include <vector>
//
#include <machine/abstract_machine.hpp>
#include <operators/aggregator.hpp>
#include <operators/base_op.hpp>
#include <operators/bond_op.hpp>
#include <operators/local_op_chain.hpp>

/**
 * @brief namespace machine hosts the classes related to the restricted
 * boltzmann machine and the sampling
 */
namespace sampler {

/**
 * @brief abstract_sampler is the base class to the samplers. It is capable
 * of registering operators and aggregators and stores the RBM base class.
 */
class abstract_sampler {
   protected:
    machine::abstract_machine& rbm_;  ///< The RBM reference

    std::vector<operators::base_op*> ops_;      ///< The vector of operators
    std::vector<operators::aggregator*> aggs_;  ///< The vector of aggregators
    std::vector<operators::local_op_chain*>
        chains_;  ///< The vector of aggregators

    size_t n_samples_;  ///< Number of samples

    int pfaff_refresh_;  ///< refresh pfaffian context every n loops
    int lut_exchange_;   ///< refresh pfaffian context every n loops

    /**
     * @brief Abstract sampler constructor.
     *
     * @param rbm The RBM object reference.
     */
    abstract_sampler(machine::abstract_machine& rbm, size_t n_samples,
                     int pfaff_refresh_ = 0, int lut_exchange_ = 0);

   public:
    /**
     * @brief default vitual destructor.
     */
    virtual ~abstract_sampler() = default;

    /**
     * @brief The sample function
     *
     */
    virtual void sample() = 0;

    /**
     * @brief Log message (needs to be invoked after optimization log).
     */
    virtual void log() {}

    /**
     * @brief Register a list of operators.
     *
     * @param ops A vector of operator pointers.
     */
    void register_ops(const std::vector<operators::base_op*>& ops);

    /**
     * @brief Register one operator.
     *
     * @param op A pointer to an operator.
     */
    void register_op(operators::base_op* op);

    /**
     * @brief Register one operator.
     *
     * @param op A pointer to an operator.
     */
    void register_op(operators::local_op_chain* op);
    /**
     * @brief Register one operator.
     *
     * @param op A pointer to an operator.
     */
    void register_op(operators::bond_op* op) {
        register_op(dynamic_cast<operators::local_op_chain*>(op));
    }

    /**
     * @brief Clear the registered operators.
     */
    void clear_ops();

    /**
     * @brief Register a list of aggregators.
     *
     * @param aggs A vector of aggregator pointers.
     */
    void register_aggs(const std::vector<operators::aggregator*>& aggs);

    /**
     * @brief Register one aggregator.
     *
     * @param agg The aggregator pointer.
     */
    void register_agg(operators::aggregator* agg);

    /**
     * @brief Clear the registered aggregators.
     */
    void clear_aggs();

    /**
     * @brief Get number of samples
     *
     * @return Number of samples
     */
    size_t get_n_samples() const;

    /**
     * @brief Get number of samples of current mpi process
     *
     * @return Number of samples
     */
    virtual size_t get_my_n_samples() const = 0;

    /**
     * @brief Set number of samples
     *
     * @param Number of samples
     */
    virtual void set_n_samples(size_t samples);

    void evaluate_and_aggregate(const machine::spin_state& state,
                                machine::rbm_context& context,
                                double p = 1.) const;

    bool pfaffian_refresh(const machine::spin_state& state,
                          machine::pfaff_context& context, int i,
                          const std::vector<size_t>& flips) const;

    void exchange_luts(int i) const;
};

}  // namespace sampler
