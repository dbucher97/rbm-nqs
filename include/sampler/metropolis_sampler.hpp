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
#include <sampler/abstract_sampler.hpp>

namespace sampler {

/**
 * @brief A Metropolis sampler for the RBM. Capable of doing Metropolis
 * sampling with a numnber of Markov chains.
 */
class metropolis_sampler : public abstract_sampler {
    using Base = abstract_sampler;

    std::mt19937& rng_;  ///< Referenec to the RNG;

    size_t n_chains_;      ///< Number of chains.
    size_t step_size_;     ///< The steps taken between two samples
    size_t warmup_steps_;  ///< The number of steps in the beginning before
                           ///< sampling.
    double bond_flips_;    ///< use bond flip for update proposal.

    double acceptance_rate_ = 0;  ///< The acceptance rate of all chains

    std::uniform_int_distribution<size_t>
        f_dist_;  ///< The flip distribution, which bit should get flipped

    std::uniform_real_distribution<double> u_dist_{
        0, 1};  ///< Uniform distribution for accepting a new state.

    /**
     * @brief Sample a Markov chain.
     *
     * @param n_samples Nmber of samples which this chain generates.
     *
     * @return Acceptance rate of this chain.
     */
    double sample_chain(size_t n_samples);

   public:
    /**
     * @brief The Metropolis sampler constructor.
     *
     * @param rbm The RBM reference.
     * @param n_samples Number of samples.
     * @param rng The RNG reference.
     * @param n_chains The number of Markov chains
     * @param step_size The steps taken between two samples
     * @param warmup_steps The warmup steps before sampling
     * @param bond_flips Use bond flips for update proposal
     */
    metropolis_sampler(machine::abstract_machine& rbm, size_t n_samples,
                       std::mt19937& rng, size_t n_chains = 1,
                       size_t step_size = 5, size_t warmup_steps = 100,
                       double bond_flips = 0.5, int refresh = 0);

    virtual void sample() override;

    virtual void log() override;

    /**
     * @brief Returns the acceptance rate.
     *
     * @return accepance rate.
     */
    double get_acceptance_rate() { return acceptance_rate_; }

    virtual size_t get_my_n_samples() const override;
};

}  // namespace sampler
