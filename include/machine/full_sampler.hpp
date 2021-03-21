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

// #include <gmp.h>

#include <Eigen/Dense>
//
#include <machine/abstract_sampler.hpp>
#include <machine/rbm_base.hpp>

namespace machine {

/**
 * @brief full_sampler is a sampler which samples over all basis states.
 * It weights the results of aggregators with the probability of the
 * corresponding state.
 * It uses gray codes to visit all z-basis states of a spin 1/2 system, which
 * can be identified with a string of bits or a number.
 */
class full_sampler : public abstract_sampler {
    using Base = abstract_sampler;

    /**
     * @brief Number of bits used for parallel execution. `n_visible` -
     * `bits_parallel_` bits will be sampled in one parallel part of the
     * execution, e.g. if `bits_parallel_` = 3, get all combinations of the
     * first three bits 000, 001, 010, 100, 011, 101, 110, 111 and execute the
     * calculation of the remaining bits in parallel for each of combintations
     * of the first three bits, so 2^3 = 8 parrallel runs.
     */
    size_t bits_parallel_;

   protected:
    /**
     * @brief Convert the state in number (bit) representation into a Eigen
     * Matrix used as input for RBM. 0 -> -1. 1 -> 1..
     *
     * @param state_num The state in bit form.
     * @param state The Eigen Matrix reference, which will be filled.
     */
    void get_state(size_t state_num, Eigen::MatrixXcd& state);

    /**
     * @brief Tests if a particular bit is 1.
     *
     * @param s The number to test.
     * @param i The index of the bit.
     *
     * @return `true` if bit `i` of `s` is 1.
     */
    inline bool test_bit(size_t s, size_t i) { return (s >> i) & 1; };

   public:
    /**
     * @brief Full sampler constructor.
     *
     * @param rbm_base Reference to the RBM.
     * @param bits_parallel Number of parallel bits.
     */
    full_sampler(rbm_base&, size_t bits_parallel);

    virtual void sample(size_t n_samples = 0) override { sample(false); };

    /**
     * @brief Sample over all basis states.
     *
     * @param keep_state Keeps the state vector.
     */
    virtual void sample(bool keep_state);
};

}  // namespace machine
