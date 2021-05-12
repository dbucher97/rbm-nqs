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

#include <omp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <vector>
//
#include <machine/rbm_base.hpp>

namespace operators {

/**
 * @brief Base class of the operator. The operator has a result matrix for
 * each thread for parallelization.
 */
class base_op {
    size_t r_,                              ///< Number of rows
        c_;                                 ///< Number of cols
    std::vector<Eigen::MatrixXcd> result_;  ///< Vector of result matrices

   protected:
    /**
     * @brief Result getter
     *
     * @return Reference to the result matrix of the current thread.
     */
    inline Eigen::MatrixXcd& get_result_() {
        return result_[omp_get_thread_num()];
    }

   public:
    /**
     * @brief Operator constructor.
     *
     * @param r Number of rows
     * @param c Number of cols
     */
    base_op(size_t r = 1, size_t c = 1)
        : r_{r}, c_{c}, result_(omp_get_max_threads()) {
        // Initialize all result matrices.
        for (size_t i = 0; i < result_.size(); i++)
            result_[i] = Eigen::MatrixXcd(r_, c_);
    }
    /**
     * @brief Default virtual destructor.
     */
    virtual ~base_op() = default;

    /**
     * @brief Evaluater the operator vaule for the current sample. For
     * evaluation, not all of the parameters are needed, but provided for
     * convenience.
     *
     * @param rbm Reference to the RBM.
     * @param state Reference to the current state.
     * @param thetas Reference to the precalculated thetas.
     */
    virtual void evaluate(machine::rbm_base& rbm, const Eigen::MatrixXcd& state,
                          const Eigen::MatrixXcd& thetas) = 0;

    /**
     * @brief Checks if operator is scalar
     *
     * @return True if operator is scalar.
     */
    bool is_scalar() const { return r_ == 1 && c_ == 1; }
    /**
     * @brief Checks if operator is vector.
     *
     * @return True if operator is vector.
     */
    bool is_vector() const { return c_ == 1; }
    /**
     * @brief Rows getter.
     *
     * @return Number of rows.
     */
    size_t rows() const { return r_; }
    /**
     * @brief Cols getter
     *
     * @return Number of cols.
     */
    size_t cols() const { return c_; }

    /**
     * @brief Gets the result of the current thread.
     *
     * @return Reference to the result matrix of the current thread.
     */
    const Eigen::MatrixXcd& get_result() const {
        return result_[omp_get_thread_num()];
    }
};
}  // namespace operators
