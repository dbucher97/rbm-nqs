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

#include <array>
#include <complex>
#include <vector>
//
#include <model/abstract_model.hpp>

/**
 * @brief Namespace for the physical models.
 */
namespace model {

typedef Eigen::Matrix<std::complex<double>, 2, 2> Matrix2cd;
typedef Eigen::Matrix<std::complex<double>, 16, 16> Matrix16cd;

extern Matrix2cd sx;       ///< 1 site Pauli-x operator
extern Matrix2cd sy;       ///< 1 site Pauli-z operator
extern Matrix16cd vertex;  ///< 4 site Pauli-x operator
extern Matrix16cd plaq;    ///< 4 site Pauli-z operator

/**
 * @brief The Kitaev toric code Model.
 */
class toric : public abstract_model {
    using Base = abstract_model;

    Eigen::MatrixXcd plaq_, vertex_;

   public:
    /**
     * @brief Kitaev toric code model consturctor.
     *
     * @param size Number of unitcells in one direction.
     * @param J the coupling in all three directions
     */
    toric(size_t size, double J);
};
}  // namespace model
