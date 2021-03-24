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
#include <fstream>

/**
 * @brief `ofstream` wrapper for saving a `Eigen::Matrix`.
 *
 * @param file `ofstream` reference.
 * @param mat reference to the Matrix.
 *
 * @return `ofstream` reference.
 */
std::ofstream& operator<<(std::ofstream& file, Eigen::MatrixXcd& mat);

/**
 * @brief `ifstream` wrapper for loading a `Eigen::Matrix`.
 *
 * @param file `ifstream` reference.
 * @param mat reference to the Matrix.
 *
 * @return `ifstream` reference.
 */
std::ifstream& operator>>(std::ifstream& file, Eigen::MatrixXcd& mat);
