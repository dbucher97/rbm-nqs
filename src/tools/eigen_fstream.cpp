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

#include <Eigen/Dense>
#include <fstream>
#include <stdexcept>
//
#include <tools/eigen_fstream.hpp>

std::ofstream& operator<<(std::ofstream& file, Eigen::MatrixXcd& mat) {
    size_t rows = mat.rows(), cols = mat.cols();
    // Save num rows and cols.
    file.write((char*)&rows, sizeof(size_t));
    file.write((char*)&cols, sizeof(size_t));
    // Save data of matrix.
    file.write((char*)mat.data(),
               rows * cols * sizeof(Eigen::MatrixXcd::Scalar));
    return file;
}

std::ifstream& operator>>(std::ifstream& file, Eigen::MatrixXcd& mat) {
    size_t rows, cols;
    // Load num rows and cols
    file.read((char*)&rows, sizeof(size_t));
    file.read((char*)&cols, sizeof(size_t));
    // Guard matrix `mat` to be of same size.
    if (rows != mat.rows() && cols != mat.cols()) {
        throw std::runtime_error(
            "Unable to load Eigen Matrix: dimension mismatch!");
    }
    // Load data of matrix into `mat`.
    file.read((char*)mat.data(),
              rows * cols * sizeof(Eigen::MatrixXcd::Scalar));
    return file;
}
