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

#include <complex>
#include <vector>
#include <unsupported/Eigen/KroneckerProduct>
//
#include <model/abstract_model.hpp>

namespace model {
// Definition of the 1 site sparse Pauli matrices.

const SparseXcd kron(const std::vector<const SparseXcd>& args) {
    SparseXcd so(1, 1);
    so.insert(0, 0) = 1;
    for (const auto& arg : args) so = kroneckerProduct(arg, so).eval();
    return so;
}

const SparseXcd sx() {
    SparseXcd ret(2, 2);
    ret.insert(0, 1) = 1;
    ret.insert(1, 0) = 1;
    return ret;
}

const SparseXcd sy() {
    SparseXcd ret(2, 2);
    ret.insert(0, 1) = std::complex<double>(0, -1);
    ret.insert(1, 0) = std::complex<double>(0, 1);
    return ret;
}

const SparseXcd sz() {
    SparseXcd ret(2, 2);
    ret.insert(0, 0) = 1;
    ret.insert(1, 1) = -1;
    return ret;
}

}  // namespace model

using namespace model;

void abstract_model::remove_helper_hamiltoian() {
    for(size_t i = 0; i < helpers_; i++) {
        hamiltonian_->pop_back();
    }
    helpers_ = 0;
}
