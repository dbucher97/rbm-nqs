/*
 * Copyright (C) 2021  David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <Eigen/Dense>
//
#include <optimizer/gradient_descent.hpp>
#include <tools/logger.hpp>

using namespace optimizer;

gradient_descent::gradient_descent(machine::rbm_base& rbm,
                                   machine::abstract_sampler& sampler,
                                   operators::base_op& hamiltonian,
                                   const ini::decay_t& lr)
    : Base{rbm, sampler, hamiltonian, lr} {}

void gradient_descent::optimize(double norm) {
    // Get the result
    auto& hr = a_h_.get_result();
    auto& dr = a_d_.get_result();

    std::complex<double> h = hr.sum() / norm;
    Eigen::MatrixXcd d = dr.rowwise().sum() / norm;
    Eigen::MatrixXcd dh =
        (dr.conjugate().array().rowwise() * hr.row(0).array()).rowwise().sum() /
        norm;
    // std::cout << hr << std::endl;
    // std::cout << std::endl;

    std::cout << norm << std::endl;

    // Log energy, energy variance and sampler properties.
    logger::log(std::real(h) / rbm_.n_visible, "Energy");
    sampler_.log();

    // Calculate the gradient descent
    Eigen::MatrixXcd dw = dh - d.conjugate() * h;

    // Apply plugin if set
    if (!plug_) {
        dw *= lr_.get();
    } else {
        dw = lr_.get() * plug_->apply(dw);
    }
    dw.real() /= 2.;

    // Update the weights.
    rbm_.update_weights(dw);
}
