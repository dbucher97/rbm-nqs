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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
//
#include <machine/rbm_base.hpp>
#include <math.hpp>
#include <tools/eigen_fstream.hpp>

using namespace machine;

rbm_base::rbm_base(size_t n_alpha, size_t n_v_bias, lattice::bravais& l,
                   size_t pop_mode, size_t cosh_mode)
    : n_alpha{n_alpha},
      n_visible{l.n_total},
      n_params_{n_v_bias + n_alpha + n_alpha * n_visible},
      lattice_{l},
      weights_(n_visible, n_alpha),
      h_bias_(n_alpha, 1),
      v_bias_(n_v_bias, 1),
      n_vb_{n_v_bias},
      n_updates_{0},
      psi_{pop_mode == 0 ? &rbm_base::psi_default : &rbm_base::psi_alt},
      psi_over_psi_{pop_mode == 0 ? &rbm_base::psi_over_psi_default
                                  : &rbm_base::psi_over_psi_alt},
      cosh_{(pop_mode == 0 || cosh_mode == 0) ? &math::cosh1 : &math::cosh2},
      tanh_{(pop_mode == 0 || cosh_mode == 0) ? &math::tanh1 : &math::tanh2} {}

rbm_base::rbm_base(size_t n_alpha, lattice::bravais& l, size_t pop_mode,
                   size_t cosh_mode)
    : rbm_base{n_alpha, l.n_total, l, pop_mode, cosh_mode} {}

size_t rbm_base::get_n_params() const {
    size_t x = n_params_;
    for (auto& c : correlators_) {
        x += c->get_n_params();
    }
    return x;
}

void rbm_base::initialize_weights(std::mt19937& rng, double std_dev,
                                  double std_dev_imag) {
    n_updates_ = 0;

    // If std_dev_imag < 0, use the normal std_dev;
    if (std_dev_imag < 0) std_dev_imag = std_dev;

    // Initialize the normal distribution
    std::normal_distribution<double> real_dist{0, std_dev};
    std::normal_distribution<double> imag_dist{0, std_dev_imag};

    // Fill all weigthts and biases
    for (size_t i = 0; i < n_vb_; i++) {
        v_bias_(i) = std::complex<double>(real_dist(rng), imag_dist(rng));
    }
    for (size_t i = 0; i < n_alpha; i++) {
        h_bias_(i) = std::complex<double>(real_dist(rng), imag_dist(rng));
        for (size_t j = 0; j < n_visible; j++) {
            weights_(j, i) =
                std::complex<double>(real_dist(rng), imag_dist(rng));
        }
    }

    for (auto& c : correlators_)
        c->initialize_weights(rng, std_dev, std_dev_imag);
}

void rbm_base::update_weights(const Eigen::MatrixXcd& dw) {
    // Update the weights with the `dw` of size `n_params`
    v_bias_ -= dw.block(0, 0, n_vb_, 1);
    h_bias_ -= dw.block(n_vb_, 0, n_alpha, 1);
    // Turn vector of size `n_alpha` * `n_visible` into matrix `n_alpha` x
    // `n_visible`
    Eigen::MatrixXcd dww = dw.block(n_vb_ + n_alpha, 0, n_alpha * n_visible, 1);
    weights_ -= Eigen::Map<Eigen::MatrixXcd>(dww.data(), n_visible, n_alpha);

    size_t offset = n_params_;
    for (auto& c : correlators_) {
        c->update_weights(dw, offset);
    }

    // Increment updates tracker.
    n_updates_++;
}

Eigen::MatrixXcd rbm_base::get_thetas(const Eigen::MatrixXcd& state) const {
    // Calculate the thetas from `state`
    Eigen::MatrixXcd ret = (state.transpose() * weights_).transpose() + h_bias_;
    for (auto& c : correlators_) c->add_thetas(state, ret);
    return ret;
}

void rbm_base::update_thetas(const Eigen::MatrixXcd& state,
                             const std::vector<size_t>& flips,
                             Eigen::MatrixXcd& thetas) const {
    // Update the thetas for a given number of flips
    for (auto& f : flips) {
        // Just subtract a row from weights from the thetas
        thetas -= 2 * weights_.row(f).transpose() * state(f);
    }
    std::vector<std::vector<size_t>> cidxs;
    for (auto& c : correlators_) {
        c->get_cidxs_from_flips(flips, cidxs);
        c->update_thetas(state, *(cidxs.end() - 1), thetas);
    }
}

Eigen::MatrixXcd rbm_base::derivative(const Eigen::MatrixXcd& state,
                                      const Eigen::MatrixXcd& thetas) const {
    // Calculate thr derivative of the RBM with respect to the parameters.
    // The formula for this can be calculated by pen and paper.
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(get_n_params(), 1);
    result.block(0, 0, n_vb_, 1) = state;
    // Eigen::MatrixXcd tanh = thetas.array().tanh();
    Eigen::MatrixXcd tanh = (*tanh_)(thetas);
    result.block(n_vb_, 0, n_alpha, 1) = tanh;
    Eigen::MatrixXcd x = state * tanh.transpose();
    // Transform weights matrix into a vector.
    result.block(n_vb_ + n_alpha, 0, n_alpha * n_visible, 1) =
        Eigen::Map<Eigen::MatrixXcd>(x.data(), n_alpha * n_visible, 1);

    size_t offset = n_params_;
    for (auto& c : correlators_) c->derivative(state, tanh, result, offset);

    return result;
}

bool rbm_base::flips_accepted(double prob, const Eigen::MatrixXcd& state,
                              const std::vector<size_t>& flips,
                              Eigen::MatrixXcd& thetas) const {
    // First copy the old thetas
    Eigen::MatrixXcd new_thetas = thetas;

    // Calculate the acceptance value
    double a =
        std::pow(std::abs(psi_over_psi(state, flips, thetas, new_thetas)), 2);

    // accept with given probability
    if (prob < a) {
        // Update the thetas
        thetas = new_thetas;
        return true;
    } else {
        return false;
    }
}

bool rbm_base::save(const std::string& name) {
    // Open the output stream
    std::ofstream output{name + ".rbm", std::ios::binary};
    if (output.is_open()) {
        // Write the matrices into the outputstream. (<eigen_fstream.h>)
        output << weights_ << h_bias_ << v_bias_;

        // Write `n_updates_` into the outputstream.
        output.write((char*)&n_updates_, sizeof(size_t));

        for (auto& c : correlators_) c->save(output);

        output.close();
        // Give a status update.
        std::cout << "Saved RBM to '" << name << ".rbm'!" << std::endl;
        return true;
    } else {
        return false;
    }
}

bool rbm_base::load(const std::string& name) {
    // Open the input stream
    std::ifstream input{name + ".rbm", std::ios::binary};
    if (input.good()) {
        // Read the matrices from the inputstream. (<eigen_fstream.h>)
        input >> weights_ >> h_bias_ >> v_bias_;

        // Read the n_updates_ from the inputstream.
        input.read((char*)&n_updates_, sizeof(size_t));

        for (auto& c : correlators_) c->load(input);

        input.close();

        // Give a status update.
        std::cout << "Loaded RBM from '" << name << ".rbm'!" << std::endl;
        return true;
    } else {
        return false;
    }
}

void rbm_base::add_correlator(const std::vector<std::vector<size_t>>& corr) {
    correlators_.push_back(std::make_unique<correlator>(corr, n_alpha));
}

void rbm_base::add_correlators(
    const std::vector<std::vector<std::vector<size_t>>>& corr) {
    for (const auto& c : corr) {
        add_correlator(c);
    }
}

//
// Overrideable functions, specific to the basic RBM.
//

std::complex<double> rbm_base::psi_notheta(
    const Eigen::MatrixXcd& state) const {
    std::complex<double> corr_part = 1.;
    for (auto& c : correlators_) c->psi(state, corr_part);
    return corr_part * (v_bias_.array() * state.array()).exp().prod();
}

std::complex<double> rbm_base::psi_default(
    const Eigen::MatrixXcd& state, const Eigen::MatrixXcd& thetas) const {
    // Calculate the \psi with `thetas`
    std::complex<double> cosh_part = math::lncosh(thetas).sum();
    return psi_notheta(state) * std::exp(cosh_part);
}

std::complex<double> rbm_base::psi_alt(const Eigen::MatrixXcd& state,
                                       const Eigen::MatrixXcd& thetas) const {
    // Calculate the \psi with `thetas`
    std::complex<double> cosh_part = (*cosh_)(thetas).array().prod();
    return psi_notheta(state) * cosh_part;
}

std::complex<double> rbm_base::log_psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    const Eigen::MatrixXcd& thetas, Eigen::MatrixXcd& updated_thetas) const {
    if (flips.empty()) return 0.;

    std::complex<double> ret = 0;
    // Claculate the visible bias part, calcels out for all not flipped sites.
    for (auto& f : flips) ret -= 2. * state(f) * v_bias_(f);

    std::vector<std::vector<size_t>> cidxs;
    for (auto& c : correlators_) {
        c->get_cidxs_from_flips(flips, cidxs);
        ret += c->log_psi_over_psi(state, *(cidxs.end() - 1));
    }

    // Update the thetas with the flips
    update_thetas(state, flips, updated_thetas);

    // Caclulate the diffrenece of the lncoshs, which is the same as the log
    // of the ratio of coshes.
    ret += (math::lncosh(updated_thetas) - math::lncosh(thetas)).sum();

    return ret;
}

std::complex<double> rbm_base::psi_over_psi_alt(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    const Eigen::MatrixXcd& thetas, Eigen::MatrixXcd& updated_thetas) const {
    if (flips.empty()) return 1.;

    std::complex<double> ret = 1;
    // Claculate the visible bias part, calcels out for all not flipped sites.
    for (auto& f : flips) ret *= std::exp(-2. * state(f) * v_bias_(f));

    std::vector<std::vector<size_t>> cidxs;
    for (auto& c : correlators_) {
        c->get_cidxs_from_flips(flips, cidxs);
        ret *= std::exp(c->log_psi_over_psi(state, *(cidxs.end() - 1)));
    }
    // Update the thetas with the flips
    update_thetas(state, flips, updated_thetas);

    ret *= ((*cosh_)(updated_thetas).array() / (*cosh_)(thetas).array()).prod();

    return ret;
}

