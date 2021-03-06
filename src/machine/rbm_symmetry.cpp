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

#include <cmath>
#include <complex>
#include <iostream>
#include <random>
//
#include <machine/rbm_symmetry.hpp>
#include <math.hpp>
#include <tools/state.hpp>
#include <tools/time_keeper.hpp>

using namespace machine;

rbm_symmetry::rbm_symmetry(size_t n_alpha, lattice::bravais& l, size_t pop_mode,
                           size_t cosh_mode)
    : Base{n_alpha, l.n_total / l.symmetry_size(), l, pop_mode, cosh_mode},
      symmetry_{lattice_.construct_symmetry()},
      basis_{lattice_.construct_symm_basis()},
      reverse_symm_(n_visible) {
    // std::cout << l.symmetry_size() << std::endl;
    for (size_t f = 0; f < n_visible; f++) {
        reverse_symm_[f] =
            Eigen::PermutationMatrix<Eigen::Dynamic>(symmetry_.size());

        for (size_t s = 0; s < symmetry_.size(); s++) {
            reverse_symm_[f].indices()(s) = symmetry_[s].indices()(f);
        }
    }
}

rbm_context rbm_symmetry::get_context(const spin_state& state) const {
    Eigen::MatrixXcd vstate = state.to_vec();
    // Get thetas with symmetries involved
    Eigen::MatrixXcd ret(n_alpha_, symmetry_.size());
    for (size_t s = 0; s < symmetry_.size(); s++) {
        ret.col(s) = weights_.transpose() * (symmetry_[s] * vstate) + h_bias_;
        for (auto& c : correlators_) c->add_thetas(state, ret, s);
    }
    if (pfaffian_) {
        return {ret, pfaffian_->get_context(state)};
    } else {
        return {ret};
    }
}

void rbm_symmetry::update_context(const spin_state& state,
                                  const std::vector<size_t>& flips,
                                  rbm_context& context) const {
    time_keeper::start("Update context");
    Eigen::MatrixXcd& thetas = context.thetas;
    // Same as base class but with symmetries involved
    std::vector<std::vector<size_t>> cidxs;
    for (auto& c : correlators_) {
        c->get_cidxs_from_flips(flips, cidxs);
    }

    // for (const auto& f : flips) {
    //     thetas.noalias() -=
    //         (2. * state(f)) * (weights_.transpose() * reverse_symm_[f]);
    // }
    for (size_t s = 0; s < symmetry_.size(); s++) {
        // Possibility for speedup thetas - perm matrix weights.transpose()
        // state
        for (auto& f : flips) {
            thetas.col(s) -=
                (state[f] ? -2. : 2.) *
                weights_.row(symmetry_[s].indices()(f)).transpose();
        }
        for (size_t i = 0; i < correlators_.size(); i++) {
            correlators_[i]->update_thetas(state, cidxs[i], thetas, s);
        }
    }

    if (pfaffian_) {
        pfaffian_->update_context(state, flips, context.pfaff());
    }
    time_keeper::end("Update context");
}

Eigen::MatrixXcd rbm_symmetry::derivative(const spin_state& state,
                                          const rbm_context& context) const {
    Eigen::MatrixXcd vstate = state.to_vec();
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(get_n_params(), 1);
    for (size_t i = 0; i < n_vb_; i++) {
        for (auto& s : symmetry_) {
            result(i) += vstate(s.indices()(basis_[i]));
        }
    }
    // result.block(0, 0, n_vb_, 1) =
    //     Eigen::Map<Eigen::MatrixXcd>(vstate.data(), n_vb_, symmetry_size())
    //         .rowwise()
    //         .sum();
    // Same as baseclass
    // Eigen::MatrixXcd tanh = thetas.array().tanh();
    Eigen::ArrayXXcd tanh(context.thetas.rows(), context.thetas.cols());
    (*tanh_)(context.thetas, tanh);
    result.block(n_vb_, 0, n_alpha_, 1) = tanh.rowwise().sum();
    size_t n_tot = n_visible * n_alpha_;
    // Symmetries involved
    for (size_t s = 0; s < symmetry_.size(); s++) {
        Eigen::MatrixXcd x =
            (symmetry_[s] * vstate) * tanh.col(s).matrix().transpose();
        result.block(n_vb_ + n_alpha_, 0, n_tot, 1) +=
            Eigen::Map<Eigen::MatrixXcd>(x.data(), n_tot, 1);
    }

    size_t offset = n_params_;
    for (auto& c : correlators_) c->derivative(state, tanh, result, offset);
    if (pfaffian_)
        pfaffian_->derivative(state, context.pfaff(), result, offset);

    return result;
}

void rbm_symmetry::add_correlator(
    const std::vector<std::vector<size_t>>& corr) {
    auto symm = lattice_.construct_uc_symmetry();
    correlators_.push_back(std::make_unique<correlator>(corr, n_alpha_, symm));
}

std::complex<double> rbm_symmetry::psi_notheta(const spin_state& state) const {
    Eigen::MatrixXcd vstate = state.to_vec();
    std::complex<double> result;
    for (size_t i = 0; i < n_vb_; i++) {
        for (auto& s : symmetry_) {
            result += v_bias_(i) * vstate(s.indices()(basis_[i]));
        }
    }
    return std::exp(result);
}

// std::complex<double> rbm_symmetry::log_psi_over_psi(
//     const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
//     rbm_context& context, rbm_context& updated_context) {
//     if (flips.empty()) return 0.;

//     // Just adjusted for the single v_bias
//     // std::complex<double> ret = 0;
//     // for (auto& f : flips) ret -= 2. * state(f) * v_bias_(f % n_vb_);

//     // std::vector<std::vector<size_t>> cidxs;
//     // for (auto& c : correlators_) {
//     //     c->get_cidxs_from_flips(flips, cidxs);
//     //     ret += c->log_psi_over_psi(state, *(cidxs.end() - 1));
//     // }
//     std::complex<double> ret = log_psi_over_psi_bias(state, flips);

//     // Same as base class
//     update_context(state, flips, updated_context);

//     size_t num = tools::state_to_num(state);
//     size_t num2 = num;
//     for (auto& f : flips) num2 ^= (1 << f);
//     ret += lncosh(updated_context, num2) - lncosh(context, num);
//     // ret += std::log(
//     //     (updated_context.thetas.array().cosh() /
//     //     context.thetas.array().cosh())
//     //         .prod());

//     return ret;
// }

// std::complex<double> rbm_symmetry::psi_over_psi_alt(
//     const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
//     rbm_context& context, rbm_context& updated_context) {
//     if (flips.empty()) return 1.;

//     // Just adjusted for the single v_bias
//     std::complex<double> ret = 1;
//     for (auto& f : flips) ret *= std::exp(-2. * state(f) * v_bias_(f %
//     n_vb_));

//     std::vector<std::vector<size_t>> cidxs;
//     for (auto& c : correlators_) {
//         c->get_cidxs_from_flips(flips, cidxs);
//         ret *= std::exp(c->log_psi_over_psi(state, *(cidxs.end() - 1)));
//     }

//     // Same as base class
//     update_context(state, flips, updated_context);

//     size_t num = tools::state_to_num(state);
//     size_t num2 = num;
//     for (auto& f : flips) num2 ^= (1 << f);
//     ret *= cosh(updated_context, num2) / cosh(context, num);
//     // ret *= (updated_context.thetas.array().cosh() /
//     // context.thetas.array().cosh()).prod();

//     return ret;
// }
