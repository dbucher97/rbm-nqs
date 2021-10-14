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
 * You should have received a copy of the GNU Affero General Public License *
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.  *
 */

#include <limits.h>
#include <mpi.h>

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
//
#include <machine/rbm_base.hpp>
#include <math.hpp>
#include <tools/eigen_fstream.hpp>
#include <tools/mpi.hpp>
#include <tools/state.hpp>
#include <tools/time_keeper.hpp>

using namespace machine;

rbm_base::rbm_base(size_t n_alpha, size_t n_v_bias, lattice::bravais& l,
                   size_t pop_mode, size_t cosh_mode)
    : Base{l, n_v_bias + (n_alpha + n_alpha * l.n_total) * n_v_bias},
      n_alpha_{n_alpha * n_v_bias},
      weights_(n_visible, n_alpha_),
      h_bias_(n_alpha_, 1),
      v_bias_(n_v_bias, 1),
      n_vb_{n_v_bias},
      psi_over_psi_{pop_mode == 0 ? &rbm_base::psi_over_psi_default
                                  : &rbm_base::psi_over_psi_alt},
      cosh_mode_{cosh_mode},
      cosh_{(cosh_mode == 0) ? &math::cosh1 : &math::cosh2},
      lncosh_{&math::lncosh},
      tanh_{(cosh_mode != 0) ? &math::tanh1 : &math::tanh2} {}

rbm_base::rbm_base(size_t n_alpha, lattice::bravais& l, size_t pop_mode,
                   size_t cosh_mode)
    : rbm_base{n_alpha, l.n_total, l, pop_mode, cosh_mode} {}

void rbm_base::initialize_weights(std::mt19937& rng, double std_dev,
                                  double std_dev_imag,
                                  const std::string& init_type) {
    n_updates_ = 0;

    if (mpi::master) {
        // If std_dev_imag < 0, use the normal std_dev;
        if (std_dev_imag < 0) std_dev_imag = std_dev;

        // Initialize the normal distribution
        std::normal_distribution<double> real_dist{0, std_dev};
        std::normal_distribution<double> imag_dist{0, std_dev_imag};

        // Fill all weigthts and biases
        if (n_vb_ == n_visible &&
            lattice_.supports_custom_weight_initialization() &&
            init_type != "") {
            lattice_.initialize_vb(init_type, v_bias_);
        } else {
            for (size_t i = 0; i < n_vb_; i++) {
                v_bias_(i) =
                    std::complex<double>(real_dist(rng), imag_dist(rng));
            }
        }
        for (size_t i = 0; i < n_alpha_; i++) {
            h_bias_(i) = std::complex<double>(real_dist(rng), imag_dist(rng));
            for (size_t j = 0; j < n_visible; j++) {
                weights_(j, i) =
                    std::complex<double>(real_dist(rng), imag_dist(rng));
            }
        }
    }
    MPI_Bcast(v_bias_.data(), v_bias_.size(), MPI_DOUBLE_COMPLEX, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(h_bias_.data(), h_bias_.size(), MPI_DOUBLE_COMPLEX, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(weights_.data(), weights_.size(), MPI_DOUBLE_COMPLEX, 0,
              MPI_COMM_WORLD);

    // Initialize weights
    for (auto& c : correlators_)
        c->initialize_weights(rng, std_dev, std_dev_imag);
}

int g_lut = 0, g_tot = 0;

void rbm_base::update_weights(const Eigen::MatrixXcd& dw) {
    // Update the weights with the `dw` of size `n_params`
    v_bias_ -= dw.block(0, 0, n_vb_, 1);
    h_bias_ -= dw.block(n_vb_, 0, n_alpha_, 1);
    // Turn vector of size `n_alpha` * `n_visible` into matrix `n_alpha` x
    // `n_visible`
    Eigen::MatrixXcd dww =
        dw.block(n_vb_ + n_alpha_, 0, n_alpha_ * n_visible, 1);
    weights_ -= Eigen::Map<Eigen::MatrixXcd>(dww.data(), n_visible, n_alpha_);

    size_t offset = n_params_;
    for (auto& c : correlators_) {
        c->update_weights(dw, offset);
    }
    if (pfaffian_) pfaffian_->update_weights(dw, offset);

    // Increment updates tracker.
    n_updates_++;
    lut_.clear();
    lut_update_nums_.clear();
    lut_update_vals_.clear();
    // mpi::cout << "\n" << (double)g_lut / (double)g_tot << mpi::endl;
    g_lut = 0;
    g_tot = 0;
}

std::complex<double> rbm_base::psi_over_psi(const Eigen::MatrixXcd& state,
                                            const std::vector<size_t>& flips,
                                            rbm_context& context,
                                            rbm_context& updated_context,
                                            bool* didupdate) {
    time_keeper::start("PoP");
    std::complex<double> ret = (this->*psi_over_psi_)(
        state, flips, context, updated_context, didupdate);
    if (pfaffian_) {
        ret *= pfaffian_->psi_over_psi(state, flips, context.pfaff(),
                                       updated_context.pfaff());
    }
    time_keeper::end("PoP");
    return ret;
}

rbm_context rbm_base::get_context(const Eigen::MatrixXcd& state) const {
    // Calculate the thetas from `state`
    Eigen::MatrixXcd thetas =
        (state.transpose() * weights_).transpose() + h_bias_;
    for (auto& c : correlators_) c->add_thetas(state, thetas);
    if (pfaffian_) {
        return {thetas, pfaffian_->get_context(state)};
    } else {
        return {thetas};
    }
}

void rbm_base::update_context(const Eigen::MatrixXcd& state,
                              const std::vector<size_t>& flips,
                              rbm_context& context) const {
    Eigen::MatrixXcd& thetas = context.thetas;
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

    if (pfaffian_) {
        pfaffian_->update_context(state, flips, context.pfaff());
    }
}

Eigen::MatrixXcd rbm_base::derivative(const Eigen::MatrixXcd& state,
                                      const rbm_context& context) const {
    // Calculate thr derivative of the RBM with respect to the parameters.
    // The formula for this can be calculated by pen and paper.
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(get_n_params(), 1);
    result.block(0, 0, n_vb_, 1) = state;
    // Eigen::MatrixXcd tanh = thetas.array().tanh();
    Eigen::ArrayXXcd tanh(context.thetas.rows(), context.thetas.cols());
    (*tanh_)(context.thetas, tanh);

    result.block(n_vb_, 0, n_alpha_, 1) = tanh;
    Eigen::ArrayXXcd x = state * tanh.matrix().transpose();
    // Transform weights matrix into a vector.
    result.block(n_vb_ + n_alpha_, 0, n_alpha_ * n_visible, 1) =
        Eigen::Map<Eigen::ArrayXXcd>(x.data(), n_alpha_ * n_visible, 1);

    size_t offset = n_params_;
    for (auto& c : correlators_) c->derivative(state, tanh, result, offset);
    if (pfaffian_)
        pfaffian_->derivative(state, context.pfaff(), result, offset);

    return result;
}

bool rbm_base::save(const std::string& name, bool silent) {
    if (!mpi::master) return true;
    // Open the output stream
    std::ofstream output{name + ".rbm", std::ios::binary};
    if (output.is_open()) {
        // Write the matrices into the outputstream. (<eigen_fstream.h>)
        output << weights_ << h_bias_ << v_bias_;

        // Write `n_updates_` into the outputstream.
        output.write((char*)&n_updates_, sizeof(size_t));

        for (auto& c : correlators_) c->save(output);

        if (pfaffian_) pfaffian_->save(output);

        output.close();
        // Give a status update.
        if (!silent)
            std::cout << "Saved RBM to '" << name << ".rbm'!" << std::endl;
        return true;
    } else {
        return false;
    }
}

bool rbm_base::load(const std::string& name) {
    bool rc = false;
    if (mpi::master) {
        // Open the input stream
        std::ifstream input{name + ".rbm", std::ios::binary};
        if (input.good()) {
            // Read the matrices from the inputstream. (<eigen_fstream.h>)
            input >> weights_ >> h_bias_ >> v_bias_;

            // Read the n_updates_ from the inputstream.
            input.read((char*)&n_updates_, sizeof(size_t));

            for (auto& c : correlators_) c->load(input);

            if (pfaffian_) pfaffian_->load(input);

            input.close();

            // Give a status update.
            std::cout << "Loaded RBM from '" << name << ".rbm'!" << std::endl;

            rc = true;
        }
    }

    MPI_Bcast(&rc, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    if (rc) {
        MPI_Bcast(&n_updates_, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(weights_.data(), weights_.size(), MPI_DOUBLE_COMPLEX, 0,
                  MPI_COMM_WORLD);
        MPI_Bcast(h_bias_.data(), h_bias_.size(), MPI_DOUBLE_COMPLEX, 0,
                  MPI_COMM_WORLD);
        MPI_Bcast(v_bias_.data(), v_bias_.size(), MPI_DOUBLE_COMPLEX, 0,
                  MPI_COMM_WORLD);
        for (auto& c : correlators_) c->bcast(0);

        if (pfaffian_) pfaffian_->bcast(0);
    }
    return rc;
}

void rbm_base::add_correlator(const std::vector<std::vector<size_t>>& corr) {
    correlators_.push_back(std::make_unique<correlator>(corr, n_alpha_));
}

std::complex<double> rbm_base::psi_notheta(
    const Eigen::MatrixXcd& state) const {
    return std::exp((v_bias_.array() * state.array()).sum());
}

std::complex<double> rbm_base::psi_default(const Eigen::MatrixXcd& state,
                                           rbm_context& context) {
    // Calculate the \psi with `thetas`
    size_t num = tools::state_to_num(state);
    std::complex<double> cosh_part = cosh(context, num);
    for (auto& c : correlators_) c->psi(state, cosh_part);
    return psi_notheta(state) * cosh_part;
}

std::complex<double> rbm_base::log_psi_over_psi_bias(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips) const {
    std::complex<double> ret = 0;
    // Claculate the visible bias part, calcels out for all not flipped sites.
    for (auto& f : flips) ret -= 2. * state(f) * v_bias_(f % n_vb_);

    std::vector<std::vector<size_t>> cidxs;
    for (auto& c : correlators_) {
        c->get_cidxs_from_flips(flips, cidxs);
        ret += c->log_psi_over_psi(state, *(cidxs.end() - 1));
    }
    return ret;
}

std::complex<double> rbm_base::log_psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    rbm_context& context, rbm_context& updated_context, bool* didupdate) {
    if (flips.empty()) return 0.;

    std::complex<double> ret = log_psi_over_psi_bias(state, flips);

    size_t num = tools::state_to_num(state);
    size_t num2 = num;
    for (auto& f : flips) num2 ^= (1 << f);
    if (true || lut_.find(num2) != lut_.end()) {
        // Update the thetas with the flips
        update_context(state, flips, updated_context);
    } else {
        if (didupdate) *didupdate = false;
    }

    // Caclulate the diffrenece of the lncoshs, which is the same as the log
    // of the ratio of coshes.
    ret += lncosh(updated_context, num2) - lncosh(context, num);

    return ret;
}

std::complex<double> rbm_base::psi_over_psi_alt(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
    rbm_context& context, rbm_context& updated_context, bool* didupdate) {
    if (flips.empty()) return 1.;

    std::complex<double> ret = std::exp(log_psi_over_psi_bias(state, flips));

    size_t num = tools::state_to_num(state);
    size_t num2 = num;
    for (auto& f : flips) num2 ^= (1 << f);
    if (true || lut_.find(num2) != lut_.end()) {
        // Update the thetas with the flips
        update_context(state, flips, updated_context);
    } else {
        if (didupdate) *didupdate = false;
    }

    ret *= cosh(updated_context, num2) / cosh(context, num);

    return ret;
}

#define COSH_LUT(func)                                   \
    g_tot++;                                             \
    time_keeper::start("hashmap");                       \
    auto lx = lut_.find(statenum);                       \
    time_keeper::end("hashmap");                         \
    if (lx != lut_.end()) {                              \
        g_lut++;                                         \
        return lx->second;                               \
    } else {                                             \
        time_keeper::start("evaluation");                \
        std::complex<double> ret = func(context.thetas); \
        time_keeper::end("evaluation");                  \
        /*lut_update_nums_.push_back(statenum);*/        \
        /*lut_update_vals_.push_back(ret);       */      \
        lut_[statenum] = ret;                            \
        return ret;                                      \
    }

std::complex<double> rbm_base::cosh(rbm_context& context, size_t statenum) {
    if (cosh_mode_ == 2) return std::exp(lncosh(context, statenum));
    COSH_LUT(cosh_);
}
std::complex<double> rbm_base::lncosh(rbm_context& context, size_t statenum) {
    if (cosh_mode_ != 2) return std::log(cosh(context, statenum));
    COSH_LUT(lncosh_);
}

#undef COSH_LUT

void rbm_base::exchange_luts() {
    std::vector<int> update_sizes(mpi::n_proc);
    std::vector<int> starts(mpi::n_proc);
    int m_size = lut_update_nums_.size();
    MPI_Allgather(&m_size, 1, MPI_INT, &update_sizes[0], 1, MPI_INT,
                  MPI_COMM_WORLD);
    int total = 0;
    for (int i = 0; i < mpi::n_proc; i++) {
        starts[i] = total;
        total += update_sizes[i];
    }
    std::vector<size_t> all_update_nums(total);
    std::vector<std::complex<double>> all_update_vals(total);
    MPI_Allgatherv(&lut_update_nums_[0], m_size, MPI_UNSIGNED_LONG,
                   &all_update_nums[0], &update_sizes[0], &starts[0],
                   MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    MPI_Allgatherv(&lut_update_vals_[0], m_size, MPI_DOUBLE_COMPLEX,
                   &all_update_vals[0], &update_sizes[0], &starts[0],
                   MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    lut_update_nums_.clear();
    lut_update_vals_.clear();

    for (int i = 0; i < mpi::n_proc; i++) {
        if (i != mpi::rank) {
            int stop = (i == mpi::n_proc - 1) ? total : starts[i + 1];
            for (int j = starts[i]; j < stop; j++) {
                lut_[all_update_nums[j]] = all_update_vals[j];
            }
        }
    }
}
