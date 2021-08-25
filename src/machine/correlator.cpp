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

#include <iostream>
//
#include <machine/correlator.hpp>
#include <tools/eigen_fstream.hpp>
#include <tools/mpi.hpp>

using namespace machine;

correlator::correlator(const std::vector<std::vector<size_t>>& corr,
                       size_t n_hidden,
                       const std::vector<std::vector<size_t>>& uc_symm)
    : cmax{corr.size()},
      corr_(corr.size()),
      symm_(uc_symm.size() == 0 ? 1 : uc_symm.size()),
      rev_symm_(symm_.size()),
      bias_(cmax / symm_.size(), 1),
      weights_(cmax, n_hidden),
      n_hidden_{n_hidden} {
    size_t c = 0;
    for (auto& ci : corr_) {
        for (auto idx : ci) {
            corr_[c].push_back(idx);
            rev_map_[idx] = c;
        }
        c++;
    }
    if (uc_symm.size() == 0) {
        rev_symm_.push_back({});
        for (size_t i = 0; i < cmax; i++) {
            symm_[0].push_back(i);
            rev_symm_[0].push_back(i);
        }
    } else {
        c = 0;
        for (auto& si : uc_symm) {
            rev_symm_[c].reserve(si.size());
            size_t ic = 0;
            for (auto idx : si) {
                symm_[c].push_back(idx);
                rev_symm_[c][idx] = ic;
                ic++;
            }
            c++;
        }
    }
}

std::complex<double> correlator::evaluate(const Eigen::MatrixXcd& state,
                                          size_t cidx, size_t sidx) const {
    std::complex<double> ret = 1;
    for (auto i : corr_[symm_[sidx][cidx]]) {
        ret *= state(i);
    }
    return ret;
}

void correlator::initialize_weights(std::mt19937& rng, double std_dev,
                                    double std_dev_imag) {
    // If std_dev_imag < 0, use the normal std_dev;
    if (mpi::master) {
        if (std_dev_imag < 0) std_dev_imag = std_dev;

        // Initialize the normal distribution
        std::normal_distribution<double> real_dist{0, std_dev};
        std::normal_distribution<double> imag_dist{0, std_dev_imag};

        // Fill all weigthts and biases
        for (size_t i = 0; i < static_cast<size_t>(bias_.size()); i++) {
            bias_(i) = std::complex<double>(real_dist(rng), imag_dist(rng));
        }
        for (size_t i = 0; i < cmax; i++) {
            for (size_t j = 0; j < n_hidden_; j++) {
                weights_(i, j) =
                    std::complex<double>(real_dist(rng), imag_dist(rng));
            }
        }
    }
    MPI_Bcast(weights_.data(), weights_.size(), MPI_DOUBLE_COMPLEX, 0,
              MPI_COMM_WORLD);
}

size_t correlator::get_n_params() const {
    return bias_.size() + weights_.size();
}

void correlator::update_weights(const Eigen::MatrixXcd& dw, size_t& offset) {
    bias_ -= dw.block(offset, 0, bias_.size(), 1);
    offset += bias_.size();

    Eigen::MatrixXcd dww = dw.block(offset, 0, weights_.size(), 1);
    weights_ -=
        Eigen::Map<Eigen::MatrixXcd>(dww.data(), n_hidden_, cmax).transpose();
    offset += weights_.size();
}

void correlator::psi(const Eigen::MatrixXcd& state,
                     std::complex<double>& res) const {
    for (size_t i = 0; i < cmax; i++) {
        res *= std::exp(bias_(i % bias_.size()) * evaluate(state, i));
    }
}

void correlator::add_thetas(const Eigen::MatrixXcd& state,
                            Eigen::MatrixXcd& thetas, size_t sidx) const {
    for (size_t i = 0; i < cmax; i++) {
        thetas.col(sidx) +=
            weights_.row(i).transpose() * evaluate(state, i, sidx);
    }
}

void correlator::get_cidxs_from_flips(
    const std::vector<size_t>& flips,
    std::vector<std::vector<size_t>>& cidxsa) const {
    std::vector<size_t> cidxs;
    for (auto& f : flips) {
        try {
            size_t cidx = get_cidx(f);
            auto cidx_idx = std::find(cidxs.begin(), cidxs.end(), cidx);
            if (cidx_idx == cidxs.end()) {
                cidxs.push_back(cidx);
            } else {
                cidxs.erase(cidx_idx);
            }
        } catch (const std::out_of_range&) {
        }
    }
    cidxsa.push_back(cidxs);
}

void correlator::update_thetas(const Eigen::MatrixXcd& state,
                               const std::vector<size_t>& cidxs,
                               Eigen::MatrixXcd& thetas, size_t sidx) const {
    for (auto& i : cidxs) {
        thetas.col(sidx) -= 2. *
                            weights_.row(cidx_with_symm(i, sidx)).transpose() *
                            evaluate(state, i);
    }
}

std::complex<double> correlator::log_psi_over_psi(
    const Eigen::MatrixXcd& state, const std::vector<size_t>& cidxs) const {
    std::complex<double> ret = 0;
    for (auto& i : cidxs) {
        ret -= 2. * bias_(i % bias_.size()) * evaluate(state, i);
    }
    return ret;
}

void correlator::derivative(const Eigen::MatrixXcd& state,
                            const Eigen::MatrixXcd& tanh,
                            Eigen::MatrixXcd& result, size_t& offset) const {
    result.block(offset, 0, bias_.size(), 1).setZero();
    for (size_t i = 0; i < cmax; i++) {
        result(offset + i % bias_.size()) += evaluate(state, i);
    }
    offset += bias_.size();
    for (size_t i = 0; i < cmax; i++) {
        result.block(offset + i * tanh.rows(), 0, tanh.rows(), 1).setZero();
        for (size_t s = 0; s < symm_.size(); s++) {
            result.block(offset + i * tanh.rows(), 0, tanh.rows(), 1) +=
                evaluate(state, i, s) * tanh.col(s);
        }
    }
    offset += cmax * tanh.rows();
}

void correlator::load(std::ifstream& input) { input >> weights_ >> bias_; }

void correlator::save(std::ofstream& output) { output << weights_ << bias_; }

void correlator::bcast(int rank) {
    MPI_Bcast(weights_.data(), weights_.size(), MPI_DOUBLE_COMPLEX, rank,
              MPI_COMM_WORLD);
    MPI_Bcast(bias_.data(), bias_.size(), MPI_DOUBLE_COMPLEX, rank,
              MPI_COMM_WORLD);
}
