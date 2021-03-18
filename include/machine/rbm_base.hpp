/**
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
#include <complex>
#include <random>
//
#include <lattice/bravais.hpp>

namespace machine {

class rbm_base {
   protected:
    rbm_base(size_t, size_t, lattice::bravais&);

   public:
    const size_t n_alpha;
    const size_t n_visible;
    const size_t n_params;

    void interrupt(int);

    rbm_base(size_t, lattice::bravais&);
    virtual ~rbm_base() = default;

    Eigen::MatrixXcd& get_weights() { return weights_; }
    Eigen::MatrixXcd& get_h_bias() { return h_bias_; }
    Eigen::MatrixXcd& get_v_bias() { return v_bias_; }

    void initialize_weights(std::mt19937&, double, double = -1.);

    void update_weights(const Eigen::MatrixXcd&);

    virtual std::complex<double> psi(const Eigen::MatrixXcd& state,
                                     const Eigen::MatrixXcd&) const;

    virtual Eigen::MatrixXcd get_thetas(const Eigen::MatrixXcd& state) const;

    virtual void update_thetas(const Eigen::MatrixXcd& state,
                               const std::vector<size_t>& flips,
                               Eigen::MatrixXcd& thetas) const;

    virtual std::complex<double> log_psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const Eigen::MatrixXcd& thetas, Eigen::MatrixXcd& updated_thetas) const;

    virtual Eigen::MatrixXcd derivative(const Eigen::MatrixXcd& state,
                                        const Eigen::MatrixXcd& thetas) const;

    std::complex<double> log_psi_over_psi(const Eigen::MatrixXcd& state,
                                          const std::vector<size_t>& flips,
                                          const Eigen::MatrixXcd& thetas) const;

    std::complex<double> log_psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips) const;

    std::complex<double> psi_over_psi(const Eigen::MatrixXcd& state,
                                      const std::vector<size_t>& flips,
                                      const Eigen::MatrixXcd& thetas) const;

    std::complex<double> psi_over_psi(const Eigen::MatrixXcd& state,
                                      const std::vector<size_t>& flips) const;

    bool flips_accepted(double prob, const Eigen::MatrixXcd& state,
                        const std::vector<size_t>& flips,
                        Eigen::MatrixXcd& thetas) const;

    bool flips_accepted(double prob, const Eigen::MatrixXcd& state,
                        const std::vector<size_t>& flips) const;

    bool save(const std::string&);

    bool load(const std::string&);

    size_t get_n_updates() { return n_updates_; }

   protected:
    lattice::bravais& lattice_;

    Eigen::MatrixXcd weights_;
    Eigen::MatrixXcd h_bias_;
    Eigen::MatrixXcd v_bias_;

    const size_t n_vb_;

    size_t n_updates_;

    // std::vector<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>>
    //     symmetry_;
};

}  // namespace machine
