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

#include <complex>
#include <random>
#include <vector>
#include <fstream>
//
#include <lattice/bravais.hpp>
#include <machine/context.hpp>

namespace machine {

class pfaffian {
    const lattice::bravais& lattice_;
    const size_t ns_;
    const size_t n_symm_;
    Eigen::MatrixXcd fs_;

    Eigen::MatrixXi bs_;
    Eigen::MatrixXi ss_;

   public:
    pfaffian(const lattice::bravais&, size_t n_uc = 0);

    void init_weights(std::mt19937& rng, double std, bool normalize = false);

    pfaff_context get_context(const Eigen::MatrixXcd& state) const;

    void update_context(const Eigen::MatrixXcd& state,
                        const std::vector<size_t>& flips,
                        pfaff_context& context) const;

    void derivative(const Eigen::MatrixXcd& state,
                                const pfaff_context& context,
                                Eigen::MatrixXcd& result, size_t& offset) const;

    inline std::complex<double> psi(const Eigen::MatrixXcd& state,
                                    const pfaff_context& context) const {
        return context.pfaff;
    }

    inline std::complex<double> psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        pfaff_context& updated_context) const {
        update_context(state, flips, updated_context);
        return updated_context.update_factor;
    }

    void update_weights(const Eigen::MatrixXcd& dw, size_t& offset);

    Eigen::MatrixXcd& get_weights() { return fs_; }

    inline size_t get_n_params() const { return fs_.size(); }

    void save(std::ofstream& output);
    void load(std::ifstream& input);
    /* Eigen::MatrixXi& get_bs() { return bs_; }
    Eigen::MatrixXi& get_ss() { return ss_; } */

   private:
    inline bool spidx(size_t i, const Eigen::MatrixXcd& state,
                      bool flip) const {
        return (std::real(state(i)) < 0) ^ flip;
    }

    inline size_t idx(size_t i, size_t j, const Eigen::MatrixXcd& state,
                      bool flipi = false, bool flipj = false) const {
        size_t ret =
            (ns_ - 1) * (2 * spidx(i, state, flipi) + spidx(j, state, flipj));
        return ret + bs_(i, j);
    }

    inline std::complex<double> a(size_t i, size_t j,
                                  const Eigen::MatrixXcd& state,
                                  bool flipi = false,
                                  bool flipj = false) const {
        return fs_(idx(i, j, state, flipi, flipj), ss_(i)) -
               fs_(idx(j, i, state, flipj, flipi), ss_(j));
    }

    Eigen::MatrixXcd get_mat(const Eigen::MatrixXcd& state) const;
};

}  // namespace machine
