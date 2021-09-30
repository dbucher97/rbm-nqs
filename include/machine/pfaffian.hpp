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

#include <Eigen/Sparse>
#include <complex>
#include <fstream>
#include <random>
#include <vector>
//
#include <lattice/bravais.hpp>
#include <machine/context.hpp>

namespace machine {

class pfaffian {
    const lattice::bravais& lattice_;
    const size_t ns_;
    const size_t n_symm_;
    const std::vector<Eigen::PermutationMatrix<Eigen::Dynamic>> symmetry_;
    const std::vector<size_t> symm_basis_;

    Eigen::MatrixXcd fs_;

    Eigen::MatrixXi bs_;
    Eigen::MatrixXi ss_;

    const bool no_updating_;

    static size_t calc_n_params(size_t ns,
                                const std::vector<size_t>& symm_basis);

   public:
    pfaffian(const lattice::bravais&, const std::vector<double>& symmetry = {},
             bool no_updating = false);

    void init_weights(std::mt19937& rng, double std, bool normalize = false);
    void init_weights_hf(
        const std::vector<Eigen::SparseMatrix<std::complex<double>>>& mats,
        const std::vector<std::vector<size_t>>& acts_on);

    pfaff_context get_context(const Eigen::MatrixXcd& state) const;

    void update_context(const Eigen::MatrixXcd& state,
                        const std::vector<size_t>& flips,
                        pfaff_context& context) const;

    void derivative(const Eigen::MatrixXcd& state, const pfaff_context& context,
                    Eigen::MatrixXcd& result, size_t& offset) const;

    inline std::complex<double> psi(const Eigen::MatrixXcd& state,
                                    const pfaff_context& context) const {
        // auto context2 = get_context(state);
        return context.pfaff * std::pow(10, context.exp);
    }

    inline std::complex<double> psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const pfaff_context& context, pfaff_context& updated_context) const {
        // Eigen::MatrixXcd state2 = state;
        // for (auto& f : flips) state2(f) *= -1;
        // std::cout << std::abs(psi(state2, updated_context) /
        //                           psi(state, context) -
        //                       updated_context.update_factor)
        //           << ", ";
        // for (auto& f : flips) std::cout << f << ", ";
        // std::cout << std::endl;
        return updated_context.update_factor;
        // return updated_context.update_factor;
    }

    void update_weights(const Eigen::MatrixXcd& dw, size_t& offset);

    Eigen::MatrixXcd& get_weights() { return fs_; }

    inline size_t get_n_params() const { return fs_.size(); }

    void save(std::ofstream& output);
    void load(std::ifstream& input);

    bool load_from_pfaffian_psi(const std::string& filename);
    /* Eigen::MatrixXi& get_bs() { return bs_; }
    Eigen::MatrixXi& get_ss() { return ss_; } */
    void bcast(int rank);

   private:
    bool spidx(size_t i, const Eigen::MatrixXcd& state, bool flip) const;
    int spidx(size_t i, size_t j, const Eigen::MatrixXcd& state,
              bool flipi = false, bool flipj = false) const;

    size_t idx(size_t i, size_t j) const;

    inline std::complex<double> a(size_t i, size_t j,
                                  const Eigen::MatrixXcd& state,
                                  bool flipi = false, bool flipj = false) const;

    Eigen::MatrixXcd get_mat(const Eigen::MatrixXcd& state) const;
};

}  // namespace machine
