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

#pragma once

#include <Eigen/Dense>
#include <complex>
#include <fstream>
#include <machine/spin_state.hpp>
#include <random>
#include <unordered_map>
#include <vector>

namespace machine {
class correlator {
   public:
    const size_t cmax;

   private:
    std::vector<std::vector<size_t>> corr_;
    std::vector<std::vector<size_t>> symm_;
    std::vector<std::vector<size_t>> rev_symm_;
    std::unordered_map<size_t, size_t> rev_map_;

    Eigen::MatrixXcd bias_;
    Eigen::MatrixXcd weights_;

    const size_t n_hidden_;

    inline size_t cidx_with_symm(size_t cidx, size_t sidx) const {
        return rev_symm_[sidx][cidx];
    }

    inline size_t get_cidx(size_t idx) const { return rev_map_.at(idx); }

   public:
    correlator(const std::vector<std::vector<size_t>>& corr, size_t n_hidden,
               const std::vector<std::vector<size_t>>& symm = {});

    std::complex<double> evaluate(const spin_state& state, size_t cidx,
                                  size_t sidx = 0) const;

    /**
     * @brief Initializes the weights randomly with given standard deviations.
     * The imaginary part can have a different standard deviation.
     *
     * @param rng Reference to the RNG.
     * @param std_dev Standard Deviation.
     * @param std_dev_imag Standard Deviation for the imaginary part, default
     * -1. will use the same as real part.
     */
    void initialize_weights(std::mt19937& rng, double std_dev,
                            double std_dev_imag = -1.);

    size_t get_n_params() const;

    void update_weights(const Eigen::MatrixXcd& dw, size_t& offset);

    void psi(const spin_state& state, std::complex<double>& res) const;

    void add_thetas(const spin_state& state, Eigen::MatrixXcd& res,
                    size_t sidx = 0) const;

    void get_cidxs_from_flips(const std::vector<size_t>& flips,
                              std::vector<std::vector<size_t>>& cidxs) const;

    void update_thetas(const spin_state& state,
                       const std::vector<size_t>& cidxs,
                       Eigen::MatrixXcd& thetas, size_t sidx = 0) const;

    std::complex<double> log_psi_over_psi(
        const spin_state& state, const std::vector<size_t>& cidxs) const;

    void derivative(const spin_state& state, const Eigen::MatrixXcd& thetas,
                    Eigen::MatrixXcd& result, size_t& offset) const;

    void load(std::ifstream& input);
    void save(std::ofstream& output);

    void bcast(int rank);
};
}  // namespace machine
