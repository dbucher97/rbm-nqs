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

// #include <gmp.h>

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
//
#include <sampler/full_sampler.hpp>
#include <tools/eigen_fstream.hpp>
#include <tools/ini.hpp>
#include <tools/mpi.hpp>
#include <tools/state.hpp>

using namespace sampler;

full_sampler::full_sampler(machine::abstract_machine& rbm_, size_t bp,
                           int pfaff_refresh, int lut_exchange)
    : Base{rbm_, (size_t)(1 << rbm_.n_visible), pfaff_refresh, lut_exchange},
      bits_parallel_{bp} {}

void full_sampler::sample(bool keep_state) {
    // Initialize aggregators
    for (auto agg : aggs_) {
        agg->set_zero();
    }
    // Number of parallel gray code runs
    size_t b_len = (size_t)std::pow(2, bits_parallel_);

    // Number of total pit flips
    size_t max = (size_t)std::pow(2, rbm_.n_visible - bits_parallel_);

    double p_tot = 0;

    // The state vector if state should be kept
    Eigen::MatrixXcd vec;
    Eigen::MatrixXcd local_vec;
    Eigen::MatrixXi local_vec_idx;
    if (keep_state) {
        if (mpi::master) vec = Eigen::MatrixXcd((int)(1 << rbm_.n_visible), 1);
        local_vec = Eigen::MatrixXcd(max, 1);
        local_vec_idx = Eigen::MatrixXi(max, 1);
    }

    // Start the parallel runs
    for (size_t b = mpi::rank; b < b_len; b += mpi::n_proc) {
        size_t x = 0;
        size_t x_last = 0;
        size_t flip;

        // Get the state for `b`
        Eigen::MatrixXcd state(rbm_.n_visible, 1);
        tools::num_to_state(b, state);

        // Precalculate context
        auto context = rbm_.get_context(state);

        // Equalize pfaffian exponents, since total scaling is irrelevant, but
        // relative scaling between thread contexts make a diffrerence.
        // if (rbm_.has_pfaffian()) {
        //     auto& pfaff_context = context.pfaff();
        //     if (mpi::master) {
        //         pfaff_exp = pfaff_context.exp;
        //     }
        //     MPI_Bcast(&pfaff_exp, 1, MPI_INT, 0, MPI_COMM_WORLD);

        //     pfaff_context.exp -= pfaff_exp;
        // }

        // Do the spin flips according to gray codes and evalueate
        // observables
        std::complex<double> psi;
        for (size_t i = 1; i <= max; i++) {
            // Get the \psi of the current state and calculate probability

            // context = rbm_.get_context(state);
            psi = rbm_.psi(state, context);
            // if (std::isnan(std::real(psi)) || std::isnan(std::imag(psi))) {
            //     psi = rbm_.psi(state, context);
            // }
            double p = std::pow(std::abs(psi), 2);
            if (!std::isnan(p)) {
                // std::cout << psi << ", " << tools::state_to_num(state)
                //           << std::endl;
                // std::vector<size_t> ones;
                // for (size_t i = 0; i < rbm_.n_visible; i++) {
                //     if (std::real(state(i)) > 0) ones.push_back(i);
                // }
                // rbm_.get_lattice().print_lattice(ones);
            }

            // If keep state store \psi into the state vector
            if (keep_state) {
                local_vec(i - 1) = psi;
                local_vec_idx(i - 1) = tools::state_to_num(state);
            }

            // Cumulate probability for normalization
            p_tot += p;

            evaluate_and_aggregate(state, context, p);

            // Do the gray code update
            if (i != max) {
                // Gray code update
                x = i ^ (i >> 1);

                // Calculate the bit which needs to be flipped
                flip = std::log2l(x ^ x_last) + bits_parallel_;
                x_last = x;

                // Update \thetas and state

                rbm_.update_context(state, {flip}, context);
                pfaffian_refresh(state, context.pfaff(), i, {flip});
                state(flip) *= -1;

                exchange_luts(i);
            }
            // std::cout <<
            // "==================================================="
            //           << std::endl;
        }
        if (keep_state) {
            if (mpi::master) {
                for (size_t i = 0; i < max; i++) {
                    vec(local_vec_idx(i)) = local_vec(i);
                }

                int pi = 1;
                for (size_t p = b + 1; p < b + mpi::n_proc && p < b_len; p++) {
                    MPI_Recv(local_vec_idx.data(), local_vec_idx.size(),
                             MPI_INT, pi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(local_vec.data(), local_vec.size(),
                             MPI_DOUBLE_COMPLEX, pi, 1, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    for (size_t i = 0; i < max; i++) {
                        vec(local_vec_idx(i)) = local_vec(i);
                    }
                    pi++;
                }
            } else {
                MPI_Send(local_vec_idx.data(), local_vec_idx.size(), MPI_INT, 0,
                         0, MPI_COMM_WORLD);
                MPI_Send(local_vec.data(), local_vec.size(), MPI_DOUBLE_COMPLEX,
                         0, 1, MPI_COMM_WORLD);
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &p_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Print the state vector if `keep_state`
    if (keep_state && mpi::master) {
        vec /= std::sqrt(p_tot);
        std::ofstream statefile{ini::name + ".state", std::ios::binary};
        statefile << vec;
        statefile.close();
        std::cout << "State stored to '" << ini::name + ".state"
                  << "'" << std::endl;
    }
    for (auto agg : aggs_) {
        agg->finalize(p_tot);
    }
}

size_t full_sampler::get_my_n_samples() const {
    size_t b_len = (size_t)std::pow(2, bits_parallel_);
    size_t max = (size_t)std::pow(2, rbm_.n_visible - bits_parallel_);
    size_t ret = 0;
    for (size_t b = mpi::rank; b < b_len; b += mpi::n_proc) {
        ret += max;
    }
    return ret;
}
