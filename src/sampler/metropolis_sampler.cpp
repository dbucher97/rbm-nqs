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
#include <mpi.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <random>
//
#include <lattice/toric_lattice.hpp>
#include <machine/spin_state.hpp>
#include <operators/base_op.hpp>
#include <sampler/metropolis_sampler.hpp>
#include <tools/ini.hpp>
#include <tools/logger.hpp>
#include <tools/state.hpp>
#include <tools/time_keeper.hpp>

// #define RANDOM_UPDATES
// #define TORIC_UPDATES

using namespace sampler;
int g_chain = 0;

metropolis_sampler::metropolis_sampler(machine::abstract_machine& rbm,
                                       size_t n_samples, std::mt19937& rng,
                                       size_t n_chains, size_t step_size,
                                       size_t warmup_steps, double bond_flips,
                                       int refresh, int lut_exchange)
    : Base{rbm, n_samples, refresh, lut_exchange},
      rng_{rng},
      n_chains_{n_chains},
      step_size_{step_size},
      n_sweeps_{rbm.n_visible},
      warmup_steps_{warmup_steps},
      bond_flips_{bond_flips},
      f_dist_{0, rbm.n_visible - 1} {}

void metropolis_sampler::sample() {
    // Initialize aggregators
    for (auto agg : aggs_) {
        agg->set_zero();
    }
    g_chain = 0;

    // Divide the `total_samples` between the chains.
    size_t samples_per_chain = n_samples_ / n_chains_;
    size_t residue = n_samples_ - samples_per_chain * n_chains_;

    // Initialize acceptance_rate
    acceptance_rate_ = 0;
    double local_ar = 0;
    // if (mpi::master) {
    //     int n = 261135;
    //     Eigen::MatrixXcd state(rbm_.n_visible, 1);
    //     for (int i = 0; i < state.size(); i++) {
    //         if (n >> i & 1)
    //             state(i) = 1;
    //         else
    //             state(i) = -1;
    //     }
    //     auto ct = rbm_.get_context(state);
    //     // std::cout << "\nPSI " << rbm_.psi(state, ct) << std::endl;
    // }

    // Run the chains in parallel.
    for (size_t c = mpi::rank; c < n_chains_; c += mpi::n_proc) {
        double ar = sample_chain(samples_per_chain + (c == 0 ? residue : 0));
        g_chain++;
        // Accumulate acceptance_rate
        local_ar += ar;
    }
    MPI_Allreduce(&local_ar, &acceptance_rate_, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    // Average acceptance rate.
    acceptance_rate_ /= n_chains_;

    // Exchange luts
    // std::vector<size_t> sizes(mpi::n_proc);
    // size_t m_size = sampled_.size();
    // MPI_Allgather(&m_size, 1, MPI_UNSIGNED_LONG, &sizes[0], 1,
    //               MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    // Eigen::MatrixXcd state;
    // for (auto& s : sampled_) {
    //     std::cout << s << ", " << n_lut_[s] << std::endl;

    //     // tools::num_to_state(sampled_[s], state);
    //     // auto context = rbm_.get_context(state);
    //     // evaluate_and_aggregate(state, context, n_lut_[sampled_[s]]);
    // }
    // std::cout << sampled_.size() << std::endl;

    // sampled_.clear();
    // n_lut_.clear();

    // Finalize aggregators
    for (auto agg : aggs_) {
        agg->finalize(n_samples_);
    }
}

double metropolis_sampler::sample_chain(size_t total_samples) {
    size_t total_steps = total_samples * step_size_ + warmup_steps_;
    size_t ar = 0;
    Eigen::ArrayXd accs(2);
    accs.setZero();
    Eigen::ArrayXd tries(3);
    tries.setZero();

    // Initilaize random state
    machine::spin_state state(rbm_.n_visible);
    state.set_random(rng_);

    // if (ini::lattice_type == "hex") {
    //     if (u_dist_(rng_) < 0.5) {
    //         std::complex<double> r = (u_dist_(rng_) < 0.5 ? -1. : 1.);
    //         state.setConstant(-r);
    //         auto& lat = rbm_.get_lattice();
    //         size_t s = 0;
    //         state(s) = r;
    //         s = lat.nns(s)[0];
    //         while (s != 0) {
    //             state(s) = r;
    //             if (lat.b_idx(s) == 0) {
    //                 s = lat.nns(s)[0];
    //             } else {
    //                 s = lat.nns(s)[1];
    //             }
    //         }
    //     }
    // }
    // if (ups % 2 == 1) {
    //     state(0) *= -1;
    // }

#ifdef TORIC_UPDATES
    auto plaq = dynamic_cast<lattice::toric_lattice*>(&rbm_.get_lattice())
                    ->construct_plaqs();
    std::uniform_int_distribution<size_t> b_dist(0, plaq.size() / 2 - 1);
#else
    auto& bx = rbm_.get_lattice().get_bonds();
    std::vector<size_t> ba;
    std::vector<size_t> bb;
    for (auto& b : bx) {
        if (b.type != 2) {
            ba.push_back(b.a);
            bb.push_back(b.b);
        }
    }
    std::uniform_int_distribution<size_t> b_dist(0, ba.size() - 1);
#endif

    // Retrieve context for state
    auto context = rbm_.get_context(state);

    std::vector<size_t> flips(1);

    std::uniform_real_distribution<double> rdist(0, 1);

    size_t proposed = 0;

    // Do the Metropolis sampling
    for (size_t step = 0; step < total_steps; step++) {
        // Get the flips vector by randomly selecting one site.
        time_keeper::start("Metropolis sweep");

        std::vector<size_t> idxs(rbm_.n_visible);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::shuffle(idxs.begin(), idxs.end(), rng_);
        for (size_t sweep = 0; sweep < n_sweeps_; sweep++) {
            // flips.clear();
            // flips.push_back(idxs[sweep]);
            // if (u_dist_(rng_) < bond_flips_) {
            //     sweep++;
            //     if (sweep < n_sweeps_) flips.push_back(idxs[sweep]);
            // }
            flips.clear();
            // With probability 1/2 flip a second site.
            double x = u_dist_(rng_);
            int type;
            if (x < bond_flips_) {
                type = 1;
#ifdef RANDOM_UPDATES
                size_t a = f_dist_(rng_);
                size_t b = a;
                while (a == b) b = f_dist_(rng_);
                flips = {a, b};
                sweep++;
#else
#ifdef TORIC_UPDATES
                auto p = plaq[b_dist(rng_) * 2 + 1];
                flips = {p.idxs[0], p.idxs[1], p.idxs[2], p.idxs[3]};
                sweep += 3;
#else
                size_t bidx = b_dist(rng_);
                flips = {ba[bidx], bb[bidx]};
                sweep++;
#endif
#endif
            } else {
                flips.push_back(f_dist_(rng_));
                type = 0;
            }
            proposed++;
            // tries(type)++;

            machine::rbm_context new_context = context;
            // Calculate the probability of changing to new configuration
            const double T = 1;
            double acc =
                std::pow(std::abs(rbm_.psi_over_psi(state, flips, context,
                                                    new_context, false)),
                         2 / T);

            // Accept new configuration with given probability
            if (u_dist_(rng_) < acc) {
                // if (!didupdate) rbm_.update_context(state, flips,
                // new_context);
                context = new_context;
                ar++;
                accs(type)++;

                // Refresh pfaffian context if demanded
                pfaffian_refresh(state, context.pfaff(), ar, flips);

                state.flip(flips);
            }
        }

        time_keeper::end("Metropolis sweep");

        // Exchange RBM LUT if demanded
        exchange_luts(step);

        // If a sample is required
        if ((step >= warmup_steps_) &&
            ((step - warmup_steps_) % step_size_ == 0)) {
            // Evaluate oprators
            evaluate_and_aggregate(state, context);
            // Store state num
            // size_t statenum = tools::state_to_num(state);
            // if (n_lut_.find(statenum) == n_lut_.end()) {
            //     n_lut_[statenum] = 1;
            //     sampled_.push_back(statenum);
            // } else {
            //     n_lut_[statenum]++;
            // }
        }
    }
    // if (mpi::rank == 1 && rbm_.get_n_updates() == 179 && g_chain == 1) {
    //     std::cout << "Acceptance Rate " << ar / (double)total_steps
    //               << std::endl;
    // }

    // Normalize acceptance rate
    return ar / (double)proposed;
}

void metropolis_sampler::log() {
    logger::log(acceptance_rate_, "AccetpanceRate");
}

size_t metropolis_sampler::get_my_n_samples() const {
    size_t samples_per_chain = n_samples_ / n_chains_;
    size_t residue = n_samples_ - samples_per_chain * n_chains_;
    size_t ret = 0;
    for (size_t c = mpi::rank; c < n_chains_; c += mpi::n_proc) {
        ret += samples_per_chain + (c == 0 ? residue : 0);
    }
    return ret;
}

void metropolis_sampler::set_n_samples(size_t samples) {
    double f = samples / (double)n_samples_;
    warmup_steps_ *= f;
    n_samples_ = samples;
}
