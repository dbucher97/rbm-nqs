/**
 * src/main.cpp
 * Copyright (c) 2021 David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARjANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

// #include <omp.h>

#include <omp.h>

#include <Eigen/Dense>
#include <chrono>
#include <complex>
#include <iostream>
#include <random>
//
#include <lattice/honeycomb.hpp>
#include <machine/full_sampler.hpp>
#include <machine/metropolis_sampler.hpp>
#include <machine/rbm_symmetry.hpp>
#include <model/kitaev.hpp>
#include <operators/aggregator.hpp>
#include <operators/bond_op.hpp>
#include <operators/derivative_op.hpp>
#include <operators/local_op.hpp>
#include <operators/local_op_chain.hpp>
#include <optimizer/plugin.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>

using namespace Eigen;

// std::ostream& operator<<(std::ostream& os, const machine::sample_t& sa) {
//     for (size_t i = 0; i < sa.vis.size(); i++) {
//         os << (sa.vis[i] ? "↑" : "↓");
//     }
//     os << " " << sa.cv;
//     return os;
// }
//
//
//
static int x = 5;
int f(int j) { return x * j; }

int main() {
    omp_set_num_threads(8);
    Eigen::setNbThreads(1);

    std::mt19937 rng{34521434L};
    model::kitaev km{2, {-1, -1, -1}};
    operators::base_op& H = km.get_hamiltonian();
    operators::aggregator Hagg{H, true};
    // operators::prod_aggregator Hagg2{H, H};

    machine::rbm_symmetry rbm{3, km.get_lattice()};
    rbm.initialize_weights(rng, 0.001);

    // operators::derivative_op D{rbm.n_alpha, rbm.get_symmetry().size()};
    // operators::aggregator Dagg{D};

    // machine::full_sampler sampler{rbm, 3};
    machine::metropolis_sampler sampler{rbm, rng, 8};

    //
    optimizer::stochastic_reconfiguration sr{
        rbm, sampler, km.get_hamiltonian(), 0.05, 0.03, 0.99, 0.1, 1e-4, 0.9};
    sr.register_observables();
    // optimizer::adam_plugin p{sr.get_n_total()};
    // sr.set_plugin(&p);

    size_t n_samples = 400;
    size_t n_epochs = 400;
    double t_sample = 0, t_optim = 0;

    for (size_t i = 0; i < n_epochs; i++) {
        auto a = std::chrono::system_clock::now();
        sampler.sample(n_samples);
        auto b = std::chrono::system_clock::now();
        Eigen::setNbThreads(0);
        sr.optimize();
        Eigen::setNbThreads(1);
        auto c = std::chrono::system_clock::now();
        t_sample += std::chrono::duration_cast<std::chrono::milliseconds>(b - a)
                        .count();
        t_optim += std::chrono::duration_cast<std::chrono::milliseconds>(c - b)
                       .count();
        // n_samples++;
    }
    std::cout << t_sample / n_epochs << ", " << t_optim / n_epochs << std::endl;
    machine::full_sampler sampler2{rbm, 3};
    sampler2.register_op(&H);
    sampler2.register_agg(&Hagg);
    sampler2.sample();
    std::cout << Hagg.get_result() / rbm.n_visible << std::endl;
    //
    return 0;
}
