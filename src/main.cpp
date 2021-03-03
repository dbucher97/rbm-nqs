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

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <random>
#include <unordered_map>
//

#include <lattice/honeycomb.hpp>
#include <machine/rbm.hpp>
#include <machine/sampler.hpp>
#include <model/kitaev.hpp>
#include <operators/aggregator.hpp>
#include <operators/bond_op.hpp>
#include <operators/derivative_op.hpp>
#include <operators/local_op.hpp>
#include <operators/local_op_chain.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>

using namespace Eigen;

// std::ostream& operator<<(std::ostream& os, const machine::sample_t& sa) {
//     for (size_t i = 0; i < sa.vis.size(); i++) {
//         os << (sa.vis[i] ? "↑" : "↓");
//     }
//     os << " " << sa.cv;
//     return os;
// }

int main() {
    std::mt19937 rng{345214534L};
    model::kitaev km{8, {-1, -1, -1}};
    operators::base_op& H = km.get_hamiltonian();
    operators::aggregator Hagg{H, true};
    // operators::prod_aggregator Hagg2{H, H};

    machine::rbm rbm{2, km.get_lattice()};
    rbm.initialize_weights(rng, 0.01);

    machine::sampler sampler{rbm, rng};
    optimizer::stochastic_reconfiguration sr{
        rbm, sampler, km.get_hamiltonian(), 0.01, 1, 1e-4};
    sr.register_observables();
    size_t n_samples = 100;
    // sampler.sample(10, 1, 0);
    for (int i = 0; i < 1000; i++) {
        sampler.sample(n_samples, 5, 100);
        sr.optimize(n_samples);
        // n_samples++;
    }
    // machine::sampler sampler2{rbm, rng};
    // sampler2.register_op(&H);
    // sampler2.register_agg(&Hagg);
    // sampler2.sample(10000, 10, 100);
    // std::cout << Hagg.get_result(10000) << std::endl;

    return 0;
}
