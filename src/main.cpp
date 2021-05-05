/*
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

#include <fcntl.h>
#include <omp.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <chrono>
#include <complex>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unsupported/Eigen/KroneckerProduct>
//
#include <lattice/honeycomb.hpp>
#include <lattice/honeycombS3.hpp>
#include <lattice/toric_lattice.hpp>
#include <machine/correlator.hpp>
#include <machine/file_psi.hpp>
#include <machine/full_sampler.hpp>
#include <machine/metropolis_sampler.hpp>
#include <machine/rbm_base.hpp>
#include <machine/rbm_symmetry.hpp>
#include <model/isingS3.hpp>
#include <model/kitaev.hpp>
#include <model/kitaevS3.hpp>
#include <model/toric.hpp>
#include <operators/aggregator.hpp>
#include <operators/bond_op.hpp>
#include <operators/derivative_op.hpp>
#include <operators/local_op.hpp>
#include <operators/local_op_chain.hpp>
#include <operators/overlap_op.hpp>
#include <operators/store_state.hpp>
#include <optimizer/abstract_optimizer.hpp>
#include <optimizer/gradient_descent.hpp>
#include <optimizer/plugin.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>
#include <tools/ini.hpp>
#include <tools/logger.hpp>
#include <tools/state.hpp>

using namespace Eigen;

volatile static bool g_interrupt = false;
volatile static bool g_saved = false;

void interrupt(int sig) {
    g_interrupt = sig == SIGINT;
    while (!g_saved) {
        usleep(100);
    }
    exit(0);
}

void progress_bar(size_t i, size_t n_epochs, double energy) {
    double progress = i / (double)n_epochs;
    int digs = (int)std::log10(n_epochs) - (int)std::log10(i);
    if (i == 0) digs = (int)std::log10(n_epochs);
    std::cout << "\rEpochs: (" << std::string(digs, ' ') << i << "/" << n_epochs
              << ") ";
    struct winsize size;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
    int plen = size.ws_col - (2 * (int)std::log10(n_epochs) + 35);
    int p = (int)(plen * progress + 0.5);
    int m = plen - p;
    std::cout << "[" << std::string(p, '#') << std::string(m, ' ') << "]";
    std::cout << std::showpoint;
    std::cout << " Energy: " << energy;
    std::cout << std::flush;
}

std::vector<size_t> to_indices(const MatrixXcd& vec) {
    std::vector<size_t> ret;
    for (size_t v = 0; v < vec.size(); v++) {
        if (std::real(vec(v)) > 0) ret.push_back(v);
    }
    return ret;
}
size_t to_idx(const MatrixXcd& vec) {
    for (size_t v = 0; v < vec.size(); v++) {
        if (std::real(vec(v)) > 0) return v;
    }
    return -1;
}
void test_symmetry() {
    model::kitaev km{4, -1};
    auto& lattice = km.get_lattice();
    MatrixXcd vec(lattice.n_total, 1);
    vec.setConstant(-1);
    vec(0) = 1;
    vec(1) = 1;
    vec(2) = 1;
    vec(8) = 1;

    auto symm = lattice.construct_symmetry();
    for (auto& s : symm) {
        lattice.print_lattice(to_indices(s * vec));
    }
}

void print_bonds() {
    lattice::honeycombS3 lat{1};
    auto bonds = lat.get_bonds();
    for (auto& bond : bonds) {
        std::cout << bond.a << "," << bond.b << "," << bond.type << std::endl;
    }
    lat.print_lattice({0, lat.up(0, 1)});
}

void test_S3() {
    model::isingS3 km{3, -1};
    machine::file_psi m{km.get_lattice(), "isingS3.state"};
    machine::full_sampler sampler{m, 3};
    auto& h = km.get_hamiltonian();
    operators::aggregator agg{h};
    sampler.register_op(&h);
    sampler.register_agg(&agg);
    sampler.sample(false);
    std::cout.precision(17);
    std::cout << agg.get_result() << std::endl;
    std::cout << agg.get_result() / km.get_lattice().n_total << std::endl;
}

void debug() {
    // auto& h = m.get_hamiltonian();
    //
    // machine::rbm_base rbm(3, m.get_lattice());
    //
    // Eigen::MatrixXcd state(rbm.n_visible, 1);
    // std::uniform_real_distribution<double> u_dist(0, 1);
    // std::mt19937 rng;
    // for (size_t i = 0; i < rbm.n_visible; i++) {
    //     state(i) = u_dist(rng) < 0.5 ? 1. : -1.;
    // }
    // MatrixXcd thetas = rbm.get_thetas(state);
    //
    // h.evaluate(rbm, state, thetas);
    lattice::toric_lattice lat{3};
    auto plaq = lat.construct_plaqs();
    for (auto& p : plaq) {
        for (auto& i : p.idxs) {
            std::cout << i << ", ";
        }
        std::cout << p.type << std::endl;
    }
    // auto symm = lat.construct_symmetry();
    // std::vector<size_t> v1(plaq[0].idxs.begin(), plaq[0].idxs.end());
    // std::vector<size_t> v2(plaq[1].idxs.begin(), plaq[1].idxs.end());
    // v1.insert(v1.end(), v2.begin(), v2.end());
    // MatrixXcd vec(lat.n_total, 1);
    // for (auto& s : symm) {
    //     std::vector<size_t> st;
    //     for (auto v : v1) {
    //         vec.setZero();
    //         vec(v) = 1;
    //         vec = s * vec;
    //         // std::cout << to_idx(vec) << std::endl;
    //         st.push_back(to_idx(vec));
    //     }
    //     for (auto& x : st) std::cout << x << ", ";
    //     std::cout << std::endl;
    //     lat.print_lattice(st);
    //     std::cout << std::endl;
    // }
}

int main(int argc, char* argv[]) {
    // debug();
    // return 0;

    int rc = ini::load(argc, argv);
    if (rc != 0) {
        return rc;
    }

    logger::init();

    omp_set_num_threads(ini::n_threads);
    Eigen::setNbThreads(1);

    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};

    std::unique_ptr<model::abstract_model> model;
    switch (ini::model) {
        case ini::model_t::KITAEV:
            model = std::make_unique<model::kitaev>(ini::n_cells, ini::J);
            break;
        case ini::model_t::KITAEV_S3:
            model = std::make_unique<model::kitaevS3>(ini::n_cells, ini::J);
            break;
        case ini::model_t::ISING_S3:
            model = std::make_unique<model::isingS3>(ini::n_cells, ini::J);
            break;
        case ini::model_t::TORIC:
            model = std::make_unique<model::toric>(ini::n_cells, ini::J);
            break;
        default:
            return 1;
    }

    std::unique_ptr<machine::rbm_base> rbm;
    switch (ini::rbm) {
        case ini::rbm_t::BASIC:
            rbm = std::make_unique<machine::rbm_base>(ini::n_hidden,
                                                      model->get_lattice());
            break;
        case ini::rbm_t::SYMMETRY:
            rbm = std::make_unique<machine::rbm_symmetry>(ini::n_hidden,
                                                          model->get_lattice());
            break;
        default:
            return 1;
    }
    if (ini::rbm_correlators && model->get_lattice().has_correlators()) {
        auto c = model->get_lattice().get_correlators();
        rbm->add_correlators(c);
    }
    if (ini::rbm_force || !rbm->load(ini::name)) {
        rbm->initialize_weights(rng, ini::rbm_weights, ini::rbm_weights_imag);
    }

    std::unique_ptr<machine::abstract_sampler> sampler;
    switch (ini::sa_type) {
        case ini::sampler_t::FULL:
            sampler = std::make_unique<machine::full_sampler>(
                *rbm, ini::sa_full_n_parallel_bits);
            break;
        case ini::sampler_t::METROPOLIS:
            sampler = std::make_unique<machine::metropolis_sampler>(
                *rbm, rng, ini::sa_metropolis_n_chains,
                ini::sa_metropolis_n_steps_per_sample,
                ini::sa_metropolis_n_warmup_steps);
            break;
        default:
            return 1;
    }

    std::unique_ptr<optimizer::abstract_optimizer> optimizer;
    switch (ini::opt_type) {
        case ini::optimizer_t::SR:
            optimizer = std::make_unique<optimizer::stochastic_reconfiguration>(
                *rbm, *sampler, model->get_hamiltonian(), ini::opt_lr,
                ini::opt_sr_reg);
            break;
        case ini::optimizer_t::SGD:
            optimizer = std::make_unique<optimizer::gradient_descent>(
                *rbm, *sampler, model->get_hamiltonian(), ini::opt_lr);
            break;
        default:
            return 1;
    }
    optimizer->register_observables();
    std::unique_ptr<optimizer::base_plugin> p;
    if (ini ::opt_plugin.length() > 0) {
        if (ini::opt_plugin == "adam") {
            p = std::make_unique<optimizer::adam_plugin>(rbm->get_n_params());
        } else if (ini::opt_plugin == "momentum") {
            p = std::make_unique<optimizer::momentum_plugin>(
                rbm->get_n_params());
        } else {
            return 1;
        }
        optimizer->set_plugin(p.get());
    }

    std::cout << "Number of parameters: " << rbm->get_n_params() << std::endl;

    if (ini::train) {
        // Start getchar non-block
        struct termios oldt, newt;
        int oldf;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
        // End getchar non-block

        progress_bar(0, ini::n_epochs, -1);
        int ch = 0;
        for (size_t i = 0; i < ini::n_epochs && ch != ''; i++) {
            sampler->sample(ini::sa_n_samples);
            Eigen::setNbThreads(ini::n_threads);
            optimizer->optimize();
            Eigen::setNbThreads(1);
            logger::newline();
            progress_bar(i + 1, ini::n_epochs,
                         optimizer->get_current_energy() / rbm->n_visible);
            ch = getchar();
        }

        // Start getchar non-block
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        fcntl(STDIN_FILENO, F_SETFL, oldf);
        // End getchar non-block

        std::cout << std::endl;
        rbm->save(ini::name);

    } else {
        std::cout << "nothing to do!" << std::endl;
    }

    machine::full_sampler samp{*rbm, 3};
    operators::aggregator agg{model->get_hamiltonian()};
    samp.register_op(&(model->get_hamiltonian()));
    samp.register_agg(&agg);
    samp.sample(true);
    std::cout.precision(17);
    std::cout << std::real(agg.get_result()(0)) / rbm->n_visible << std::endl;

    return 0;
}
