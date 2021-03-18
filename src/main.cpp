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
//
#include <lattice/honeycomb.hpp>
#include <machine/exact_sampler.hpp>
#include <machine/full_sampler.hpp>
#include <machine/metropolis_sampler.hpp>
#include <machine/rbm_base.hpp>
#include <machine/rbm_symmetry.hpp>
#include <model/kitaev.hpp>
#include <operators/aggregator.hpp>
#include <operators/bond_op.hpp>
#include <operators/derivative_op.hpp>
#include <operators/local_op.hpp>
#include <operators/local_op_chain.hpp>
#include <operators/overlap_op.hpp>
#include <operators/store_state.hpp>
#include <optimizer/plugin.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>
#include <tools/ini.hpp>
#include <tools/logger.hpp>

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

int main(int argc, char* argv[]) {
    int rc = ini::load(argc, argv);
    if (rc != 0) {
        return rc;
    }

    logger::init();

    omp_set_num_threads(ini::n_threads);
    Eigen::setNbThreads(1);

    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};
    model::kitaev km{ini::n_cells, ini::J};

    std::unique_ptr<machine::rbm_base> rbm;
    switch (ini::rbm) {
        case ini::rbm_t::BASIC:
            rbm = std::make_unique<machine::rbm_base>(ini::n_hidden,
                                                      km.get_lattice());
            break;
        case ini::rbm_t::SYMMETRY:
            rbm = std::make_unique<machine::rbm_symmetry>(ini::n_hidden,
                                                          km.get_lattice());
            break;
        default:
            return 1;
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

    optimizer::stochastic_reconfiguration sr{
        *rbm, *sampler, km.get_hamiltonian(), ini::sr_lr, ini::sr_reg};
    sr.register_observables();
    std::unique_ptr<optimizer::base_plugin> p;
    if (ini::sr_plugin.length() > 0) {
        if (ini::sr_plugin == "adam") {
            p = std::make_unique<optimizer::adam_plugin>(rbm->n_params);
        } else if (ini::sr_plugin == "momentum") {
            p = std::make_unique<optimizer::momentum_plugin>(rbm->n_params);
        } else {
            return 1;
        }
        sr.set_plugin(p.get());
    }

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
            Eigen::setNbThreads(0);
            sr.optimize();
            Eigen::setNbThreads(1);
            logger::newline();
            progress_bar(i + 1, ini::n_epochs,
                         sr.get_current_energy() / rbm->n_visible);
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

    return 0;
}
