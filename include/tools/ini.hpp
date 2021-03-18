/**
 * Copyright (c) 2021 David Bucher <David.Bucher@physik.lmu.de>
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

#include <boost/any.hpp>
#include <string>
#include <vector>

namespace ini {
struct decay_t {
    double initial;
    double min;
    double decay;
};

enum rbm_t { BASIC, SYMMETRY };
enum sampler_t { FULL, METROPOLIS, EXACT };

extern std::istream& operator>>(std::istream&, rbm_t&);
extern std::istream& operator>>(std::istream&, sampler_t&);
extern void validate(boost::any&, const std::vector<std::string>&, decay_t*,
                     int);
// Programm
extern size_t seed;
extern size_t n_threads;
extern std::string name;
extern std::string ini_file;
extern bool train;

// Model
extern size_t n_cells;
extern double J;

// RBM
extern rbm_t rbm;
extern size_t n_hidden;
extern bool rbm_force;
extern double rbm_weights;
extern double rbm_weights_imag;

// Sampler
extern sampler_t sa_type;
extern size_t sa_metropolis_n_chains;
extern size_t sa_metropolis_n_steps_per_sample;
extern size_t sa_metropolis_n_warmup_steps;
extern size_t sa_n_samples;
extern size_t sa_full_n_parallel_bits;
extern std::string sa_exact_gs_file;

// Optimizer
extern std::string sr_plugin;
extern decay_t sr_lr;
extern decay_t sr_reg;

// Train
extern size_t n_epochs;

void parse_ini_file(int argc, char* argv[]);
int load(int argc, char* argv[]);

}  // namespace ini
