/*
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

/**
 * @brief The `ini` namespace storing the configuration of the project, every
 * setting is stored as a `extern` attribute in the namespace, this allows for
 * global access to those attributes.
 * The config loading and command line options are handled by
 * `boost_program_options`.
 */
namespace ini {
/**
 * @brief Decay type with only initial, min, decay values without logic.
 */
struct decay_t {
    double initial;
    double min;
    double decay;
};

/**
 * @brief RBM Type enum
 */
enum rbm_t { BASIC, SYMMETRY };
/**
 * @brief Sampler Type enum
 */
enum sampler_t { FULL, METROPOLIS, EXACT };
/**
 * @brief Optimizer Type enum
 */
enum optimizer_t { SGD, SR };
/**
 * @brief Model Type enum
 */
enum model_t { KITAEV, KITAEV_S3, ISING_S3, TORIC };

/**
 * @brief `istream` wrapper for loading RBM type.
 *
 * @param input `istream` reference.
 * @param rbm `rbm_t` reference.
 *
 * @return `istream` reference.
 */
extern std::istream& operator>>(std::istream& input, rbm_t& rbm);
/**
 * @brief `istream` wrapper for loading Sampler type.
 *
 * @param input `istream` reference.
 * @param sampler `sampler_t` reference.
 *
 * @return `istream` reference.
 */
extern std::istream& operator>>(std::istream& input, sampler_t& sampler);
/**
 * @brief `istream` wrapper for loading Optimizer type.
 *
 * @param input `istream` reference.
 * @param optimizer `optimizer_t` reference.
 *
 * @return `istream` reference.
 */
extern std::istream& operator>>(std::istream& input, optimizer_t& optimizer);
/**
 * @brief `istream` wrapper for loading Model type.
 *
 * @param input `istream` reference.
 * @param optimizer `optimizer_t` reference.
 *
 * @return `istream` reference.
 */
extern std::istream& operator>>(std::istream& input, model_t& optimizer);
/**
 * @brief Loads an `decay_t` with a string vector from `boost_program_options`.
 *
 * @param f will be the decay_t object
 * @param strs vector of strings to load from
 * @param decay_t Don't know, copied from StackOverflow 🙃...
 * @param int
 */
extern void validate(boost::any& f, const std::vector<std::string>& strs,
                     decay_t*, int);

/**
 * @brief Checks if a ini file is specified in the command line options, if
 * this is the case `load` also reads the `ini` file.
 *
 * @param argc Argument count
 * @param argv[] Arguments
 */
void parse_ini_file(int argc, char* argv[]);
/**
 * @brief Loads the configuration form the command line options and/or ini
 * file.
 *
 * @param argc Argument count
 * @param argv[] Arguments
 *
 * @return return code.
 */
int load(int argc, char* argv[]);

// All the configuration variables:

// Programm
extern size_t seed;
extern size_t n_threads;
extern std::string name;
extern std::string ini_file;
extern bool train;

// Model
extern model_t model;
extern size_t n_cells;
extern int n_cells_b;
extern double J;
extern double helper_strength;

// RBM
extern rbm_t rbm;
extern size_t n_hidden;
extern double alpha; // Overrides n_hidden
extern bool rbm_force;
extern double rbm_weights;
extern double rbm_weights_imag;
extern std::string rbm_weights_init_type;
extern size_t rbm_correlators;
extern size_t rbm_pop_mode;
extern size_t rbm_cosh_mode;
extern bool rbm_pfaffian;
extern size_t rbm_pfaffian_symmetry;

// Sampler
extern sampler_t sa_type;
extern size_t sa_metropolis_n_chains;
extern size_t sa_metropolis_n_steps_per_sample;
extern size_t sa_metropolis_n_warmup_steps;
extern size_t sa_n_samples;
extern size_t sa_full_n_parallel_bits;
extern std::string sa_exact_gs_file;
extern bool sa_metropolis_bond_flips;

// Optimizer
extern optimizer_t opt_type;
extern std::string opt_plugin;
extern decay_t opt_lr;
extern decay_t opt_sr_reg;
extern bool opt_sr_iterative;
extern size_t opt_sr_max_iterations;
extern double opt_sgd_real_factor;
extern double opt_adam_beta1;
extern double opt_adam_beta2;
extern double opt_adam_eps;
extern double opt_mom_alpha;

// Train
extern size_t n_epochs;

}  // namespace ini
