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

#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <vector>
//
#include <tools/ini.hpp>

namespace ini {

// Program variables

// Program
size_t seed = 421390484L;
size_t n_threads = 8;
std::string name = "";
std::string ini_file = "";
bool train = false;

// Model
model_t model = KITAEV;
size_t n_cells = 2;
int n_cells_b = -1;
double J = -1.;
double helper_strength = 0.;

// RBM
rbm_t rbm = BASIC;
size_t n_hidden = 3;
bool rbm_force = false;
size_t rbm_pop_mode = 0;
size_t rbm_cosh_mode = 0;
double rbm_weights = 0.0001;
double rbm_weights_imag = -1;
std::string rbm_weights_init_type = "";
size_t rbm_correlators = 1;

// Sampler
sampler_t sa_type = METROPOLIS;
size_t sa_n_samples = 1000;
size_t sa_metropolis_n_chains = 16;
size_t sa_metropolis_n_warmup_steps = 100;
size_t sa_metropolis_n_steps_per_sample = 10;
size_t sa_full_n_parallel_bits = 3;
std::string sa_exact_gs_file = "";
bool sa_metropolis_bond_flips = true;

// Optimizer
optimizer_t opt_type = SR;
std::string opt_plugin = "";
decay_t opt_lr = {0.001, 0.001, 1.};
decay_t opt_sr_reg = {1, 1e-4, 0.9};
double opt_sgd_real_factor = 1.;
double opt_adam_beta1 = 0.9;
double opt_adam_beta2 = 0.999;
double opt_adam_eps = 1e-8;
double opt_mom_alpha = 0.3;
bool opt_sr_use_gmres = false;

// Train
size_t n_epochs = 600;

}  // namespace ini

void ini::parse_ini_file(int argc, char* argv[]) {
    namespace po = boost::program_options;
    // Try to load only ini file from command line options, ignore all other
    // options.
    po::options_description desc("Allowed options");
    desc.add_options()("infile,i", po::value(&ini_file), "ini file for params");
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .allow_unregistered()
                  .run(),
              vm);
    po::notify(vm);
    if (ini_file.length() != 0) {
        std::ifstream file{ini_file};
        if (!file.good()) {
            throw std::runtime_error("INI File '" + ini_file +
                                     "' does not exist!");
        }
        file.close();
    }
}

int ini::load(int argc, char* argv[]) {
    namespace po = boost::program_options;
    try {
        // Define all options
        parse_ini_file(argc, argv);
        po::options_description desc("Allowed options");
        // clang-format off
    desc.add_options()
    // Program
    ("help,h",                                "produce help message")
    ("train",                                 po::bool_switch(&train),                      "train the RBM")
    ("seed",                                  po::value(&seed),                             "seed of the rng")
    ("infile,i",                              po::value<std::string>(),                     "ini file for params")
    ("name,n",                                po::value(&name),                             "set name of current rbm")
    ("n_threads,t",                           po::value(&n_threads),                        "set number of omp threads")
    // Model
    ("model.type",                            po::value(&model),                            "Model type.")
    ("model.n_cells,c",                       po::value(&n_cells),                          "set number of unit cells in one dimension")
    ("model.n_cells_b",                       po::value(&n_cells_b),                        "set number of unit cells in another dimension (if set to -1, use n_cells)")
    ("model.J",                               po::value(&J),                                "Interaction coefficient")
    ("model.helper_strength",                 po::value(&helper_strength),                  "Helper Hamiltonian strength")
    // RBM
    ("rbm.type",                              po::value(&rbm),                              "set rbm type")
    ("rbm.n_hidden",                          po::value(&n_hidden),                         "set number of hidden units")
    ("rbm.force,f",                           po::bool_switch(&rbm_force),                  "force retraining of RBM")
    ("rbm.weights",                           po::value(&rbm_weights),                      "set stddev for weights initialization")
    ("rbm.weights_imag",                      po::value(&rbm_weights_imag),                 "set stddev for imag weights initialization (if not set = rbm.weights)")
    ("rbm.weights_type",                      po::value(&rbm_weights_init_type),            "Initialization type for RBM weights, for special initial states")
    ("rbm.correlators",                       po::value(&rbm_correlators),                  "enables correlators if set to 1 and correlators are available for the model")
    ("rbm.pop_mode",                          po::value(&rbm_pop_mode),                     "switches between Psi calculation modes.")
    ("rbm.cosh_mode",                         po::value(&rbm_cosh_mode),                    "turns cosh approximation on")
    // Sampler
    ("sampler.type",                          po::value(&sa_type),                          "set sampler type")
    ("sampler.n_samples",                     po::value(&sa_n_samples),                     "set sampler n sampler (metropolis only)")
    ("sampler.full.n_parallel_bits",          po::value(&sa_full_n_parallel_bits),          "set number of bits executed in parallel in full sampling")
    ("sampler.metropolis.n_chains",           po::value(&sa_metropolis_n_chains),           "set number of MCMC chains in Metropolis sampling")
    ("sampler.metropolis.n_warmup_steps",     po::value(&sa_metropolis_n_warmup_steps),     "set number of MCMC warmup steps")
    ("sampler.metropolis.n_steps_per_sample", po::value(&sa_metropolis_n_steps_per_sample), "set number of MCMC steps between a sample")
    ("sampler.exact.gs_file",                 po::value(&sa_exact_gs_file),                 "set file of ground state for exact sampling")
    ("sampler.metropolis.bond_flips",         po::value(&sa_metropolis_bond_flips),         "use bond flips for update proposal")
    // Optimizer
    ("optimizer.type",                        po::value(&opt_type),                         "set optimizer type")
    ("optimizer.learning_rate,l",             po::value(&opt_lr)->multitoken(),             "set learning rate optionally with decay factor")
    ("optimizer.sr.regularization,r",         po::value(&opt_sr_reg)->multitoken(),         "set regularization diagonal shift decay rate optionally with decay factor")
    ("optimizer.sr.use_gmres,r",              po::value(&opt_sr_use_gmres),                 "Use efficient GMRES for covariance matrix inversion")
    ("optimizer.sgd.real_factor,r",           po::value(&opt_sgd_real_factor),              "set the factor the real part of the update vector is divided by (default 1.)")
    ("optimizer.plugin",                      po::value(&opt_plugin),                       "set optional plugin for SR adam/momentum")
    ("optimizer.adam.beta1",                  po::value(&opt_adam_beta1),                   "set ADAM plug beta1")
    ("optimizer.adam.beta2",                  po::value(&opt_adam_beta2),                   "set ADAM plug beta2")
    ("optimizer.adam.eps",                    po::value(&opt_adam_eps),                     "set ADAM plug eps")
    ("optimizer.mom.alpha",                   po::value(&opt_mom_alpha),                    "set momentum plug alpha")
    // Train
    ("n_epochs,e",                            po::value(&n_epochs),                         "set number of epochs for training");
        // clang-format on
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        // Also parse ini file if available.
        if (ini_file.length() > 0) {
            std::ifstream config_stream(ini_file);
            po::store(po::parse_config_file(config_stream, desc), vm);
            config_stream.close();
        }

        po::notify(vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 1;
        }

        // Generate a name `n[n_cells]_h[rbm.n_hidden]` if not set.
        if (!vm.count("name")) {
            std::ostringstream oss;
            oss << "n" << n_cells << "_h" << n_hidden;
            name = oss.str();
        }

    } catch (const po::unknown_option& e) {
        std::cerr << e.what();
        return -1;
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << std::endl;
        return -2;
    }

    std::cout << "Starting '" << name << "'!" << std::endl;

    return 0;
}

std::istream& ini::operator>>(std::istream& is, ini::rbm_t& rbm) {
    std::string token;
    is >> token;
    if (token == "basic") {
        rbm = ini::rbm_t::BASIC;
    } else if (token == "symmetry") {
        rbm = ini::rbm_t::SYMMETRY;
    } else {
        throw std::runtime_error("RBM Type '" + token + "' not available!");
    }
    return is;
}

std::istream& ini::operator>>(std::istream& is, ini::sampler_t& sa) {
    std::string token;
    is >> token;
    if (token == "full") {
        sa = ini::sampler_t::FULL;
    } else if (token == "exact") {
        sa = ini::sampler_t::EXACT;
    } else if (token == "metropolis") {
        sa = ini::sampler_t::METROPOLIS;
    } else {
        throw std::runtime_error("Sampler Type '" + token + "' not available!");
    }
    return is;
}

std::istream& ini::operator>>(std::istream& is, ini::optimizer_t& opt) {
    std::string token;
    is >> token;
    if (token == "SR") {
        opt = ini::optimizer_t::SR;
    } else if (token == "SGD") {
        opt = ini::optimizer_t::SGD;
    } else {
        throw std::runtime_error("Optimizer Type '" + token +
                                 "' not available!");
    }
    return is;
}

std::istream& ini::operator>>(std::istream& is, ini::model_t& model) {
    std::string token;
    is >> token;
    if (token == "kitaev") {
        model = ini::model_t::KITAEV;
    } else if (token == "kitaevS3") {
        model = ini::model_t::KITAEV_S3;
    } else if (token == "isingS3") {
        model = ini::model_t::ISING_S3;
    } else if (token == "toric") {
        model = ini::model_t::TORIC;
    } else {
        throw std::runtime_error("Model Type '" + token + "' not available!");
    }
    return is;
}

void ini::validate(boost::any& v, const std::vector<std::string>& values,
                   ini::decay_t*, int) {
    std::vector<std::string> vals;

    // Split all strings by comma, for better support of ini loading.
    for (std::string s : values) {
        size_t pos = 0;
        std::string token;
        while ((pos = s.find(',')) != std::string::npos) {
            token = s.substr(0, pos);
            vals.push_back(token);
            s.erase(0, pos + 1);
        }
        vals.push_back(s);
    }
    ini::decay_t t;
    t.initial = std::stod(vals[0]);
    // if size of string is 3 load min and decay, otherwise set min and decay
    // to inital and 1. for a constant val.
    if (vals.size() == 3) {
        t.min = std::stod(vals[1]);
        t.decay = std::stod(vals[2]);
    } else {
        t.min = t.initial;
        t.decay = 1.;
    }
    v = t;
}
