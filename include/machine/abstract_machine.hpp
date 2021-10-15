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

#pragma once

#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <random>
//
#include <lattice/bravais.hpp>
#include <machine/context.hpp>
#include <machine/correlator.hpp>
#include <machine/pfaffian.hpp>

namespace machine {

/**
 * @brief The RBM base class, which functions also as a RBM with no symmetry
 * used.
 */
class abstract_machine {
   public:
    size_t n_visible;

   protected:
    const size_t n_params_;  ///< Number of total parameters.

    lattice::bravais& lattice_;  ///< Reference to the Lattice.

    size_t n_updates_ = 0;  ///< The number of updates received.

    std::vector<std::unique_ptr<correlator>>
        correlators_;                         ///< Correlator references
    std::unique_ptr<pfaffian> pfaffian_ = 0;  ///< Pfaffian reference

    /**
     * @breif abstract machine protected constructor
     *
     * @param lattice Bravais Lattice reference
     * @param n_params Number of params (default 0) */
    abstract_machine(lattice::bravais& lattice, size_t n_params = 0)
        : n_visible{lattice.n_total}, n_params_{n_params}, lattice_{lattice} {}

   public:
    /**
     * @brief Default virtual destructor.
     */
    virtual ~abstract_machine() = default;

    /**
     * @brief Returns the number of updates received.
     *
     * @return `n_updates_`.
     */
    inline size_t get_n_updates() { return n_updates_; }

    /**
     * @brief Lattice getter.
     *
     * @return lattice reference.
     */
    inline lattice::bravais& get_lattice() { return lattice_; }

    /**
     * @brief n_params_ getter
     *
     * @return n_params of RBM (without correlators)
     */
    inline size_t get_n_params() const {
        if (pfaffian_) {
            return get_n_neural_params() + pfaffian_->get_n_params();
        } else {
            return get_n_neural_params();
        }
    }

    /**
     * @brief n_params_ getter
     *
     * @return n_params of RBM without pfaffian
     */
    inline size_t get_n_neural_params() const {
        size_t ret = n_params_;
        for (const auto& corr : correlators_) {
            ret += corr->get_n_params();
        }
        return ret;
    }

    virtual inline size_t symmetry_size() const { return 1; }

    /**
     * @brief Initializes the weights randomly with given standard deviations.
     * The imaginary part can have a different standard deviation.
     *
     * @param rng Reference to the RNG.
     * @param std_dev Standard Deviation.
     * @param std_dev_imag Standard Deviation for the imaginary part, default
     * -1. will use the same as real part.
     * @param type Type of default initialization. Lattice dependent.
     */
    virtual void initialize_weights(std::mt19937& rng, double std_dev,
                                    double std_dev_imag = -1.,
                                    const std::string& type = ""){};

    /**
     * @brief Updates the weights with a vector received from the optimizer.
     *
     * @param dw A update vector of size `n_params`.
     */
    virtual void update_weights(const Eigen::MatrixXcd& dw){};

    /**
     * @brief Updates the weights with a vector received from the optimizer.
     *
     * @param dw A update vector of size `n_params`.
     */
    inline void update_weights_nc(const Eigen::MatrixXcd& dw) {
        update_weights(dw);
        n_updates_--;
    };

    /**
     * @brief Calculates the rbm_context
     *
     * @param state Tht \sigma, the z-basis state in MatrixXcd vector form.
     *
     * @return A new RBM context, including the thetas.
     */
    virtual rbm_context get_context(const Eigen::MatrixXcd& state) const = 0;

    /**
     * @brief Updates the context if a number of particular spins are flipped.
     * This is a more efficient computation than just recalculating the context
     * in each step.
     *
     * @param state The old state.
     * @param flips A vector if indices, which spins are going to be flipped.
     * @param context The precalculated context.
     */
    virtual void update_context(const Eigen::MatrixXcd& state,
                                const std::vector<size_t>& flips,
                                rbm_context& context) const = 0;

    /**
     * @brief Calculates the derivative of the RBM with repsect to the
     * parameters.
     *
     * @param state The current state.
     * @param context The precalculated context.
     *
     * @return MatrixXcd vector of size `n_params`.
     */
    virtual Eigen::MatrixXcd derivative(const Eigen::MatrixXcd& state,
                                        const rbm_context& context) const = 0;

    /**
     * @brief Get the \psi(\sigma) form the RBM.
     *
     * @param state The \sigma or the z-basis state.
     * @param context The precalculated context.
     *
     * @return \psi(\sigma)
     */
    virtual std::complex<double> psi(const Eigen::MatrixXcd& state,
                                     rbm_context& context) = 0;

    /**
     * @brief Computes the ratio of \psi with some spins. Function pointer
     * wrapper
     *
     * @param state The current state.
     * @param flips The vector of indices of spins to flip.
     * @param context The precalculated context.
     * @param updated_context A updated context reference to spare the
     * reupdating of the context if a flip combination was accepted.
     *
     * @return returns the ratio \psi(\sigma')/\psi(\sigma)
     */
    virtual std::complex<double> psi_over_psi(const Eigen::MatrixXcd& state,
                                              const std::vector<size_t>& flips,
                                              rbm_context& context,
                                              rbm_context& updated_context,
                                              bool* did_update = 0) = 0;

    /**
     * @brief Computes ratio of \psi with some spins flipped to the
     * \psi with no spins flipped.
     *
     * @param state The current state.
     * @param flips The vector of indices of spins to flip.
     * @param context The precalculated context.
     *
     * @return returns the ratio \psi(\sigma')/\psi(\sigma)
     */
    inline std::complex<double> psi_over_psi(const Eigen::MatrixXcd& state,
                                             const std::vector<size_t>& flips,
                                             rbm_context& context) {
        rbm_context updated_context = context;
        return psi_over_psi(state, flips, context, updated_context);
    }

    /**
     * @brief Saves the current state to the file '`name`.rbm'.
     *
     * @param name the filename
     *
     * @return true if worked.
     */
    virtual bool save(const std::string& name, bool silent = false) {
        return true;
    };

    /**
     * @brief Loads the state from the file '`name`.rbm'.
     *
     * @param name the filename
     *
     * @return true if worked.
     */
    virtual bool load(const std::string& name) { return true; };

    virtual inline void add_correlator(
        const std::vector<std::vector<size_t>>& correlator) {}

    inline void add_correlators(
        const std::vector<std::vector<std::vector<size_t>>>& correlators) {
        for (const auto& c : correlators) {
            add_correlator(c);
        }
    }

    inline std::unique_ptr<pfaffian>& add_pfaffian(
        const std::vector<double>& symm, bool no_updating = false) {
        pfaffian_ = std::make_unique<pfaffian>(lattice_, symm, no_updating);
        return pfaffian_;
    }

    inline bool has_pfaffian() const { return pfaffian_ != 0; }
    inline pfaffian& get_pfaffian() const {
        if (!has_pfaffian()) {
            throw std::runtime_error("There is no Pfaffian to get");
        } else {
            return *pfaffian_;
        }
    }

    virtual void exchange_luts() {}
};
}  // namespace machine
