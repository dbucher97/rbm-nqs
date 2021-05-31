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

// #define ALT_POP

#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <random>
//
#include <lattice/bravais.hpp>
#include <machine/correlator.hpp>

namespace machine {

/**
 * @brief The RBM base class, which functions also as a RBM with no symmetry
 * used.
 */
class rbm_base {
   public:
    const size_t n_alpha;    ///< Number of hidden units.
    const size_t n_visible;  ///< Number of visible units.

   protected:
    const size_t n_params_;  ///< Number of total parameters.

    lattice::bravais& lattice_;  ///< Reference to the Lattice.

    Eigen::MatrixXcd weights_;  ///< The weights matrix.
    Eigen::MatrixXcd h_bias_;   ///< The hidden bias vector.
    Eigen::MatrixXcd v_bias_;   ///< The visible bias vector.

    const size_t n_vb_;  ///< Size of the visible bias vector.

    size_t n_updates_;  ///< The number of updates received.

    std::complex<double> (rbm_base::*psi_)(const Eigen::MatrixXcd&,
                                           const Eigen::MatrixXcd&) const;
    std::complex<double> (rbm_base::*psi_over_psi_)(const Eigen::MatrixXcd&,
                                                    const std::vector<size_t>&,
                                                    const Eigen::MatrixXcd&,
                                                    Eigen::MatrixXcd&) const;

    Eigen::MatrixXcd (*cosh_)(const Eigen::MatrixXcd&);
    Eigen::MatrixXcd (*tanh_)(const Eigen::MatrixXcd&);

    std::vector<std::unique_ptr<correlator>> correlators_;

    /**
     * @brief Hidden Constructor for the RBM used by derived classes to get
     * access to custor visible bias sizes.
     *
     * @param n_alpha Number of hidden units.
     * @param n_vb Number of visible units.
     * @param lattice Reference to the Lattice.
     */
    rbm_base(size_t n_alpha, size_t n_vb, lattice::bravais& lattice,
             size_t pop_mode = 0, size_t cosh_mode = 0);

   public:
    /**
     * @brief Constructor for the base RBM withoud symmtry.
     *
     * @param n_alpha Number of hidden units.
     * @param lattice Reference to the Lattice.
     */
    rbm_base(size_t n_alpha, lattice::bravais& lattice, size_t pop_mode = 0,
             size_t cosh_mode = 0);
    /**
     * @brief Default virtual destructor.
     */
    virtual ~rbm_base() = default;

    /**
     * @brief Weights getter.
     *
     * @return The weights.
     */
    Eigen::MatrixXcd& get_weights() { return weights_; }
    /**
     * @brief Hidden bias getter.
     *
     * @return The hidden bias.
     */
    Eigen::MatrixXcd& get_h_bias() { return h_bias_; }
    /**
     * @brief Visible bias getter.
     *
     * @return The visible bias.
     */
    Eigen::MatrixXcd& get_v_bias() { return v_bias_; }

    /**
     * @brief Returns the number of updates received.
     *
     * @return `n_updates_`.
     */
    size_t get_n_updates() { return n_updates_; }

    /**
     * @brief Lattice getter.
     *
     * @return lattice reference.
     */
    lattice::bravais& get_lattice() { return lattice_; }

    /**
     * @brief n_params_ getter
     *
     * @return n_params of RBM (without correlators)
     */
    size_t get_n_params() const;

    virtual size_t symmetry_size() const { return 1; }

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
    void initialize_weights(std::mt19937& rng, double std_dev,
                            double std_dev_imag = -1.,
                            const std::string& type = "");

    /**
     * @brief Updates the weights with a vector received from the optimizer.
     *
     * @param Eigen::MatrixXcd A update vector of size `n_params`.
     */
    void update_weights(const Eigen::MatrixXcd&);

    /**
     * @brief Calculates the angles \theta_{js}
     *
     * @param state Te \sigma, the z-basis state in MatrixXcd vector form.
     *
     * @return A new MatrixXcd with the calculated \thetas, 2D if symmetry
     * involved.
     */
    virtual Eigen::MatrixXcd get_thetas(const Eigen::MatrixXcd& state) const;

    /**
     * @brief Updates the thetas if a number of particular spins are flipped.
     * This is a more efficient computation than just recalculating the thetas
     * in each step.
     *
     * @param state The old state.
     * @param flips A vector if indices, which spins are going to be flipped.
     * @param thetas The precalculated \thetas.
     */
    virtual void update_thetas(const Eigen::MatrixXcd& state,
                               const std::vector<size_t>& flips,
                               Eigen::MatrixXcd& thetas) const;

    /**
     * @brief Calculates the derivative of the RBM with repsect to the
     * parameters.
     *
     * @param state The current state.
     * @param thetas The precalculated \thetas.
     *
     * @return MatrixXcd vector of size `n_params`.
     */
    virtual Eigen::MatrixXcd derivative(const Eigen::MatrixXcd& state,
                                        const Eigen::MatrixXcd& thetas) const;

    /**
     * @brief Get the \psi(\sigma) form the RBM.
     *
     * @param state The \sigma or the z-basis state.
     * @param thetas The pre calculated angles \theta, see the paper.
     *
     * @return \psi(\sigma)
     */
    inline std::complex<double> psi(const Eigen::MatrixXcd& state,
                                    const Eigen::MatrixXcd& thetas) const {
        return (this->*psi_)(state, thetas);
    }

    /**
     * @briefComputes the ratio of \psi with some spins. Function pointer
     * wrapper
     *
     * @param state The current state.
     * @param flips The vector of indices of spins to flip.
     * @param thetas The precalculated thetas.
     * @param updated_thetas A updated thetas reference to spare the reupdating
     * of the thetas if a flip combination was accepted.
     *
     * @return returns the ratio \psi(\sigma')/\psi(\sigma)
     */
    inline std::complex<double> psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const Eigen::MatrixXcd& thetas,
        Eigen::MatrixXcd& updated_thetas) const {
        return (this->*psi_over_psi_)(state, flips, thetas, updated_thetas);
    }

    /**
     * @brief Computes ratio of \psi with some spins flipped to the
     * \psi with no spins flipped.
     *
     * @param state The current state.
     * @param flips The vector of indices of spins to flip.
     * @param thetas The precalculated thetas.
     *
     * @return returns the ratio \psi(\sigma')/\psi(\sigma)
     */
    inline std::complex<double> psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const Eigen::MatrixXcd& thetas) const {
        Eigen::MatrixXcd updated_thetas = thetas;
        return psi_over_psi(state, flips, thetas, updated_thetas);
    }

    /**
     * @brief Computes the acceptance value of a specific flip combination and
     * accepts the flip if `prob` is smaller than the acceptance value.
     *
     * @param prob Random double between zero and one.
     * @param state The current state.
     * @param flips The vector of proposed spin flips.
     * @param thetas The precalculated \thetas.
     *
     * @return A bool if proposed flips have been accepted.
     */
    bool flips_accepted(double prob, const Eigen::MatrixXcd& state,
                        const std::vector<size_t>& flips,
                        Eigen::MatrixXcd& thetas) const;

    /**
     * @brief Saves the current state to the file '`name`.rbm'.
     *
     * @param name the filename
     *
     * @return true if worked.
     */
    bool save(const std::string& name);

    /**
     * @brief Loads the state from the file '`name`.rbm'.
     *
     * @param name the filename
     *
     * @return true if worked.
     */
    bool load(const std::string& name);

    virtual void add_correlator(
        const std::vector<std::vector<size_t>>& correlator);
    void add_correlators(
        const std::vector<std::vector<std::vector<size_t>>>& correlators);

   protected:
    /**
     * @brief The bias part of the psi calculation
     *
     * @param state current state
     *
     * @return the bias part of psi.
     */
    virtual std::complex<double> psi_notheta(
        const Eigen::MatrixXcd& state) const;

    /**
     * @brief Get the \psi(\sigma) form the RBM. Alternative version
     *
     * @param state The \sigma or the z-basis state.
     * @param thetas The pre calculated angles \theta, see the paper.
     *
     * @return \psi(\sigma)
     */
    virtual std::complex<double> psi_default(
        const Eigen::MatrixXcd& state, const Eigen::MatrixXcd& thetas) const;

    /**
     * @brief Get the \psi(\sigma) form the RBM. Alternative version
     *
     * @param state The \sigma or the z-basis state.
     * @param thetas The pre calculated angles \theta, see the paper.
     *
     * @return \psi(\sigma)
     */
    virtual std::complex<double> psi_alt(const Eigen::MatrixXcd& state,
                                         const Eigen::MatrixXcd& thetas) const;

    /**
     * @brief Computes the log of a ratio of \psi with some spins flipped to the
     * \psi with no spins flipped.
     *
     * @param state The current state.
     * @param flips The vector of indices of spins to flip.
     * @param thetas The precalculated thetas.
     * @param updated_thetas A updated thetas reference to spare the reupdating
     * of the thetas if a flip combination was accepted.
     *
     * @return returns the ratio log(\psi(\sigma')/\psi(\sigma))
     */
    virtual std::complex<double> log_psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const Eigen::MatrixXcd& thetas, Eigen::MatrixXcd& updated_thetas) const;

    /**
     * @brief Computes the ratio of \psi with some spins
     * flipped to the \psi with no spins flipped. Direct calculation, without
     * `log_psi_over_psi`.
     *
     * @param state The current state.
     * @param flips The vector of indices of spins to flip.
     * @param thetas The precalculated thetas.
     * @param updated_thetas A updated thetas reference to spare the reupdating
     * of the thetas if a flip combination was accepted.
     *
     * @return returns the ratio \psi(\sigma')/\psi(\sigma)
     */
    inline std::complex<double> psi_over_psi_default(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const Eigen::MatrixXcd& thetas,
        Eigen::MatrixXcd& updated_thetas) const {
        return std::exp(log_psi_over_psi(state, flips, thetas, updated_thetas));
    }

    /**
     * @brief Alternative Version: Computes the ratio of \psi with some spins
     * flipped to the \psi with no spins flipped. Direct calculation, without
     * `log_psi_over_psi`.
     *
     * @param state The current state.
     * @param flips The vector of indices of spins to flip.
     * @param thetas The precalculated thetas.
     * @param updated_thetas A updated thetas reference to spare the reupdating
     * of the thetas if a flip combination was accepted.
     *
     * @return returns the ratio \psi(\sigma')/\psi(\sigma)
     */
    virtual std::complex<double> psi_over_psi_alt(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const Eigen::MatrixXcd& thetas, Eigen::MatrixXcd& updated_thetas) const;
};
}  // namespace machine
