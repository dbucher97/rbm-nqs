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

//
#include <machine/abstract_machine.hpp>

namespace machine {

/**
 * @brief The RBM base class, which functions also as a RBM with no symmetry
 * used.
 */
class rbm_base : public abstract_machine {
    using Base = abstract_machine;

   public:
    const size_t n_alpha;  ///< Number of hidden units.

   protected:
    Eigen::MatrixXcd weights_;  ///< The weights matrix.
    Eigen::MatrixXcd h_bias_;   ///< The hidden bias vector.
    Eigen::MatrixXcd v_bias_;   ///< The visible bias vector.

    const size_t n_vb_;  ///< Size of the visible bias vector.

    std::complex<double> (rbm_base::*psi_)(const Eigen::MatrixXcd&,
                                           const rbm_context&) const;
    std::complex<double> (rbm_base::*psi_over_psi_)(const Eigen::MatrixXcd&,
                                                    const std::vector<size_t>&,
                                                    const rbm_context&,
                                                    rbm_context&) const;

    Eigen::MatrixXcd (*cosh_)(const Eigen::MatrixXcd&);
    Eigen::MatrixXcd (*tanh_)(const Eigen::MatrixXcd&);

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
     * @brief Weights getter.
     *
     * @return The weights.
     */
    inline Eigen::MatrixXcd& get_weights() { return weights_; }
    /**
     * @brief Hidden bias getter.
     *
     * @return The hidden bias.
     */
    inline Eigen::MatrixXcd& get_h_bias() { return h_bias_; }
    /**
     * @brief Visible bias getter.
     *
     * @return The visible bias.
     */
    inline Eigen::MatrixXcd& get_v_bias() { return v_bias_; }

    virtual void initialize_weights(std::mt19937& rng, double std_dev,
                                    double std_dev_imag = -1.,
                                    const std::string& type = "") final;

    virtual void update_weights(const Eigen::MatrixXcd&) override;

    virtual rbm_context get_context(
        const Eigen::MatrixXcd& state) const override;

    virtual void update_context(const Eigen::MatrixXcd& state,
                                const std::vector<size_t>& flips,
                                rbm_context& context) const override;

    virtual Eigen::MatrixXcd derivative(
        const Eigen::MatrixXcd& state,
        const rbm_context& context) const override;

    virtual inline std::complex<double> psi(
        const Eigen::MatrixXcd& state,
        const rbm_context& context) const override {
        std::complex<double> ret = 1.;
        if (pfaffian_) ret = pfaffian_->psi(state, context.pfaff());
        return ret * (this->*psi_)(state, context);
    }

    using Base::psi_over_psi;
    virtual inline std::complex<double> psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const rbm_context& context,
        rbm_context& updated_context) const override {
        std::complex<double> ret = 1.;
        if (pfaffian_)
            ret =
                pfaffian_->psi_over_psi(state, flips, updated_context.pfaff());

        return ret *
               (this->*psi_over_psi_)(state, flips, context, updated_context);
    }

    virtual bool save(const std::string& name) final;

    virtual bool load(const std::string& name) final;

    virtual void add_correlator(
        const std::vector<std::vector<size_t>>& correlator) override;

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
     * @param context The precalculated context.
     *
     * @return \psi(\sigma)
     */
    virtual std::complex<double> psi_default(const Eigen::MatrixXcd& state,
                                             const rbm_context& context) const;

    /**
     * @brief Get the \psi(\sigma) form the RBM. Alternative version
     *
     * @param state The \sigma or the z-basis state.
     * @param context The precalculated context.
     *
     * @return \psi(\sigma)
     */
    virtual std::complex<double> psi_alt(const Eigen::MatrixXcd& state,
                                         const rbm_context& context) const;

    /**
     * @brief Computes the log of a ratio of \psi with some spins flipped to the
     * \psi with no spins flipped.
     *
     * @param state The current state.
     * @param flips The vector of indices of spins to flip.
     * @param context The precalculated context.
     * @param updated_context A updated context reference to spare the
     * reupdating of the context if a flip combination was accepted.
     *
     * @return returns the ratio log(\psi(\sigma')/\psi(\sigma))
     */
    virtual std::complex<double> log_psi_over_psi(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const rbm_context& context, rbm_context& updated_context) const;

    /**
     * @brief Computes the ratio of \psi with some spins
     * flipped to the \psi with no spins flipped. Direct calculation, without
     * `log_psi_over_psi`.
     *
     * @param state The current state.
     * @param flips The vector of indices of spins to flip.
     * @param context The precalculated context.
     * @param updated_context A updated context reference to spare the
     * reupdating of the context if a flip combination was accepted.
     *
     * @return returns the ratio \psi(\sigma')/\psi(\sigma)
     */
    inline std::complex<double> psi_over_psi_default(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const rbm_context& context, rbm_context& updated_context) const {
        return std::exp(
            log_psi_over_psi(state, flips, context, updated_context));
    }

    /**
     * @brief Alternative Version: Computes the ratio of \psi with some spins
     * flipped to the \psi with no spins flipped. Direct calculation, without
     * `log_psi_over_psi`.
     *
     * @param state The current state.
     * @param flips The vector of indices of spins to flip.
     * @param context The precalculated context.
     * @param updated_context A updated context reference to spare the
     * reupdating of the context if a flip combination was accepted.
     *
     * @return returns the ratio \psi(\sigma')/\psi(\sigma)
     */
    virtual std::complex<double> psi_over_psi_alt(
        const Eigen::MatrixXcd& state, const std::vector<size_t>& flips,
        const rbm_context& context, rbm_context& updated_context) const;
};
}  // namespace machine
