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
#include <unordered_map>

namespace machine {

/**
 * @brief The RBM base class, which functions also as a RBM with no symmetry
 * used.
 */
class rbm_base : public abstract_machine {
    using Base = abstract_machine;

   public:
    const size_t n_alpha_;  ///< Number of hidden units.

   protected:
    Eigen::MatrixXcd weights_;  ///< The weights matrix.
    Eigen::MatrixXcd h_bias_;   ///< The hidden bias vector.
    Eigen::MatrixXcd v_bias_;   ///< The visible bias vector.

    const size_t n_vb_;  ///< Size of the visible bias vector.

    std::complex<double> (rbm_base::*psi_over_psi_)(const spin_state&,
                                                    const std::vector<size_t>&,
                                                    rbm_context&, rbm_context&,
                                                    bool*);

    size_t
        cosh_mode_;  ///< Cosh mode 0 for default 1 for approximate 2 for lncosh
    std::complex<double> (*cosh_)(const Eigen::MatrixXcd&);
    std::complex<double> (*lncosh_)(const Eigen::MatrixXcd&);
    void (*tanh_)(const Eigen::MatrixXcd&, Eigen::ArrayXXcd&);

    std::unordered_map<spin_state, std::complex<double>> lut_;
    std::vector<spin_state> lut_update_nums_;
    std::vector<std::complex<double>> lut_update_vals_;

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

    virtual rbm_context get_context(const spin_state& state) const override;

    virtual void update_context(const spin_state& state,
                                const std::vector<size_t>& flips,
                                rbm_context& context) const override;

    virtual Eigen::MatrixXcd derivative(
        const spin_state& state, const rbm_context& context) const override;

    virtual inline std::complex<double> psi(const spin_state& state,
                                            rbm_context& context) override {
        std::complex<double> ret = 1.;
        if (pfaffian_) {
            ret = pfaffian_->psi(state, context.pfaff());
        }
        return psi_default(state, context) * ret;
    }

    using Base::psi_over_psi;
    virtual std::complex<double> psi_over_psi(const spin_state& state,
                                              const std::vector<size_t>& flips,
                                              rbm_context& context,
                                              rbm_context& updated_context,
                                              bool* didupdate) override;

    virtual bool save(const std::string& name, bool silent = false) final;

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
    virtual std::complex<double> psi_notheta(const spin_state& state) const;

    /**
     * @brief Get the \psi(\sigma) form the RBM. Alternative version
     *
     * @param state The \sigma or the z-basis state.
     * @param context The precalculated context.
     *
     * @return \psi(\sigma)
     */
    virtual std::complex<double> psi_default(const spin_state& state,
                                             rbm_context& context);

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
        const spin_state& state, const std::vector<size_t>& flips,
        rbm_context& context, rbm_context& updated_context,
        bool* didupdate = 0);

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
        const spin_state& state, const std::vector<size_t>& flips,
        rbm_context& context, rbm_context& updated_context,
        bool* didupdate = 0) {
        return std::exp(log_psi_over_psi(state, flips, context, updated_context,
                                         didupdate));
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
        const spin_state& state, const std::vector<size_t>& flips,
        rbm_context& context, rbm_context& updated_context,
        bool* didupdate = 0);

    std::complex<double> cosh(rbm_context& context, const spin_state& state);
    std::complex<double> lncosh(rbm_context& context, const spin_state& state);

    void exchange_luts() override;

    std::complex<double> log_psi_over_psi_bias(
        const spin_state& state, const std::vector<size_t>& flips) const;
};
}  // namespace machine
