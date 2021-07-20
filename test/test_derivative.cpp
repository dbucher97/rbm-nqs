#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <random>
//
#include <lattice/honeycomb.hpp>
#include <machine/pfaffian.hpp>
#include <machine/rbm_base.hpp>
#include <machine/rbm_symmetry.hpp>

#define SEED 52391500

#define SETUP(MACHINE)                                                \
    std::mt19937 rng(SEED);                                           \
    lattice::honeycomb lat{5};                                        \
    MACHINE rbm{10, lat};                                             \
    rbm.initialize_weights(rng, 0.1);                                 \
    std::uniform_int_distribution<size_t> f_dist(0, lat.n_total - 1); \
    std::uniform_int_distribution<size_t> u_dist(0, 1);               \
    Eigen::MatrixXcd state = Eigen::MatrixXd::Ones(lat.n_total, 1);   \
    for (size_t i = 0; i < lat.n_total; i++)                          \
        if (u_dist(rng)) state(i) *= -1;                              \
    auto context = rbm.get_context(state);

using namespace machine;

template <typename Machine>
void test_update() {
    SETUP(Machine);

    for (size_t i = 0; i < 100; i++) {
        std::vector<size_t> flips;
        flips.push_back(f_dist(rng));
        size_t second = f_dist(rng);
        while (second == flips[0]) {
            second = f_dist(rng);
        }
        flips.push_back(second);

        rbm.update_context(state, flips, context);

        for (auto& f : flips) state(f) *= -1;
    }

    auto context2 = rbm.get_context(state);

    double diff = (context.thetas - context2.thetas).array().abs().mean();
    // std::cout << diff << std::endl;
    EXPECT_NEAR(diff, 0, 1e-15);
}

template <typename Machine>
void test_derivative() {
    SETUP(rbm_base);

    std::normal_distribution<double> n_dist;

    for (size_t i = 0; i < 100; i++) {
        Eigen::MatrixXcd dw(rbm.get_n_params(), 1);
        for (int j = 0; j < dw.size(); j++) {
            dw(j) = std::complex<double>(n_dist(rng), n_dist(rng));
        }

        dw *= 1e-8;

        Eigen::MatrixXcd drbm = rbm.derivative(state, context);

        std::complex<double> psi1 = rbm.psi(state, context);
        psi1 += psi1 * (dw.transpose() * drbm)(0);
        rbm.update_weights(-dw);
        auto context2 = rbm.get_context(state);
        std::complex<double> psi2 = rbm.psi(state, context2);

        EXPECT_NEAR(std::abs(psi1 - psi2) / rbm.get_n_params(), 0, 1e-15);
        // std::cout << psi1 << ", " << psi2 << ": " << std::abs(psi1 - psi2)
        //           << std::endl;
        context = context2;

        size_t f = f_dist(rng);
        rbm.update_context(state, {f}, context);
        state(f) *= -1;
    }
}

TEST(rbm_base, update_context) { test_update<rbm_base>(); }
TEST(rbm_symmetry, update_context) { test_update<rbm_symmetry>(); }

TEST(rbm_base, test_derivative) { test_derivative<rbm_base>(); }
TEST(rbm_symmetry, test_derivative) { test_derivative<rbm_symmetry>(); }

TEST(pfaffian, update_context) {
    std::mt19937 rng(SEED);
    lattice::honeycomb lat{5};
    pfaffian pfaff{lat};
    pfaff.init_weights(rng, 0.1);
    std::uniform_int_distribution<size_t> f_dist(0, lat.n_total - 1);
    std::uniform_int_distribution<size_t> u_dist(0, 1);
    Eigen::MatrixXcd state = Eigen::MatrixXd::Ones(lat.n_total, 1);
    for (size_t i = 0; i < lat.n_total; i++)
        if (u_dist(rng)) state(i) *= -1;
    auto context = pfaff.get_context(state);

    for (size_t i = 0; i < 100; i++) {
        std::vector<size_t> flips;
        flips.push_back(f_dist(rng));
        size_t second = f_dist(rng);
        while (second == flips[0]) {
            second = f_dist(rng);
        }
        flips.push_back(second);

        pfaff.update_context(state, flips, context);

        for (auto& f : flips) state(f) *= -1;
    }

    auto context2 = pfaff.get_context(state);

    double diff = (context.inv - context2.inv).array().abs().mean();
    double pfdiff =
        std::abs(context.pfaff -
                 context2.pfaff * std::pow(10, context2.exp - context.exp));

    EXPECT_NEAR(diff, 0, 1e-10);
    EXPECT_NEAR(pfdiff * std::pow(10, context.exp), 0, 1e-12);
}

TEST(pfaffian, test_derivative) {
    std::mt19937 rng(SEED);
    lattice::honeycomb lat{5};
    pfaffian pfaff{lat};
    pfaff.init_weights(rng, 0.1);
    std::uniform_int_distribution<size_t> f_dist(0, lat.n_total - 1);
    std::uniform_int_distribution<size_t> u_dist(0, 1);
    Eigen::MatrixXcd state = Eigen::MatrixXd::Ones(lat.n_total, 1);
    for (size_t i = 0; i < lat.n_total; i++)
        if (u_dist(rng)) state(i) *= -1;
    auto context = pfaff.get_context(state);

    std::normal_distribution<double> n_dist;

    for (size_t i = 0; i < 100; i++) {
        Eigen::MatrixXcd dw(pfaff.get_n_params(), 1);
        for (int j = 0; j < dw.size(); j++) {
            dw(j) = std::complex<double>(n_dist(rng), n_dist(rng));
        }

        dw *= 1e-8;

        Eigen::MatrixXcd dpfaff(dw.size(), 1);
        size_t of = 0;
        pfaff.derivative(state, context, dpfaff, of);

        std::complex<double> psi1 = pfaff.psi(state, context);
        psi1 += psi1 * (dw.transpose() * dpfaff)(0);
        of = 0;
        pfaff.update_weights(-dw, of);
        auto context2 = pfaff.get_context(state);
        std::complex<double> psi2 = pfaff.psi(state, context2);

        EXPECT_NEAR(std::abs(psi1 - psi2) / pfaff.get_n_params(), 0, 1e-14);
        context = context2;

        size_t f = f_dist(rng);
        pfaff.update_context(state, {f}, context);
        state(f) *= -1;
    }
}

#undef SETUP
