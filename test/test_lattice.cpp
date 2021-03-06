#include <gtest/gtest.h>

#include <iostream>
#include <vector>
//
#include <lattice/bravais.hpp>
#include <lattice/honeycomb.hpp>

using namespace lattice;

inline void test_nns(const std::vector<size_t>& nns_a,
                     const std::vector<size_t>& nns_b,
                     const size_t n_coordination) {
    for (size_t i = 0; i < n_coordination; i++) {
        EXPECT_EQ(nns_a[i], nns_b[i]);
    }
}

class bravais_test : public bravais {
   public:
    bravais_test(size_t n_uc, size_t n_uc_b = 0, size_t h_shift = 0,
                 size_t n_basis = 1)
        : bravais{n_uc, 2, n_basis, 0, n_uc_b, h_shift} {}

    void construct_bonds() override {}

    void print_lattice(const std::vector<size_t>& el) const override {
        for (size_t i = n_uc - 1; i < n_uc; i--) {
            for (size_t j = 0; j < n_uc_b; j++) {
                size_t c = 0;
                for (auto& e : el) {
                    if (uc_idx(e) == i * n_uc_b + j) c++;
                }
                if (c == 0) {
                    std::cout << ".";
                } else {
                    std::cout << c;
                }
                std::cout << " ";
            }
            std::cout << std::endl;
        }
    }

    std::vector<size_t> nns(size_t idx) const override { return {}; }

    std::vector<Eigen::PermutationMatrix<Eigen::Dynamic>> construct_symmetry(
        const std::vector<double>& symm) const override {
        return {};
    }
};

TEST(bravais_lattice, go_up) {
    const int N = 5;
    bravais_test lat{N};
    Eigen::MatrixXi mat(N, N);
    for (int i = 0; i < N * N; i++) {
        mat(i) = i;
    }

    for (int s = 1; s < N; s++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                EXPECT_EQ(lat.up(mat(i, j), 0, s), (size_t)mat((i + s) % N, j));
            }
    for (int s = 1; s < N; s++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                EXPECT_EQ(lat.up(mat(i, j), 1, s), (size_t)mat(i, (j + s) % N));
            }
}

TEST(bravais_lattice, go_down) {
    const int N = 5;
    bravais_test lat{N};
    Eigen::MatrixXi mat(N, N);
    for (int i = 0; i < N * N; i++) {
        mat(i) = i;
    }

    for (int s = 1; s < N; s++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                EXPECT_EQ(lat.down(mat(i, j), 0, s),
                          (size_t)mat((i - s + N) % N, j));
    for (int s = 1; s < N; s++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                EXPECT_EQ(lat.down(mat(i, j), 1, s),
                          (size_t)mat(i, (j - s + N) % N));
}

TEST(bravais_lattice, go_up_with_shift) {
    const int N = 5, sh = 2;
    bravais_test lat{N, 0, sh};
    Eigen::MatrixXi mat(N, N);
    for (int i = 0; i < N * N; i++) {
        mat(i) = i;
    }
    for (int s = 1; s < N; s++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                EXPECT_EQ(lat.up(mat(i, j), 0, s), (size_t)mat((i + s) % N, j));
            }
    for (int s = 1; s < N; s++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                size_t inew = (N + i - ((j + s) >= N ? sh : 0)) % N;
                EXPECT_EQ(lat.up(mat(i, j), 1, s),
                          (size_t)mat(inew, (j + s) % N));
            }
}

TEST(bravais_lattice, go_down_with_shift) {
    const int N = 5, sh = 2;
    bravais_test lat{N, 0, sh};
    Eigen::MatrixXi mat(N, N);
    for (int i = 0; i < N * N; i++) {
        mat(i) = i;
    }
    for (int s = 1; s < N; s++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                EXPECT_EQ(lat.down(mat(i, j), 0, s),
                          (size_t)mat((i - s + N) % N, j));
            }
    for (int s = 1; s < N; s++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                size_t inew = (i + ((j - s) < 0 ? sh : 0)) % N;
                EXPECT_EQ(lat.down(mat(i, j), 1, s),
                          (size_t)mat(inew, (j - s + N) % N));
            }
}

TEST(bravais_lattice, go_up_with_2nd_dim) {
    const int N = 5, Nb = 3;
    bravais_test lat{N, Nb};
    Eigen::MatrixXi mat(Nb, N);
    for (int i = 0; i < N * Nb; i++) {
        mat(i) = i;
    }

    for (int s = 1; s < Nb; s++)
        for (int i = 0; i < Nb; i++)
            for (int j = 0; j < N; j++) {
                EXPECT_EQ(lat.up(mat(i, j), 0, s),
                          (size_t)mat((i + s) % Nb, j));
            }
    for (int s = 1; s < N; s++)
        for (int i = 0; i < Nb; i++)
            for (int j = 0; j < N; j++) {
                EXPECT_EQ(lat.up(mat(i, j), 1, s), (size_t)mat(i, (j + s) % N));
            }
}

TEST(bravais_lattice, go_down_with_2nd_dim) {
    const int N = 5, Nb = 3;
    bravais_test lat{N, Nb};
    Eigen::MatrixXi mat(Nb, N);
    for (int i = 0; i < N * Nb; i++) {
        mat(i) = i;
    }

    for (int s = 1; s < Nb; s++)
        for (int i = 0; i < Nb; i++)
            for (int j = 0; j < N; j++) {
                EXPECT_EQ(lat.down(mat(i, j), 0, s),
                          (size_t)mat((i - s + Nb) % Nb, j));
            }
    for (int s = 1; s < N; s++)
        for (int i = 0; i < Nb; i++)
            for (int j = 0; j < N; j++) {
                EXPECT_EQ(lat.down(mat(i, j), 1, s),
                          (size_t)mat(i, (j - s + N) % N));
            }
}

TEST(bravais_lattice, go_up_with_shift_and_2nd_dim) {
    const int N = 5, Nb = 3, sh = 2;
    bravais_test lat{N, Nb, sh};
    Eigen::MatrixXi mat(Nb, N);
    for (int i = 0; i < N * Nb; i++) {
        mat(i) = i;
    }

    for (int s = 1; s < Nb; s++)
        for (int i = 0; i < Nb; i++)
            for (int j = 0; j < N; j++) {
                EXPECT_EQ(lat.up(mat(i, j), 0, s),
                          (size_t)mat((i + s) % Nb, j));
            }
    for (int s = 1; s < N; s++)
        for (int i = 0; i < Nb; i++)
            for (int j = 0; j < N; j++) {
                size_t inew = (Nb + i - ((j + s) >= N ? sh : 0)) % Nb;
                EXPECT_EQ(lat.up(mat(i, j), 1, s),
                          (size_t)mat(inew, (j + s) % N));
            }
}

TEST(bravais_lattice, go_down_with_shift_and_2nd_dim) {
    const int N = 5, Nb = 3, sh = 2;
    bravais_test lat{N, Nb, sh};
    Eigen::MatrixXi mat(Nb, N);
    for (int i = 0; i < N * Nb; i++) {
        mat(i) = i;
    }

    for (int s = 1; s < Nb; s++)
        for (int i = 0; i < Nb; i++)
            for (int j = 0; j < N; j++) {
                EXPECT_EQ(lat.down(mat(i, j), 0, s),
                          (size_t)mat((i - s + Nb) % Nb, j));
            }
    for (int s = 1; s < N; s++)
        for (int i = 0; i < Nb; i++)
            for (int j = 0; j < N; j++) {
                size_t inew = (i + ((j - s) < 0 ? sh : 0)) % Nb;
                EXPECT_EQ(lat.down(mat(i, j), 1, s),
                          (size_t)mat(inew, (j - s + N) % N));
            }
}

TEST(bravais_lattice, uc_idx_conversions) {
    const size_t N = 5, Nb = 3, B = 3;
    bravais_test lat{N, Nb, 0, B};

    EXPECT_EQ(lat.idx(lat.up(0), 0), B);
    EXPECT_EQ(lat.idx(lat.up(0), 1), B + 1);

    EXPECT_EQ(lat.b_idx(7 * B + 2), 2U);
    EXPECT_EQ(lat.b_idx(9 * B + 1), 1U);

    EXPECT_EQ(lat.uc_idx(7 * B + 2), 7U);
    EXPECT_EQ(lat.uc_idx(9 * B + 1), 9U);

    EXPECT_EQ(lat.uc_idx({1, 1}), Nb + 1);
    EXPECT_EQ(lat.uc_idx({2, 0}), 2 * Nb);
}

TEST(honeycomb_lattice, nearest_neighbours) {
    honeycomb lat{3};
    auto nns_at8 = lat.nns(8);
    const std::vector<size_t> nns_at8_should = {9, 7, 3};
    test_nns(nns_at8, nns_at8_should, lat.n_coordination);

    auto nns_at9 = lat.nns(9);
    const std::vector<size_t> nns_at9_should = {8, 10, 14};
    test_nns(nns_at9, nns_at9_should, lat.n_coordination);

    auto nns_at0 = lat.nns(0);
    const std::vector<size_t> nns_at0_should = {1, 5, 13};
    test_nns(nns_at0, nns_at0_should, lat.n_coordination);

    lattice::honeycomb lat2{5};
    auto nns_at17 = lat.nns(17);
    const std::vector<size_t> nns_at17_should = {16, 18, 26};
}
