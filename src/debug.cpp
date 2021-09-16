#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>
//
#include <lattice/honeycomb.hpp>
#include <machine/file_psi.hpp>
#include <math.hpp>
#include <model/abstract_model.hpp>
#include <model/isingS3.hpp>
#include <model/kitaev.hpp>
#include <operators/aggregator.hpp>
#include <optimizer/minres_adapter.hpp>
#include <sampler/full_sampler.hpp>
#include <sampler/metropolis_sampler.hpp>
#include <tools/ini.hpp>

using namespace Eigen;

std::vector<size_t> to_indices(const MatrixXcd& vec) {
    std::vector<size_t> ret;
    for (size_t v = 0; v < (size_t)vec.size(); v++) {
        if (std::real(vec(v)) > 0) ret.push_back(v);
    }
    return ret;
}
size_t to_idx(const MatrixXcd& vec) {
    for (size_t v = 0; v < (size_t)vec.size(); v++) {
        if (std::real(vec(v)) > 0) return v;
    }
    return -1;
}
void test_symmetry() {
    model::kitaev km{4, -1};
    auto& lattice = km.get_lattice();
    MatrixXcd vec(lattice.n_total, 1);
    vec.setConstant(-1);
    vec(0) = 1;
    vec(1) = 1;
    vec(2) = 1;
    vec(8) = 1;

    auto symm = lattice.construct_symmetry();
    for (auto& s : symm) {
        lattice.print_lattice(to_indices(s * vec));
    }
}

void print_bonds() {
    lattice::honeycomb lat{2, 3};
    // lat.print_lattice({});
    auto bonds = lat.get_bonds();
    for (auto& bond : bonds) {
        std::cout << bond.a << "," << bond.b << "," << bond.type << std::endl;
    }
}

void test_S3() {
    model::isingS3 km{3, -1};
    machine::file_psi m{km.get_lattice(), "isingS3.state"};
    sampler::full_sampler sampler{m, 3};
    auto& h = km.get_hamiltonian();
    operators::aggregator agg{h};
    sampler.register_op(&h);
    sampler.register_agg(&agg);
    sampler.sample(false);
    std::cout.precision(17);
    std::cout << agg.get_result() << std::endl;
    std::cout << agg.get_result() / km.get_lattice().n_total << std::endl;
}

void debug() {
    //
    size_t n_chains = 16;
    size_t step_size = 5;
    size_t warmup_steps = 100;
    size_t n_samples = 1000;
    double bond_flips = 0.5;

    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};
    std::cout.precision(17);

    model::kitaev m{3, -1};
    machine::file_psi rbm{m.get_lattice(), "notebooks/n3.state"};
    sampler::full_sampler sampler{rbm, 3};
    sampler::metropolis_sampler msampler{
        rbm, n_samples, rng, n_chains, step_size, warmup_steps, bond_flips};
    operators::aggregator agg{m.get_hamiltonian()};
    agg.track_variance();
    sampler.register_op(&(m.get_hamiltonian()));
    sampler.register_agg(&agg);
    msampler.register_op(&(m.get_hamiltonian()));
    msampler.register_agg(&agg);

    for (size_t i = 0; i < 10; i++) {
        msampler.sample();
        std::cout << "Metropolis Sampler: " << agg.get_result() / rbm.n_visible
                  << " += " << agg.get_variance() / rbm.n_visible << std::endl;
        std::cout << msampler.get_acceptance_rate() << std::endl;
    }
    sampler.sample(false);
    std::cout << "Full Sampler: " << agg.get_result() / rbm.n_visible
              << " += " << agg.get_variance() / rbm.n_visible << std::endl;
}

Eigen::SparseMatrix<std::complex<double>> kron(
    const std::vector<Eigen::SparseMatrix<std::complex<double>>>& args) {
    Eigen::SparseMatrix<std::complex<double>> so(1, 1);
    so.insert(0, 0) = 1;
    for (const auto& arg : args) so = kroneckerProduct(arg, so).eval();
    return so;
}

void debug1() {
    using SparseXcd = Eigen::SparseMatrix<std::complex<double>>;
    std::cout << "DEBUG 1" << std::endl;
    SparseXcd sx(2, 2), sy(2, 2), sz(2, 2);
    sx.insert(0, 1) = 1;
    sx.insert(1, 0) = 1;
    sy.insert(0, 1) = std::complex<double>(0, -1);
    sy.insert(1, 0) = std::complex<double>(0, 1);
    sz.insert(0, 0) = 1;
    sz.insert(1, 1) = -1;

    SparseXcd x_yz = kron({sy, sx});
    SparseXcd x_zy = kron({sx, sy});

    SparseXcd y_xz = kron({-sy, sy});
    SparseXcd y_zx = kron({sy, -sy});

    SparseXcd z_xy = kron({sx, sx});
    SparseXcd z_yx = kron({sx, sx});
}

void debug_pfaffian() {
    lattice::honeycomb lat{2};
    machine::pfaffian pfaff{lat};
    Eigen::MatrixXcd state = Eigen::MatrixXd::Random(lat.n_total, 1);
    state.array() /= state.array().abs();

    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};
    pfaff.init_weights(rng, 0.1, false);

    auto context = pfaff.get_context(state);

    std::vector<size_t> flips;

    std::uniform_int_distribution<size_t> f_dist(0, lat.n_total - 1);
    Eigen::ArrayXd arr(pfaff.get_n_params(), 1);
    arr.setZero();
    for (size_t x = 0; x < 1000; x++) {
        flips.clear();
        for (size_t i = 0; i < 2; i++) {
            size_t r = f_dist(rng);
            if (std::find(flips.begin(), flips.end(), r) == flips.end()) {
                flips.push_back(r);
            }
        }
        pfaff.update_context(state, flips, context);
        for (auto& f : flips) state(f) *= -1;
        Eigen::MatrixXcd mat(pfaff.get_n_params(), 1);
        size_t o = 0;
        pfaff.derivative(state, context, mat, o);
        std::cout << (mat.array().abs() < 1e-10).cast<int>().sum() << std::endl;
        arr += (mat.array().abs() < 1e-10).cast<double>();
    }
    arr /= 1000.0;
    std::cout << arr.transpose() << std::endl;
    double mean = arr.mean();
    double stddev = std::sqrt(arr.square().mean() - std::pow(mean, 2));

    std::cout << mean << ", " << stddev << std::endl;
    std::cout << context.pfaff << " x10^" << context.exp << std::endl;

    auto context2 = pfaff.get_context(state);
    std::cout << context2.pfaff << " x10^" << context2.exp << std::endl;
    // Eigen::MatrixXcd mat = pfaff.get_mat(state).inverse();
    // std::cout << (context.inv - mat).norm() / mat.size() << std::endl;
}

void debug_pfaffian2() {
    lattice::honeycomb lat{2};
    machine::pfaffian pfaff{lat};
    Eigen::MatrixXcd state = Eigen::MatrixXd::Random(lat.n_total, 1);
    state.array() /= state.array().abs();

    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};
    pfaff.init_weights(rng, 0.1, false);

    auto context = pfaff.get_context(state);

    Eigen::MatrixXcd derivative(pfaff.get_n_params(), 1);
    size_t o = 0;
    pfaff.derivative(state, context, derivative, o);

    Eigen::MatrixXcd upd = Eigen::MatrixXcd::Random(pfaff.get_n_params(), 1);
    upd *= 1e-6;

    std::complex<double> pf1 = pfaff.psi(state, context);
    pf1 += pf1 * (derivative.array() * upd.array()).sum();
    std::cout.precision(17);
    std::cout << pf1 << std::endl;

    o = 0;
    pfaff.update_weights(upd, o);
    auto context2 = pfaff.get_context(state);
    std::complex<double> pf2 = pfaff.psi(state, context2);
    std::cout << pf2 << std::endl;

    std::cout << std::abs(pf1 - pf2) << std::endl;
}

void test_minresqlp() {
    int na = 500, nb = 500, nn = 50;
    Eigen::MatrixXcd mat(na, nb);
    double norm = 0.1;

    double e1 = 1;
    double de = 1;
    double e2 = 1;
    mat.setRandom();

    Eigen::MatrixXcd vec;
    vec = mat.rowwise().sum().conjugate() / std::sqrt(norm);

    MatrixXcd S(na, na);
    S = mat.conjugate() * mat.transpose() / norm;
    S -= vec * vec.transpose().conjugate();

    /* MatrixXcd d = mat.cwiseAbs2().rowwise().sum() / norm - vec.cwiseAbs2();
    std::cout << (S.diagonal() - d).norm() << std::endl; */

    double maxDiag = S.diagonal().real().maxCoeff();
    S.diagonal().topRows(nn) *= (1 + e1);
    S.diagonal().bottomRows(na - nn) *= (1 + e1 + de);
    S += e2 * maxDiag * Eigen::MatrixXcd::Identity(na, na);
    // S += e2 * Eigen::MatrixXcd::Identity(na, na);

    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> x(na), y(na);
    Eigen::VectorXcd z(na);
    x.setRandom();
    y.setZero();
    z.setZero();

    Eigen::VectorXcd tmp(nb, 1);
    Eigen::VectorXcd diag(na, 1);
    diag = mat.cwiseAbs2().rowwise().sum() / norm - vec.cwiseAbs2();
    optimizer::minresqlp_adapter min{mat, vec, e1, e2, de, norm, nn, diag, tmp};

    min.itnlim = 1000;
    std::cout << "start" << std::endl;
    std::cout << min.apply(x, z) << std::endl;
    std::cout << min.getItn() << std::endl;
    std::cout << min.getAcond() << std::endl;
    std::cout << min.getRnorm() << std::endl;

    y = S * z;

    std::cout << (x - y).norm() << std::endl;
}

void debug_general_pfaffprocedure() {
    size_t n = 1000, m = 10;
    MatrixXcd A(n, n);
    A.setRandom();
    A -= A.transpose().eval();
    A /= 2;

    MatrixXcd inv = A.inverse();
    inv -= inv.transpose().eval();
    inv /= 2;

    double diff = (MatrixXcd::Identity(n, n) - inv * A).array().abs().mean();
    std::cout << diff << std::endl;

    MatrixXcd Acopy = A;
    int expA;
    std::complex<double> pfaffA = math::pfaffian10(Acopy, expA);

    MatrixXcd B(n, m), C(m, m);
    B.setRandom();
    C.setRandom();
    C -= C.transpose().eval();
    C /= 2;
    MatrixXcd Cinv = C.inverse();
    Cinv -= Cinv.transpose().eval();
    Cinv /= 2;

    MatrixXcd BCB = B * C * B.transpose();
    MatrixXcd BinvB = B.transpose() * inv * B;

    MatrixXcd ABCB = A + BCB;
    MatrixXcd CinvBinvB = Cinv + BinvB;

    MatrixXcd ABCBcopy = ABCB;
    int expABCB;
    std::complex<double> pfaffABCB = math::pfaffian10(ABCBcopy, expABCB);

    MatrixXcd CinvBinvBcopy = CinvBinvB;
    int expCinvBinvB;
    std::complex<double> pfaffCinvBinvB =
        math::pfaffian10(CinvBinvBcopy, expCinvBinvB);

    MatrixXcd Cinvcopy = Cinv;
    int expCinv;
    std::complex<double> pfaffCinv = math::pfaffian10(Cinvcopy, expCinv);

    std::complex<double> pfaffABCB2 = pfaffA * pfaffCinvBinvB / pfaffCinv;

    int expABCB2 = expA + expCinvBinvB - expCinv;

    std::cout << pfaffABCB << " x10^" << expABCB << std::endl;
    std::cout << pfaffABCB2 << " x10^" << expABCB2 << std::endl;
    std::cout << pfaffA << " x10^" << expA << std::endl;
    std::cout << pfaffCinvBinvB << std::endl;
    std::cout << pfaffCinv << std::endl;
    std::cout << std::abs(pfaffABCB2 -
                          pfaffABCB * std::pow(10, expABCB - expABCB2))
              << std::endl;

    MatrixXcd ABCBinv = ABCB.inverse();
    ABCBinv -= ABCBinv.transpose().eval();
    ABCBinv /= 2;
    MatrixXcd ABCBinv2 = inv * B * CinvBinvB.inverse() * B.transpose() * inv;
    ABCBinv2 -= ABCBinv2.transpose().eval();
    ABCBinv2 /= 2;
    ABCBinv2 -= inv;

    std::cout << (ABCBinv + ABCBinv2).array().abs().mean() << std::endl;
}

void debugAprod() {
    int n = 100, m = 20;
    Eigen::MatrixXcd mat(n, m);
    mat.setRandom();
    Eigen::MatrixXcd S = mat.conjugate() * mat.transpose();
    Eigen::MatrixXcd x(n, 1);
    x.setRandom();
    Eigen::MatrixXcd y1(n, 1);
    Eigen::MatrixXcd y2(m, 1);
    Eigen::MatrixXcd tmp(m, 1);
    Eigen::MatrixXcd tmp2(m, 1);
    Eigen::MatrixXcd vec = Eigen::MatrixXcd::Zero(n, 1);
    std::complex<double> dot;
    Eigen::MatrixXcd diag = S.diagonal();

    double norm = 1.;
    double reg[] = {0., 0.};

    optimizer::g_mat = mat.data();
    optimizer::g_vec = vec.data();
    optimizer::g_tmp = tmp.data();
    optimizer::g_dot = &dot;
    optimizer::g_diag = diag.data();

    optimizer::g_norm = norm;
    optimizer::g_reg = &reg[0];

    optimizer::g_mat_dim2 = m;

    optimizer::Aprod(&n, x.data(), y1.data());
    tmp2 = mat.transpose() * x;
    y2 = S * x;
    Eigen::MatrixXcd r1(n, 1);
    Eigen::MatrixXcd r2(n, 1);
    optimizer::Aprod(&n, y1.data(), r1.data());
    r2 = S * y2;

    std::cout << y1.squaredNorm() << std::endl;
    std::cout << r1.adjoint() * x << std::endl;

    // std::cout << (y1 - y2).cwiseAbs2().mean() << std::endl;
    // std::cout << (tmp2 - tmp).cwiseAbs2().mean() << std::endl;
}
