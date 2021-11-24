#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>
//
#include <lattice/honeycomb.hpp>
#include <machine/file_psi.hpp>
#include <machine/rbm_base.hpp>
#include <machine/spin_state.hpp>
#include <math.hpp>
#include <model/abstract_model.hpp>
#include <model/isingS3.hpp>
#include <model/kitaev.hpp>
#include <operators/aggregator.hpp>
#include <optimizer/minresqlp_adapter.hpp>
#include <sampler/full_sampler.hpp>
#include <sampler/metropolis_sampler.hpp>
#include <tools/ini.hpp>
#include <tools/mpi.hpp>

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
    operators::aggregator agg{h, sampler.get_my_n_samples()};
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
    operators::aggregator agg{m.get_hamiltonian(), sampler.get_my_n_samples()};
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

    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};
    machine::spin_state state(lat.n_total);
    state.set_random(rng);
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
        state.flip(flips);
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

    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};
    machine::spin_state state(lat.n_total);
    state.set_random(rng);
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

class mpi_op : public operators::base_op {
    int c;
    std::vector<double> data = {
        0.977032588723069,    0.042131676446022226, 0.522972009002048,
        0.9421396560875159,   0.6596102035244401,   0.19811722085692818,
        0.1946833881000004,   0.8062301027565533,   0.6562266325318439,
        0.48292217432207507,  0.7721003971742948,   0.6185413881658379,
        0.5921020910720973,   0.30968682631762035,  0.811837887251313,
        0.6541725158631927,   0.7147620088231902,   0.4647773268829073,
        0.28871355708785296,  0.16100667185583595,  0.22367624027619448,
        0.9260344138408682,   0.11971591939205839,  0.5485936384659809,
        0.24020063756596544,  0.9951943223262818,   0.11871793538892472,
        0.32819109031512095,  0.061712317868371724, 0.10710638346252987,
        0.38592808271501655,  0.47642536016437664,  0.9544768504764556,
        0.11149792063334596,  0.48076428732587784,  0.728971578762767,
        0.6755948519482451,   0.13594778958574438,  0.9157750018193435,
        0.803026551955031,    0.04552700338411331,  0.4549025242241056,
        0.25592729780108936,  0.22033681586531284,  0.5195658825402238,
        0.5497213952170441,   0.42090051466777356,  0.7916682122492474,
        0.9174124189479687,   0.7586731879556448,   0.15978385054534705,
        0.8118136128697562,   0.2638834080579403,   0.6069316133322737,
        0.491747117083838,    0.117078225042749,    0.9917580773865583,
        0.8698788302327972,   0.8251456209145848,   0.9598906838075068,
        0.44940365307257446,  0.3155972426486876,   0.3782726410028242,
        0.961742356202122,    0.7925214329259596,   0.21307731725958945,
        0.3808289524565299,   0.2913315607594972,   0.3950034867105411,
        0.14472427506489172,  0.8592815554444933,   0.1279657613256472,
        0.5899733772656922,   0.07756903222525457,  0.6575053800445048,
        0.391748644113212,    0.12309910967999027,  0.6558721866060615,
        0.038962162027456504, 0.6739260665406597,   0.49481251179803243,
        0.6777140352879767,   0.5681041828859267,   0.2557817404081909,
        0.11198020897180305,  0.6663278185460747,   0.4670812764966634,
        0.6005785862735303,   0.06722373694535444,  0.04205412612252779,
        0.3864607903849878,   0.030456075790041837, 0.13415789484902774,
        0.5002084338197285,   0.7433096579216042,   0.34077171216312485,
        0.36287913750323175,  0.29221880899062425,  0.19630683936038662,
        0.5406507921681055};

    std::vector<double> ps = {
        0.010875994302703739,  0.010594101628117553,   0.01283932787743003,
        0.0012179711727699453, 0.013614815770084478,   0.015374538681990011,
        0.006336413164670813,  0.01482163500000825,    0.015134265262609215,
        0.010249964996904878,  0.006840953286031941,   0.01690020902195116,
        0.0009470325045802447, 0.0003382585619616578,  0.008548569561994079,
        0.015092702985082408,  0.011472397587385837,   0.008445139007449006,
        0.003740274628480024,  0.003510884282858516,   0.009866532736965814,
        0.01776655181566469,   0.007939448767547683,   0.0018341151952600164,
        0.015310232235090447,  0.017486891697752516,   0.005311548549953812,
        0.014887949024443384,  0.004738711168914282,   0.009540912158783504,
        0.0063935012181789595, 0.010026265036176221,   0.011907295024053324,
        0.014122224106474948,  0.002881702206865988,   0.015412830209991379,
        0.010103981423178898,  0.014456791635032667,   0.005524526117380507,
        0.01705268049923486,   0.0041921026599170555,  0.016948576523323267,
        0.0006388486561331785, 0.010553286475190559,   0.0007133892106498745,
        0.0024233318590732977, 0.014850242737347993,   0.010990337068969767,
        0.017608218685683606,  0.006611126519572497,   0.01572371674353509,
        0.007633695625722481,  0.008784182331385141,   0.013172371756985176,
        0.009260310413545008,  0.009118474614461693,   0.003999255025992542,
        0.00805871916849993,   0.012060532966087953,   0.01093257082704249,
        0.011308052252542911,  0.011724261910912723,   0.015931347506601745,
        0.00481886983917667,   0.011137519648462487,   0.013433209253497114,
        0.0012171372798373784, 0.00020402085464400697, 0.015870477924163663,
        0.009256690167315111,  0.015759927458577648,   0.014287694295222688,
        0.016992607644848747,  0.0023881885544309945,  0.016808585390551835,
        0.010063572870545098,  0.014435235821481249,   0.008567994810313975,
        0.0052671965996169325, 0.015281060773776483,   0.006216762350864477,
        0.015781243511141633,  0.015503892390919654,   0.010393259446667769,
        0.002353621311222111,  0.0177087318336849,     0.012144367530965447,
        0.0022615160563713814, 0.017751977083877667,   0.004504770971105944,
        0.006437691509640454,  0.011857804520541328,   0.016556937238563635,
        0.005745936664732923,  0.011022659936585391,   0.0011965854829530036,
        0.014565042517693364,  0.014440621714829032,   0.010885847982597011,
        0.004183649107397657};

   public:
    mpi_op() : base_op() { c = mpi::rank; }

    void evaluate(machine::abstract_machine& rbm,
                  const machine::spin_state& state,
                  machine::rbm_context& context) override {
        auto& r = get_result_();
        r(0) = {data[c], 50 * ps[c]};
        c += mpi::n_proc;
    }

    double get_p() { return ps[c - mpi::n_proc]; }
};

void debug_mpi() {
    mpi::init(0, nullptr);
    model::kitaev k(2, -1);
    machine::rbm_base m(1, k.get_lattice());
    machine::spin_state s(m.n_visible);
    auto c = m.get_context(s);
    for (int j = 0; j < 1; j++) {
        std::cout.precision(10);
        mpi_op op;
        int pchs = 100 / mpi::n_proc;
        operators::aggregator agg(op, pchs);
        agg.track_variance();
        agg.set_zero();
        for (int i = 0; i < pchs; i++) {
            op.evaluate(m, s, c);
            agg.aggregate();
        }
        agg.finalize(100);
        std::cout.precision(17);
        mpi::cout << agg.get_result() << mpi::endl;
        mpi::cout << agg.get_variance(true) << mpi::endl;
    }
    mpi::end();
}
