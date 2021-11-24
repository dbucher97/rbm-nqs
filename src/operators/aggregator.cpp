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

#include <Eigen/Dense>
#include <stdexcept>
//
#include <operators/aggregator.hpp>
#include <tools/mpi.hpp>

#define INCLUDE_DEFAULT

using namespace operators;

aggregator::aggregator(const base_op& op, size_t samples, size_t r, size_t c)
    : n_samples_{samples},
      result_(r, c),
#ifdef INCLUDE_DEFAULT
      resultx_(r, c),
#endif
      op_{op} {
    // Initialize result as zero.
    set_zero();
}

aggregator::aggregator(const base_op& op, size_t samples)
    : aggregator{op, samples, op.rows(), op.cols()} {}

void aggregator::set_zero() {
    result_.setZero();
#ifdef INCLUDE_DEFAULT
    resultx_.setZero();
#endif
    wsum_ = 0;
    wsum2_ = 0;
    cur_n_ = 0;
    if (track_variance_) {
        variance_.setZero();
#ifdef INCLUDE_DEFAULT
        variancex_.setZero();
#endif
        bin_.setZero();
        variance_binned_.setZero();
        result_binned_.setZero();
        cur_n_bin_ = 0;
        wsum_bin_ = 0;
    }
}

void aggregator::track_variance(size_t n_bins) {
    if (n_bins > 1) binning_ = true;
    track_variance_ = true;
    n_bins_ = n_bins;
    if (binning_ && n_samples_ % n_bins != 0) {
        throw std::runtime_error("n_samples not divisable by n_bins!");
    }

    if (binning_) bin_size_ = n_samples_ / n_bins_;

    bin_ = Eigen::MatrixXcd(result_.rows(), result_.cols());
    result_binned_ = Eigen::MatrixXcd(result_.rows(), result_.cols());
    variance_ = Eigen::MatrixXd(result_.rows(), result_.cols());
#ifdef INCLUDE_DEFAULT
    variancex_ = Eigen::MatrixXd(result_.rows(), result_.cols());
#endif
    variance_binned_ = Eigen::MatrixXd(result_.rows(), result_.cols());
    tau_ = Eigen::MatrixXd(result_.rows(), result_.cols());
}

void aggregator::finalize(double ptotal) {
#ifdef INCLUDE_DEFAULT
    resultx_ /= ptotal;
    MPI_Allreduce(MPI_IN_PLACE, resultx_.data(), resultx_.size(),
                  MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#endif

    if (track_variance_) {
#ifdef INCLUDE_DEFAULT
        variancex_ /= ptotal;
        MPI_Allreduce(MPI_IN_PLACE, variancex_.data(), variancex_.size(),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        variancex_.array() -= resultx_.cwiseAbs2().array();
#endif

        Eigen::MatrixXcd results(result_.size(), mpi::n_proc);
        Eigen::VectorXd weights(mpi::n_proc);
        MPI_Allgather(result_.data(), result_.size(), MPI_DOUBLE_COMPLEX,
                      results.data(), result_.size(), MPI_DOUBLE_COMPLEX,
                      MPI_COMM_WORLD);
        MPI_Allgather(&wsum_, 1, MPI_DOUBLE, weights.data(), 1, MPI_DOUBLE,
                      MPI_COMM_WORLD);
        Eigen::MatrixXcd current = results.col(0);
        Eigen::MatrixXd corr = Eigen::MatrixXd::Zero(result_.size(), 1);
        double wsumx = weights(0);
        double wsumlast = weights(0);
        for (int i = 1; i < mpi::n_proc; i++) {
            wsumx += weights(i);
            Eigen::MatrixXcd delta = results.col(i) - current;
            current += delta * weights(i) / wsumx;
            corr += delta.cwiseAbs2() * wsumlast * weights(i) / wsumx;
            wsumlast = wsumx;
        }
        mpi::cout << current << mpi::endl;

        MPI_Allreduce(MPI_IN_PLACE, &wsum2_, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        if (binning_) {
            if (variance_.isZero() && variance_binned_.isZero()) {
                tau_.setConstant(0.5);
            } else {
                tau_ =
                    0.5 *
                    ((bin_size_ * variance_binned_.array() / variance_.array())
                         .cwiseMax(1) -
                     1);
            }

            // Average taus
            tau_ /= mpi::n_proc;
            MPI_Allreduce(MPI_IN_PLACE, tau_.data(), tau_.size(), MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD);

            MPI_Allreduce(MPI_IN_PLACE, variance_binned_.data(),
                          variance_binned_.size(), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            variance_binned_ += Eigen::Map<Eigen::MatrixXd>(
                corr.data(), variance_binned_.rows(), variance_binned_.cols());
            variance_binned_ /= ptotal - wsum2_ / ptotal;
        }

        MPI_Allreduce(MPI_IN_PLACE, variance_.data(), variance_.size(),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        variance_ += Eigen::Map<Eigen::MatrixXd>(corr.data(), variance_.rows(),
                                                 variance_.cols());

        variance_ /= (ptotal - wsum2_ / ptotal);

        sample_factor_ = ptotal / (ptotal - wsum2_ / ptotal);

#ifdef INCLUDE_DEFAULT
        variancex_ *= sample_factor_;
#endif
    }

    result_ *= wsum_ / ptotal;
    MPI_Allreduce(MPI_IN_PLACE, result_.data(), result_.size(),
                  MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
}

Eigen::MatrixXcd& aggregator::get_result() {
    // mpi::cout << "res_diff(" << result_.cols() << ", " << result_.rows() <<
    // ") "
    //           << (result_ - resultx_).norm() << mpi::endl;
    return result_;
}
Eigen::MatrixXd& aggregator::get_variance(bool sample_variance) {
    // mpi::cout << "var_diff(" << result_.cols() << ", " << result_.rows() <<
    // ") "
    //           << (variance_ - variancex_).norm() << mpi::endl;
    if (sample_variance) {
        mpi::cout << variancex_ << mpi::endl;
        return variance_;
    } else {
        variance_ /= sample_factor_;
        sample_factor_ = 1;
        return variance_;
    }
}

Eigen::MatrixXcd aggregator::aggregate_() {
    // By default, just forward the operator result.
    return op_.get_result();
}

void aggregator::aggregate(double weight) {
    cur_n_++;
    wsum_ += weight;
    wsum2_ += weight * weight;

    Eigen::MatrixXcd x = aggregate_();
    Eigen::MatrixXcd delta = weight * (x - result_);
    result_.noalias() += delta / wsum_;

#ifdef INCLUDE_DEFAULT
    resultx_ += weight * x;
#endif

    if (track_variance_) {
        cur_n_bin_++;
        // Calculate the resul of the squared observable

        variance_.array() +=
            ((x - result_).conjugate().array() * delta.array()).real();

#ifdef INCLUDE_DEFAULT
        variancex_.array() += x.cwiseAbs2().array() * weight;
#endif

        if (binning_) {
            wsum_bin_ += weight;
            bin_.noalias() += weight * x;

            if (cur_n_bin_ == bin_size_) {
                cur_n_bin_ = 0;
                bin_ /= wsum_bin_;
                delta = wsum_bin_ * (bin_ - result_binned_);
                result_binned_ += delta / wsum_;
                variance_binned_.array() += (delta.conjugate().array() *
                                             (bin_ - result_binned_).array())
                                                .real();
                bin_.setZero();
                wsum_bin_ = 0;
            }
        }
    }
}

Eigen::MatrixXd aggregator::get_stddev() const {
    if (binning_) {
        return (variance_binned_ / (mpi::n_proc * n_bins_)).array().sqrt();
    } else {
        return (variance_ / (mpi::n_proc * n_samples_)).array().sqrt();
    }
}

Eigen::MatrixXd aggregator::get_tau() const { return tau_; }

prod_aggregator::prod_aggregator(const base_op& op, const base_op& scalar,
                                 size_t samples)
    : Base{op, samples}, scalar_{scalar} {
    if (scalar_.rows() != 1 || scalar_.cols() != 1) {
        throw std::runtime_error("scalar operator must have size (1, 1).");
    }
}

Eigen::MatrixXcd prod_aggregator::aggregate_() {
    // Calculate op_a op_b^*
    return scalar_.get_result()(0) * op_.get_result().conjugate();
}

outer_aggregator::outer_aggregator(const base_op& op, size_t samples)
    : Base{op, samples, op.rows(), op.rows()} {
    // Guard operator to be vector.
    if (op_.cols() != 1) {
        throw std::runtime_error(
            "outer_aggregator needs column vector type operators");
    }
}

Eigen::MatrixXcd outer_aggregator::aggregate_() {
    // Calculate op^* op^T
    return op_.get_result().conjugate() * op_.get_result().transpose();
}

outer_aggregator_lazy::outer_aggregator_lazy(const base_op& op, size_t samples)
    : Base{op, samples, op.rows() * op.cols(), samples},
      diag_(op.rows() * op.cols()) {}

void outer_aggregator_lazy::aggregate(double weight) {
    result_.col(current_index_) =
        std::sqrt(weight) * Eigen::Map<const Eigen::MatrixXcd>(
                                op_.get_result().data(), result_.rows(), 1);
    // diag_.noalias() += result_.col(current_index_).cwiseAbs2();
    current_index_++;
}

Eigen::VectorXd& outer_aggregator_lazy::get_diag() { return diag_; }

void outer_aggregator_lazy::finalize(double val) {
    norm_ = val;
    diag_ /= norm_;
    MPI_Allreduce(MPI_IN_PLACE, diag_.data(), diag_.size(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
}

void outer_aggregator_lazy::finalize_diag(const Eigen::MatrixXcd& v) {
    diag_ -= v.cwiseAbs2();
}

void outer_aggregator_lazy::set_zero() {
    // Base::set_zero();
    current_index_ = 0;
    norm_ = 0;
    diag_.setZero();
}

optimizer::OuterMatrix outer_aggregator_lazy::construct_outer_matrix(
    aggregator& derivative, double reg1, double reg2) {
    // diag_ = result_.cwiseAbs2().rowwise().sum() / norm_;
    return {result_, derivative.get_result(), diag_, norm_, reg1, reg2};
}
