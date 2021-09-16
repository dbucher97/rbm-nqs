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

using namespace operators;

aggregator::aggregator(const base_op& op, size_t r, size_t c)
    : result_(r, c), variance_(r, c), op_{op} {
    // Initialize result as zero.
    set_zero();
}

aggregator::aggregator(const base_op& op)
    : aggregator{op, op.rows(), op.cols()} {}

void aggregator::set_zero() {
    result_.setZero();
    if (track_variance_) {
        variance_.setZero();
    }
}

void aggregator::finalize(double num) {
    // Normalize result

    result_ /= num;
    // std::complex<double> r1 = result_(0);

    MPI_Allreduce(MPI_IN_PLACE, result_.data(), result_.size(),
                  MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

    // if (result_.size() == 1) {
    //     // mpi::cout << mpi::endl;
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if (std::abs((double)mpi::n_proc * r1 - result_(0)) > 1) {
    //         throw std::runtime_error("XXXXX");
    //     }
    // }

    if (track_variance_) {
        variance_ /= num;
        // Calculate variance <0^2> - <0>^2

        MPI_Allreduce(MPI_IN_PLACE, variance_.data(), variance_.size(),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        variance_ -= result_.real().cwiseAbs2();
    }
}

Eigen::MatrixXcd& aggregator::get_result() { return result_; }
Eigen::MatrixXd& aggregator::get_variance() { return variance_; }

Eigen::MatrixXcd aggregator::aggregate_() {
    // By default, just forward the operator result.
    return op_.get_result();
}

void aggregator::aggregate(double weight) {
    // Calculate the weight * observable
    Eigen::MatrixXcd x = aggregate_();
    // if (result_.size() == 1 && std::real(x(0)) > 0)
    //     std::cout << x << ", " << mpi::rank << std::endl;
    result_.noalias() += weight * x;
    // Safly aggeregte the result

    if (track_variance_) {
        // Calculate the resul of the squared observable
        variance_.noalias() += weight * x.real().cwiseAbs2();
    }
}

prod_aggregator::prod_aggregator(const base_op& op, const base_op& scalar)
    : Base{op}, scalar_{scalar} {
    if (scalar_.rows() != 1 || scalar_.cols() != 1) {
        throw std::runtime_error("scalar operator must have size (1, 1).");
    }
}

Eigen::MatrixXcd prod_aggregator::aggregate_() {
    // Calculate op_a op_b^*
    return scalar_.get_result()(0) * op_.get_result().conjugate();
}

outer_aggregator::outer_aggregator(const base_op& op)
    : Base{op, op.rows(), op.rows()} {
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
    : Base{op, op.rows() * op.cols(), samples}, diag_(op.rows() * op.cols()) {}

void outer_aggregator_lazy::aggregate(double weight) {
    result_.col(current_index_) =
        std::sqrt(weight) * Eigen::Map<const Eigen::MatrixXcd>(
                                op_.get_result().data(), result_.rows(), 1);
    diag_.noalias() += result_.col(current_index_).cwiseAbs2();
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
