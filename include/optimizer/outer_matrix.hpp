/*
 * Copyright (C) 2021  David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// See https://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace optimizer {
class OuterMatrix;
}

namespace Eigen {
namespace internal {
// MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
template <>
struct traits<optimizer::OuterMatrix>
    : public Eigen::internal::traits<
          Eigen::SparseMatrix<std::complex<double>>> {};
}  // namespace internal
}  // namespace Eigen

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
namespace optimizer {
class OuterMatrix : public Eigen::EigenBase<OuterMatrix> {
   public:
    // Required typedefs, constants, and method:
    typedef std::complex<double> Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Index rows() const { return mp_mat->rows(); }
    Index cols() const { return mp_mat->rows(); }

    template <typename Rhs>
    Eigen::Product<OuterMatrix, Rhs, Eigen::AliasFreeProduct> operator*(
        const Eigen::MatrixBase<Rhs>& x) const {
        return Eigen::Product<OuterMatrix, Rhs, Eigen::AliasFreeProduct>(
            *this, x.derived());
    }

    // Custom API:
    OuterMatrix(double norm, double reg = 0.)
        : mp_mat(0), mp_vec(0), norm{norm}, reg{reg} {}

    void attachMyMatrix(const Eigen::MatrixXcd& mat) { mp_mat = &mat; }
    void attachMyVector(const Eigen::MatrixXcd& vec) { mp_vec = &vec; }
    const Eigen::MatrixXcd my_matrix() const { return *mp_mat; }
    const Eigen::MatrixXcd my_vector() const { return *mp_vec; }
    const double get_norm() const { return norm; }
    const double get_reg() const { return reg; }

   private:
    const Eigen::MatrixXcd* mp_mat;
    const Eigen::MatrixXcd* mp_vec;
    const double norm;
    const double reg;
};
}  // namespace optimizer

// Implementation of MatrixReplacement * Eigen::DenseVector though a
// specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
using optimizer::OuterMatrix;

template <typename Rhs>
struct generic_product_impl<OuterMatrix, Rhs, SparseShape, DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<OuterMatrix, Rhs,
                                generic_product_impl<OuterMatrix, Rhs>> {
    typedef typename Product<OuterMatrix, Rhs>::Scalar Scalar;

    template <typename Dest>
    static void scaleAndAddTo(Dest& dst, const OuterMatrix& lhs, const Rhs& rhs,
                              const Scalar& alpha) {
        assert(alpha == Scalar(1) && "scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);

        auto mat = lhs.my_matrix();
        auto vec = lhs.my_vector();
        dst += (mat.conjugate() * (mat.transpose() * rhs)) / lhs.get_norm();
        dst -= (vec.conjugate() * (vec.transpose() * rhs));
        dst += lhs.get_reg() * rhs;
    }
};

}  // namespace internal
}  // namespace Eigen
