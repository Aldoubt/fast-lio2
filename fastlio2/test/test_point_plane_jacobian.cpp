#include "point_plane_jacobian.h"

#include <cmath>
#include <iostream>
#include <random>

int main()
{
    std::mt19937 generator(20261024);
    std::normal_distribution<double> gaussian;
    auto vector = [&]() {
        return Eigen::Vector3d(gaussian(generator), gaussian(generator),
                               gaussian(generator));
    };
    double worst_error = 0.0;
    double worst_old_error = 0.0;
    constexpr double epsilon = 1e-6;
    for (int sample = 0; sample < 500; ++sample) {
        const Eigen::Matrix3d r_wi = Sophus::SO3d::exp(vector()).matrix();
        const Eigen::Matrix3d r_il = Sophus::SO3d::exp(vector()).matrix();
        const Eigen::Vector3d t_il = 0.05 * vector();
        const Eigen::Vector3d t_wi = 100.0 * vector();
        const Eigen::Vector3d point = 10.0 * vector();
        const Eigen::Vector3d normal = vector().normalized();
        // Differentiate the complete world point-to-plane residual using the
        // right rotation increment used by State::operator+=.
        auto residual = [&](const Eigen::Vector3d &delta) {
            const Eigen::Vector3d world_point =
                r_wi * Sophus::SO3d::exp(delta).matrix() *
                (r_il * point + t_il) + t_wi;
            return normal.dot(world_point) + 0.37;
        };
        Eigen::RowVector3d numerical;
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d delta = Eigen::Vector3d::Zero();
            delta(axis) = epsilon;
            numerical(axis) = (residual(delta) - residual(-delta)) /
                              (2.0 * epsilon);
        }
        const auto analytic = fastlio2::pointPlaneRotationJacobian(
            r_wi, r_il, t_il, point, normal);
        worst_error = std::max(worst_error,
                              (analytic - numerical).cwiseAbs().maxCoeff());
        const Eigen::RowVector3d old = -normal.transpose() * r_wi *
            Sophus::SO3d::hat(r_il * point + t_wi);
        worst_old_error = std::max(worst_old_error,
                                  (old - numerical).cwiseAbs().maxCoeff());
    }
    std::cout << "cases=500 max_abs_error=" << worst_error
              << " old_max_abs_error=" << worst_old_error << '\n';
    if (!std::isfinite(worst_error) || worst_error > 2e-6 ||
        worst_old_error < 1.0) {
        return 1;
    }
    return 0;
}
