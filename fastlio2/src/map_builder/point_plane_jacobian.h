#pragma once

#include <Eigen/Core>
#include <sophus/so3.hpp>

namespace fastlio2 {

// State applies a right SO(3) increment to R_wi.  The rotated lever arm is
// the point in IMU coordinates; world translation is outside that rotation.
inline Eigen::RowVector3d pointPlaneRotationJacobian(
    const Eigen::Matrix3d &r_wi, const Eigen::Matrix3d &r_il,
    const Eigen::Vector3d &t_il, const Eigen::Vector3d &point_lidar,
    const Eigen::Vector3d &normal_world)
{
    return -normal_world.transpose() * r_wi *
           Sophus::SO3d::hat(r_il * point_lidar + t_il);
}

}  // namespace fastlio2
