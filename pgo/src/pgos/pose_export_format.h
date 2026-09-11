#pragma once

#include <Eigen/Geometry>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

inline std::string formatLegacyPoseLine(const std::string &patch_name,
                                        const Eigen::Vector3d &t,
                                        const Eigen::Quaterniond &q)
{
    // This exactly preserves saveMapsCB's historic default ostream formatting.
    std::ostringstream out;
    out << patch_name << " " << t.x() << " " << t.y() << " " << t.z() << " "
        << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
    return out.str();
}

inline std::string formatTimedPoseLine(const std::string &patch_name, double stamp,
                                       const Eigen::Vector3d &t,
                                       const Eigen::Quaterniond &q)
{
    std::ostringstream out;
    out << patch_name << " " << std::fixed << std::setprecision(9) << stamp
        << std::defaultfloat << " "
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << t.x() << " " << t.y() << " " << t.z() << " "
        << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
    return out.str();
}
