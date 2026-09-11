#include <gtest/gtest.h>

#include "pgos/pose_export_format.h"

TEST(PoseExportFormat, TimedFormattingCannotChangeLegacyFormatting)
{
    const Eigen::Vector3d translation(12.3456789012345, -0.000123456789, 42.0);
    const Eigen::Quaterniond orientation(0.987654321, 0.012345678, -0.123456789, 0.0987654321);

    const std::string legacy_before = formatLegacyPoseLine("17.pcd", translation, orientation);
    const std::string timed = formatTimedPoseLine("17.pcd", 1788268266.940766811,
                                                   translation, orientation);
    const std::string legacy_after = formatLegacyPoseLine("17.pcd", translation, orientation);

    EXPECT_EQ(legacy_before, legacy_after);
    EXPECT_EQ("17.pcd 12.3457 -0.000123457 42 0.987654 0.0123457 -0.123457 0.0987654\n",
              legacy_after);
    const std::string timestamp_prefix = "17.pcd 1788268266.";
    const size_t timestamp_begin = timed.find(timestamp_prefix);
    ASSERT_NE(std::string::npos, timestamp_begin);
    EXPECT_EQ(' ', timed[timestamp_begin + timestamp_prefix.size() + 9]);
}
