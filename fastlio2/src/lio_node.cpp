#include <mutex>
#include <vector>
#include <queue>
#include <memory>
#include <iostream>
#include <chrono>
#include <limits>
#include <algorithm>
// #include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

#include "utils.h"
#include "map_builder/commons.h"
#include "map_builder/map_builder.h"

#include <pcl_conversions/pcl_conversions.h>
#include "tf2_ros/transform_broadcaster.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;
struct NodeConfig
{
    std::string imu_topic = "/livox/imu";
    std::string lidar_topic = "/livox/lidar";
    std::string body_frame = "body";
    std::string world_frame = "lidar";
    bool print_time_cost = false;
};
struct StateData
{
    bool lidar_pushed = false;
    bool input_fault = false;
    // Protected by both input locks. Set before the first nonempty IMU package
    // reaches MapBuilder; after that no startup history may be discarded.
    bool builder_has_input = false;
    const char *fault_reason = "none";
    std::mutex imu_mutex;
    std::mutex lidar_mutex;
    double last_lidar_time = -1.0;
    double last_imu_time = -1.0;
    std::deque<IMUData> imu_buffer;
    std::deque<std::pair<double, pcl::PointCloud<pcl::PointXYZINormal>::Ptr>> lidar_buffer;
    nav_msgs::msg::Path path;
};

class LIONode : public rclcpp::Node
{
public:
    LIONode() : Node("lio_node")
    {
        RCLCPP_INFO(this->get_logger(), "LIO Node Started");
        RCLCPP_INFO(this->get_logger(),
            "Measurement model: imu-point-jacobian-v1 (R_il * p_l + t_il)");
        loadParameters();

        // Keep sensor reception runnable while scan matching/serialization is busy.
        // IMU and LiDAR share one mutually-exclusive input group; only the timer
        // owns the filter, package and map. Never run two process() calls at once.
        m_input_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions input_options;
        input_options.callback_group = m_input_group;
        m_imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(m_node_config.imu_topic, 2000, std::bind(&LIONode::imuCB, this, std::placeholders::_1), input_options);
        m_lidar_sub = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(m_node_config.lidar_topic, 10, std::bind(&LIONode::lidarCB, this, std::placeholders::_1), input_options);

        m_body_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("body_cloud", 10000);
        m_world_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("world_cloud", 10000);
        m_path_pub = this->create_publisher<nav_msgs::msg::Path>("lio_path", 10000);
        m_odom_pub = this->create_publisher<nav_msgs::msg::Odometry>("lio_odom", 10000);
        m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

        m_state_data.path.poses.clear();
        m_state_data.path.header.frame_id = m_node_config.world_frame;

        m_kf = std::make_shared<IESKF>();
        m_builder = std::make_shared<MapBuilder>(m_builder_config, m_kf);
        m_timer = this->create_wall_timer(20ms, std::bind(&LIONode::timerCB, this));
    }

    void loadParameters()
    {
        this->declare_parameter("config_path", "");
        std::string config_path;
        this->get_parameter<std::string>("config_path", config_path);

        YAML::Node config = YAML::LoadFile(config_path);
        if (!config)
        {
            RCLCPP_WARN(this->get_logger(), "FAIL TO LOAD YAML FILE!");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "LOAD FROM YAML CONFIG PATH: %s", config_path.c_str());

        m_node_config.imu_topic = config["imu_topic"].as<std::string>();
        m_node_config.lidar_topic = config["lidar_topic"].as<std::string>();
        m_node_config.body_frame = config["body_frame"].as<std::string>();
        m_node_config.world_frame = config["world_frame"].as<std::string>();
        m_node_config.print_time_cost = config["print_time_cost"].as<bool>();

        m_builder_config.lidar_filter_num = config["lidar_filter_num"].as<int>();
        m_builder_config.lidar_min_range = config["lidar_min_range"].as<double>();
        m_builder_config.lidar_max_range = config["lidar_max_range"].as<double>();
        m_builder_config.scan_resolution = config["scan_resolution"].as<double>();
        m_builder_config.map_resolution = config["map_resolution"].as<double>();
        m_builder_config.cube_len = config["cube_len"].as<double>();
        m_builder_config.det_range = config["det_range"].as<double>();
        m_builder_config.move_thresh = config["move_thresh"].as<double>();
        m_builder_config.na = config["na"].as<double>();
        m_builder_config.ng = config["ng"].as<double>();
        m_builder_config.nba = config["nba"].as<double>();
        m_builder_config.nbg = config["nbg"].as<double>();

        m_builder_config.imu_init_num = config["imu_init_num"].as<int>();
        m_builder_config.near_search_num = config["near_search_num"].as<int>();
        m_builder_config.ieskf_max_iter = config["ieskf_max_iter"].as<int>();
        m_builder_config.gravity_align = config["gravity_align"].as<bool>();
        m_builder_config.esti_il = config["esti_il"].as<bool>();
        std::vector<double> t_il_vec = config["t_il"].as<std::vector<double>>();
        std::vector<double> r_il_vec = config["r_il"].as<std::vector<double>>();
        m_builder_config.t_il << t_il_vec[0], t_il_vec[1], t_il_vec[2];
        m_builder_config.r_il << r_il_vec[0], r_il_vec[1], r_il_vec[2], r_il_vec[3], r_il_vec[4], r_il_vec[5], r_il_vec[6], r_il_vec[7], r_il_vec[8];
        m_builder_config.lidar_cov_inv = config["lidar_cov_inv"].as<double>();
    }

    using Steady = std::chrono::steady_clock;
    static double elapsedMs(Steady::time_point start)
    {
        return std::chrono::duration<double, std::milli>(Steady::now() - start).count();
    }

    // All age values use the node's ROS clock, including use_sim_time replay.
    // Durations and the 1 Hz log limiter use steady time.
    void diagnostics()
    {
        const auto tick = Steady::now();
        if (tick - m_diag_log < 1s) return;
        m_diag_log = tick;
        std::scoped_lock lock(m_state_data.imu_mutex, m_state_data.lidar_mutex);
        const double now = this->now().seconds();
        double front_end = std::numeric_limits<double>::quiet_NaN();
        if (!m_state_data.lidar_buffer.empty()) {
            const auto &front = m_state_data.lidar_buffer.front();
            double offset = 0.0;
            for (const auto &point : front.second->points)
                offset = std::max(offset, double(point.curvature) / 1000.0);
            front_end = front.first + offset;
        }
        const double wait = m_waiting_imu ? elapsedMs(m_imu_wait_start) : m_imu_wait_ms;
        RCLCPP_INFO(this->get_logger(),
            "LIO_DIAG q=%zu q_peak=%zu q_capacity=100 imu_q=%zu imu_capacity=4000 input_fault=%d fault_reason=%s front_age_ms=%.3f latest_front_ms=%.3f "
            "lidar_rx_age_ms=%.3f imu_rx_age_ms=%.3f imu_gap_ms=%.3f imu_wait_ms=%.3f "
            "process_ms=%.3f convert_ms=%.3f publish_ms=%.3f timer_ms=%.3f timer_max_ms=%.3f "
            "lidar_convert_ms=%.3f imu_dt_max_ms=%.3f imu_package=%zu empty_imu_packages=%zu imu_late=%zu lidar_late=%zu "
            "odom_age_ms=%.3f last_odom_age_ms=%.3f processed=%zu odom=%zu imu_init_ros=%.9f "
            "first_odom_ros=%.9f first_odom_age_ms=%.3f first_odom_delay_ms=%.3f "
            "startup_imu_dropped=%zu startup_lidar_dropped=%zu startup_restarts=%zu",
            m_state_data.lidar_buffer.size(), m_queue_peak, m_state_data.imu_buffer.size(),
            int(m_state_data.input_fault), m_state_data.fault_reason,
            (now - front_end) * 1000.0, (m_latest_scan_end - front_end) * 1000.0,
            m_lidar_rx_age_ms, m_imu_rx_age_ms,
            m_state_data.lidar_pushed ? (m_package.cloud_end_time - m_state_data.last_imu_time) * 1000.0 : 0.0,
            wait, m_process_ms, m_convert_ms, m_publish_ms, m_timer_ms, m_timer_max_ms,
            m_input_convert_ms, m_imu_dt_max_ms, m_imu_package_count, m_empty_imu_packages, m_imu_out_of_order, m_lidar_out_of_order,
            m_odom_age_ms, m_odom_count ? (now - m_last_odom_stamp) * 1000.0 : -1.0,
            m_processed, m_odom_count, m_imu_init_ros, m_first_odom_ros, m_first_odom_age_ms, m_first_odom_delay_ms,
            m_startup_imu_dropped, m_startup_lidar_dropped, m_startup_restarts);
        if (m_state_data.input_fault || m_state_data.lidar_buffer.size() >= 20 || (now - front_end) > 0.20)
            RCLCPP_WARN(this->get_logger(), "LIO backlog: queue/age budget exceeded; no scan clearing. input_fault=1 requires node restart");
        m_timer_max_ms = 0.0;
    }

    void imuCB(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        std::scoped_lock lock(m_state_data.imu_mutex, m_state_data.lidar_mutex);
        double timestamp = Utils::getSec(msg->header);
        m_imu_rx_age_ms = (this->now().seconds() - timestamp) * 1000.0;
        // Livox/DDS can deliver device-startup history before its live stream.
        // The filter is still pristine here: select fresh inputs BEFORE it has
        // accumulated even one initialization sample. Never skip runtime IMUs.
        if (!m_state_data.builder_has_input &&
            (m_imu_rx_age_ms > 200.0 || m_imu_rx_age_ms < -50.0)) {
            ++m_startup_imu_dropped;
            return;
        }
        if (m_state_data.imu_buffer.size() >= 4000) {
            m_state_data.input_fault = true;
            m_state_data.fault_reason = "imu_capacity";
        }
        if (m_state_data.input_fault) return; // Latch closed: require an explicit node restart.
        if (timestamp < m_state_data.last_imu_time)
        {
            ++m_imu_out_of_order;
            if (m_state_data.last_imu_time - timestamp > 0.10) {
                m_state_data.input_fault = true;
                m_state_data.fault_reason = "imu_time_regression";
            }
            return; // Reject only this late sample; preserve the synchronized history.
        }
        // A backlog can be replayed only with continuous IMU history. A DDS
        // publisher may have discarded unacknowledged samples during a stall;
        // enlarging this subscriber cannot recover them. Fail closed rather
        // than extrapolating across >20 missing nominal 200 Hz samples.
        if (m_state_data.last_imu_time >= 0.0 && timestamp - m_state_data.last_imu_time > 0.10) {
            m_imu_dt_max_ms = std::max(m_imu_dt_max_ms, (timestamp - m_state_data.last_imu_time) * 1000.0);
            if (m_state_data.builder_has_input) {
                m_state_data.input_fault = true;
                m_state_data.fault_reason = "imu_history_gap";
                return;
            }
            // No estimator/IMU initializer has consumed data yet. Re-arm only
            // the input synchronizer and keep this first post-gap IMU sample.
            m_state_data.imu_buffer.clear();
            m_state_data.lidar_buffer.clear();
            m_state_data.lidar_pushed = false;
            m_state_data.last_lidar_time = -1.0;
            ++m_startup_restarts;
        }
        m_state_data.imu_buffer.emplace_back(V3D(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z) * 10.0,
                                             V3D(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                                             timestamp);
        if (m_state_data.last_imu_time >= 0.0)
            m_imu_dt_max_ms = std::max(m_imu_dt_max_ms, (timestamp - m_state_data.last_imu_time) * 1000.0);
        m_state_data.last_imu_time = timestamp;
    }
    void lidarCB(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        const auto convert_start = Steady::now();
        {
            std::scoped_lock lock(m_state_data.imu_mutex, m_state_data.lidar_mutex);
            // A 10 Hz scan begins at most 0.10 s before its end. Avoid converting
            // boot history that cannot satisfy the unchanged 0.20 s age budget.
            const double start_age = this->now().seconds() - Utils::getSec(msg->header);
            if (!m_state_data.builder_has_input && (start_age > 0.30 || start_age < -0.05)) {
                ++m_startup_lidar_dropped;
                return;
            }
        }
        // At 10 Hz a point offset near UINT32_MAX is not a long scan: a host
        // clock backstep can underflow the driver's unsigned relative time.
        // Reject before it consumes future IMUs or advances the filter state.
        if (std::any_of(msg->points.begin(), msg->points.end(), [](const auto &point) {
                return point.offset_time > 200000000U;
            })) {
            std::scoped_lock lock(m_state_data.imu_mutex, m_state_data.lidar_mutex);
            m_state_data.input_fault = true;
            m_state_data.fault_reason = "invalid_lidar_point_time";
            return;
        }
        CloudType::Ptr cloud = Utils::livox2PCL(msg, m_builder_config.lidar_filter_num, m_builder_config.lidar_min_range, m_builder_config.lidar_max_range);
        // A CustomMsg is expected to be one aggregated LiDAR scan.  Never let
        // empty/undersized packets enter the LIO state machine.
        if (!cloud || cloud->empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Dropping empty LiDAR scan");
            return;
        }
        std::scoped_lock lock(m_state_data.imu_mutex, m_state_data.lidar_mutex);
        m_input_convert_ms = elapsedMs(convert_start);
        if (m_state_data.lidar_buffer.size() >= 100) {
            m_state_data.input_fault = true;
            m_state_data.fault_reason = "lidar_capacity";
        }
        if (m_state_data.input_fault) return;
        double timestamp = Utils::getSec(msg->header);
        if (timestamp < m_state_data.last_lidar_time)
        {
            ++m_lidar_out_of_order;
            return; // Do not clear a queue whose front may already be in m_package.
        }
        double offset = 0.0;
        for (const auto &point : cloud->points)
            offset = std::max(offset, double(point.curvature) / 1000.0);
        m_latest_scan_end = timestamp + offset;
        m_lidar_rx_age_ms = (this->now().seconds() - m_latest_scan_end) * 1000.0;
        if (!m_first_scan_seen) {
            m_first_scan_seen = true;
            m_first_scan_ros = this->now().seconds();
        }
        m_state_data.lidar_buffer.emplace_back(timestamp, cloud);
        m_queue_peak = std::max(m_queue_peak, m_state_data.lidar_buffer.size());
        m_state_data.last_lidar_time = timestamp;
    }

    bool syncPackage()
    {
        std::scoped_lock lock(m_state_data.imu_mutex, m_state_data.lidar_mutex);
        if (m_state_data.input_fault) return false;
        if (m_state_data.imu_buffer.empty() || m_state_data.lidar_buffer.empty())
            return false;
        if (!m_state_data.lidar_pushed)
        {
            m_package.cloud = m_state_data.lidar_buffer.front().second;
            if (!m_package.cloud || m_package.cloud->empty()) {
                m_state_data.lidar_buffer.pop_front();
                return false;
            }
            std::sort(m_package.cloud->points.begin(), m_package.cloud->points.end(), [](PointType &p1, PointType &p2)
                      { return p1.curvature < p2.curvature; });
            m_package.cloud_start_time = m_state_data.lidar_buffer.front().first;
            m_package.cloud_end_time = m_package.cloud_start_time + m_package.cloud->points.back().curvature / 1000.0;
            m_state_data.lidar_pushed = true;
        }
        if (m_state_data.last_imu_time < m_package.cloud_end_time) {
            if (!m_waiting_imu) m_imu_wait_start = Steady::now();
            m_waiting_imu = true;
            return false;
        }
        m_imu_wait_ms = m_waiting_imu ? elapsedMs(m_imu_wait_start) : 0.0;
        m_waiting_imu = false;

        Vec<IMUData>().swap(m_package.imus);
        while (!m_state_data.imu_buffer.empty() && m_state_data.imu_buffer.front().time < m_package.cloud_end_time)
        {
            m_package.imus.emplace_back(m_state_data.imu_buffer.front());
            m_state_data.imu_buffer.pop_front();
        }
        m_imu_package_count = m_package.imus.size();
        if (m_package.imus.empty()) {
            ++m_empty_imu_packages;
            if (m_builder->status() != BuilderStatus::IMU_INIT) {
                m_state_data.input_fault = true;
                m_state_data.fault_reason = "empty_imu_package";
                return false;
            }
        }
        m_state_data.lidar_buffer.pop_front();
        m_state_data.lidar_pushed = false;
        if (!m_package.imus.empty()) m_state_data.builder_has_input = true;
        return true;
    }

    void publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub, CloudType::Ptr cloud, std::string frame_id, const double &time)
    {
        if (pub->get_subscription_count() <= 0)
            return;
        const auto convert_start = Steady::now();
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = frame_id;
        cloud_msg.header.stamp = Utils::getTime(time);
        m_convert_ms += elapsedMs(convert_start);
        const auto publish_start = Steady::now();
        pub->publish(cloud_msg);
        m_publish_ms += elapsedMs(publish_start);
    }

    void publishOdometry(rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub, std::string frame_id, std::string child_frame, const double &time)
    {
        if (odom_pub->get_subscription_count() <= 0)
            return;
        nav_msgs::msg::Odometry odom;
        odom.header.frame_id = frame_id;
        odom.header.stamp = Utils::getTime(time);
        odom.child_frame_id = child_frame;
        odom.pose.pose.position.x = m_kf->x().t_wi.x();
        odom.pose.pose.position.y = m_kf->x().t_wi.y();
        odom.pose.pose.position.z = m_kf->x().t_wi.z();
        Eigen::Quaterniond q(m_kf->x().r_wi);
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();

        V3D vel = m_kf->x().r_wi.transpose() * m_kf->x().v;
        odom.twist.twist.linear.x = vel.x();
        odom.twist.twist.linear.y = vel.y();
        odom.twist.twist.linear.z = vel.z();
        odom_pub->publish(odom);
        m_odom_age_ms = (this->now().seconds() - time) * 1000.0;
        m_last_odom_stamp = time;
        ++m_odom_count;
        if (m_first_odom_delay_ms < 0.0) {
            m_first_odom_ros = this->now().seconds();
            m_first_odom_age_ms = m_odom_age_ms;
            m_first_odom_delay_ms = (m_first_odom_ros - m_first_scan_ros) * 1000.0;
        }
    }

    void publishPath(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub, std::string frame_id, const double &time)
    {
        if (path_pub->get_subscription_count() <= 0)
            return;
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = frame_id;
        pose.header.stamp = Utils::getTime(time);
        pose.pose.position.x = m_kf->x().t_wi.x();
        pose.pose.position.y = m_kf->x().t_wi.y();
        pose.pose.position.z = m_kf->x().t_wi.z();
        Eigen::Quaterniond q(m_kf->x().r_wi);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        m_state_data.path.poses.push_back(pose);
        path_pub->publish(m_state_data.path);
    }

    void broadCastTF(std::shared_ptr<tf2_ros::TransformBroadcaster> broad_caster, std::string frame_id, std::string child_frame, const double &time)
    {
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.frame_id = frame_id;
        transformStamped.child_frame_id = child_frame;
        transformStamped.header.stamp = Utils::getTime(time);
        Eigen::Quaterniond q(m_kf->x().r_wi);
        V3D t = m_kf->x().t_wi;
        transformStamped.transform.translation.x = t.x();
        transformStamped.transform.translation.y = t.y();
        transformStamped.transform.translation.z = t.z();
        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();
        broad_caster->sendTransform(transformStamped);
    }

    void timerCB()
    {
        const auto start = Steady::now();
        const auto processed_before = m_processed;
        // A scope guard covers synchronization waits and initialization returns too.
        const auto finish = [this, start, processed_before](void *) {
            const double callback_ms = elapsedMs(start);
            if (m_processed != processed_before) m_timer_ms = callback_ms;
            m_timer_max_ms = std::max(m_timer_max_ms, callback_ms);
            diagnostics();
        };
        std::unique_ptr<void, decltype(finish)> guard(this, finish);
        if (!syncPackage()) return;
        const auto status = m_builder->status();
        const auto process_start = Steady::now();
        m_builder->process(m_package);
        m_process_ms = elapsedMs(process_start);
        ++m_processed;
        m_convert_ms = m_publish_ms = 0.0;
        if (status == BuilderStatus::IMU_INIT && m_builder->status() != BuilderStatus::IMU_INIT)
            m_imu_init_ros = this->now().seconds();
        if (m_builder->status() != BuilderStatus::MAPPING) return;

        auto stage = Steady::now();
        broadCastTF(m_tf_broadcaster, m_node_config.world_frame, m_node_config.body_frame, m_package.cloud_end_time);
        publishOdometry(m_odom_pub, m_node_config.world_frame, m_node_config.body_frame, m_package.cloud_end_time);
        m_publish_ms += elapsedMs(stage);

        if (m_body_cloud_pub->get_subscription_count() > 0) {
            stage = Steady::now();
            CloudType::Ptr body_cloud = m_builder->lidar_processor()->transformCloud(m_package.cloud, m_kf->x().r_il, m_kf->x().t_il);
            m_convert_ms += elapsedMs(stage);
            publishCloud(m_body_cloud_pub, body_cloud, m_node_config.body_frame, m_package.cloud_end_time);
        }

        if (m_world_cloud_pub->get_subscription_count() > 0) {
            stage = Steady::now();
            CloudType::Ptr world_cloud = m_builder->lidar_processor()->transformCloud(m_package.cloud, m_builder->lidar_processor()->r_wl(), m_builder->lidar_processor()->t_wl());
            m_convert_ms += elapsedMs(stage);
            publishCloud(m_world_cloud_pub, world_cloud, m_node_config.world_frame, m_package.cloud_end_time);
        }
        if (m_path_pub->get_subscription_count() > 0) {
            stage = Steady::now();
            publishPath(m_path_pub, m_node_config.world_frame, m_package.cloud_end_time);
            m_publish_ms += elapsedMs(stage);
        }
    }

private:
    rclcpp::CallbackGroup::SharedPtr m_input_group;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr m_lidar_sub;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr m_imu_sub;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_body_cloud_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_world_cloud_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr m_path_pub;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr m_odom_pub;

    rclcpp::TimerBase::SharedPtr m_timer;
    Steady::time_point m_diag_log = Steady::now(), m_imu_wait_start;
    bool m_waiting_imu = false, m_first_scan_seen = false;
    size_t m_queue_peak = 0, m_processed = 0, m_odom_count = 0;
    size_t m_startup_imu_dropped = 0, m_startup_lidar_dropped = 0, m_startup_restarts = 0;
    size_t m_imu_out_of_order = 0, m_lidar_out_of_order = 0, m_imu_package_count = 0, m_empty_imu_packages = 0;
    double m_input_convert_ms = 0.0, m_imu_dt_max_ms = 0.0;
    double m_latest_scan_end = 0.0, m_first_scan_ros = 0.0, m_imu_init_ros = -1.0;
    double m_first_odom_ros = -1.0, m_first_odom_age_ms = -1.0, m_last_odom_stamp = 0.0;
    double m_first_odom_delay_ms = -1.0, m_imu_wait_ms = 0.0;
    double m_lidar_rx_age_ms = 0.0, m_imu_rx_age_ms = 0.0;
    double m_process_ms = 0.0, m_convert_ms = 0.0, m_publish_ms = 0.0;
    double m_timer_ms = 0.0, m_timer_max_ms = 0.0, m_odom_age_ms = -1.0;
    StateData m_state_data;
    SyncPackage m_package;
    NodeConfig m_node_config;
    Config m_builder_config;
    std::shared_ptr<IESKF> m_kf;
    std::shared_ptr<MapBuilder> m_builder;
    std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    auto node = std::make_shared<LIONode>();
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
