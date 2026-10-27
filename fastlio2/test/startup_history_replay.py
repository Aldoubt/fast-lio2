#!/usr/bin/env python3
"""Private-domain replay: stale startup history, pristine gap, then runtime gap.

Original bag headers are preserved. Only /clock delivery is controlled; this
never subscribes to live sensors, sends goals, or starts a chassis/driver.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--domain', type=int, default=186)
    parser.add_argument('--runtime-gap', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.domain <= 232:
        parser.error('use an explicit private nonzero ROS domain')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(ROS_DOMAIN_ID=str(args.domain), ROS_LOCALHOST_ONLY='1',
                      ROS_LOG_DIR=str(args.output/'ros_logs'))
    import numpy as np
    import rclpy
    import rosbag2_py
    from rclpy.parameter import Parameter
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from livox_ros_driver2.msg import CustomMsg
    from rosgraph_msgs.msg import Clock
    import yaml

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/livox/imu', '/livox/lidar']))
    rows, first = [], None
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if first is None:
            first = stamp
        if stamp-first > 13_000_000_000:
            break
        rows.append((topic, raw, stamp))
    if not rows or rows[-1][2]-first < 12_000_000_000:
        raise ValueError('source must contain at least 12 seconds of raw input')
    config = yaml.safe_load(args.config.read_text())
    (args.output/'input.yaml').write_text(yaml.safe_dump(config))
    rclpy.init()
    node = rclpy.create_node('startup_history_replay', parameter_overrides=[Parameter('use_sim_time', value=True)])
    publishers = {'/livox/imu': node.create_publisher(Imu, config['imu_topic'], 2000),
                  '/livox/lidar': node.create_publisher(CustomMsg, config['lidar_topic'], 10)}
    clock = node.create_publisher(Clock, '/clock', 10)
    odometry = []
    def on_odom(msg):
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        p = msg.pose.pose.position
        odometry.append(dict(stamp=stamp, age_ms=(node.get_clock().now().nanoseconds*1e-9-stamp)*1000,
                             position=[p.x, p.y, p.z]))
    node.create_subscription(Odometry, '/fastlio2/lio_odom', on_odom, 100)
    def set_clock(stamp):
        msg = Clock()
        msg.clock.sec, msg.clock.nanosec = divmod(stamp, 10**9)
        clock.publish(msg)
    stream = (args.output/'frontend.log').open('w')
    child = subprocess.Popen([str(args.binary.resolve()), '--ros-args', '-r', '__ns:=/fastlio2',
                              '-p', 'config_path:='+str((args.output/'input.yaml').resolve()),
                              '-p', 'use_sim_time:=true'], stdout=stream, stderr=subprocess.STDOUT,
                             start_new_session=True)
    source_stamps = set()
    try:
        deadline = time.monotonic()+6
        while time.monotonic() < deadline:
            set_clock(first+3_000_000_000)
            rclpy.spin_once(node, timeout_sec=.02)
            if all(pub.get_subscription_count() for pub in publishers.values()) and clock.get_subscription_count() >= 2:
                break
        assert all(pub.get_subscription_count() for pub in publishers.values()), 'frontend not discovered'
        for _ in range(25):
            set_clock(first+3_000_000_000)
            rclpy.spin_once(node, timeout_sec=.02)
        # Deliver old boot packets at ROS time +3 s, with a >0.2 s IMU hole.
        for topic, raw, stamp in rows:
            delta = (stamp-first)*1e-9
            if delta >= 1.0:
                break
            if topic == '/livox/imu' and .20 <= delta < .42:
                continue
            publishers[topic].publish(deserialize_message(raw, Imu if topic.endswith('imu') else CustomMsg))
            rclpy.spin_once(node, timeout_sec=.001)
        started = time.monotonic()
        for topic, raw, stamp in rows:
            delta = (stamp-first)*1e-9
            if not 3.0 <= delta < 12.0:
                continue
            # A fresh gap BEFORE any scan reaches MapBuilder must re-arm only
            # the pristine synchronizer; a later gap MUST latch closed.
            if topic == '/livox/lidar' and delta < 3.5:
                continue
            if topic == '/livox/imu' and (3.10 <= delta < 3.35 or
                                         (args.runtime_gap and 8.0 <= delta < 8.30)):
                continue
            until = started+delta-3.0
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=min(.003, max(0., until-time.monotonic())))
            set_clock(stamp)
            message = deserialize_message(raw, Imu if topic.endswith('imu') else CustomMsg)
            if topic.endswith('lidar'):
                # Output must use an actual unchanged scan-end timestamp.
                begin = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
                kept = [p for p in message.points[::config['lidar_filter_num']]
                        if p.line < 4 and (p.tag & 0x30) in (0, 0x10) and
                        config['lidar_min_range']**2 <= p.x*p.x+p.y*p.y+p.z*p.z <= config['lidar_max_range']**2]
                if kept:
                    curvature = max(float(np.float32(p.offset_time)/np.float32(1e6)) for p in kept)
                    source_stamps.add(round(begin+curvature/1000., 5))
            publishers[topic].publish(message)
            rclpy.spin_once(node, timeout_sec=0.)
        deadline = time.monotonic()+1.2
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
        stream.flush()
        log = (args.output/'frontend.log').read_text()
        counters = {key: max([int(v) for v in re.findall(key+r'=(\d+)', log)] or [0])
                    for key in ['startup_imu_dropped', 'startup_lidar_dropped', 'startup_restarts']}
        stable = [r for r in odometry if first*1e-9+5 <= r['stamp'] < first*1e-9+7.8]
        stamps = np.array([row['stamp'] for row in stable])
        positions = np.array([row['position'] for row in stable])
        checks = dict(startup_history_rejected=counters['startup_imu_dropped'] > 0 and counters['startup_lidar_dropped'] > 0,
                      pristine_gap_rearmed=counters['startup_restarts'] > 0,
                      odometry_started=len(stable) >= 20,
                      scan_timestamps_preserved=bool(odometry) and all(round(row['stamp'], 5) in source_stamps for row in odometry),
                      finite_pose=bool(stable) and bool(np.isfinite(positions).all()))
        if args.runtime_gap:
            checks.update(runtime_gap_latched='input_fault=1 fault_reason=imu_history_gap' in log,
                          output_stopped=bool(odometry) and odometry[-1]['stamp'] < first*1e-9+8.5)
        else:
            checks.update(no_latched_fault='input_fault=1' not in log,
                          continued_to_end=bool(odometry) and odometry[-1]['stamp'] > first*1e-9+11.)
        summary = dict(checks=checks, passed=all(checks.values()), counters=counters,
                       odom_count=len(odometry), steady_hz=float((len(stamps)-1)/(stamps[-1]-stamps[0])) if len(stamps)>1 else None,
                       age_ms={k:float(v) for k,v in zip(['p50','p95','max'],np.percentile([r['age_ms'] for r in stable],[50,95,100]))} if stable else {},
                       max_position_step_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).max()) if len(positions)>1 else None)
        (args.output/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
        (args.output/'odom.json').write_text(json.dumps(odometry))
        print(json.dumps(summary,indent=2))
        return 0 if summary['passed'] else 1
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        stream.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
