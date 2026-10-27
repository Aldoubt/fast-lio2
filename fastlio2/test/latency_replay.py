#!/usr/bin/env python3
"""
Offline latency/continuity regression, or receive-only checks of a running sensor.

Never starts a sensor or chassis. Use a private ROS domain, a frozen FAST-LIO
configuration and calibration bundle. Only processes created here are stopped.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry, Path as PathMessage
from sensor_msgs.msg import PointCloud2
from agt_robot_interfaces.msg import AdapterStatus
from std_msgs.msg import String
from std_srvs.srv import Trigger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--calibration-bundle', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--bag', type=Path)
    parser.add_argument('--domain', type=int, default=174)
    parser.add_argument('--limit', type=float, default=65.)
    parser.add_argument('--live', action='store_true',
                        help='receive from an already running sensor')
    parser.add_argument('--clouds', action='store_true')
    parser.add_argument('--path', action='store_true')
    parser.add_argument('--stall', type=float, default=0.,
                        help='SIGSTOP only the test frontend at t=15s')
    parser.add_argument('--qos', type=Path, help='rosbag2 publisher QoS override')
    parser.add_argument('--full-session-json', type=Path,
                        help='replay a saved field_session launch command')
    parser.add_argument('--relocalize', action='store_true')
    parser.add_argument('--expect-fault', action='store_true')
    args = parser.parse_args()
    if not args.live and args.bag is None:
        parser.error('--bag is required for offline replay')
    if args.domain == 0 or not 1 <= args.domain <= 232:
        parser.error('use a private nonzero ROS_DOMAIN_ID (1..232)')
    root = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = (args.binary or root/'install/fastlio2/lib/fastlio2/lio_node').resolve()
    config = args.config.resolve()
    os.environ.update(ROS_DOMAIN_ID=str(args.domain), ROS_LOCALHOST_ONLY='1',
                      ROS_LOG_DIR=str(output/'ros_logs'))
    children, handles = [], []

    def spawn(command, label):
        stream = (output/(label+'.log')).open('w')
        handles.append(stream)
        child = subprocess.Popen([str(part) for part in command], stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child

    def stop(child):
        if child.poll() is not None:
            return
        os.killpg(child.pid, signal.SIGCONT)
        os.killpg(child.pid, signal.SIGINT)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=5)

    sim = 'use_sim_time:='+str(not args.live).lower()
    rclpy.init()
    node = rclpy.create_node('fastlio_latency_regression', parameter_overrides=[
        Parameter('use_sim_time', value=not args.live)])
    rows = {'raw': [], 'local': []}
    statuses, global_status, subscriptions = [], [], []
    visual_counts = {'body_cloud': 0, 'world_cloud': 0, 'lio_path': 0}
    request = None
    client = node.create_client(Trigger, '/agt/localization/relocalize')

    def odom(kind, message):
        stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        position, rotation = message.pose.pose.position, message.pose.pose.orientation
        rows[kind].append([time.monotonic(), stamp,
                           node.get_clock().now().nanoseconds*1e-9-stamp,
                           position.x, position.y, position.z,
                           rotation.x, rotation.y, rotation.z, rotation.w])

    def status(message):
        statuses.append({key: getattr(message, key) for key in
                         ('accepted_count', 'rejected_count', 'state')})

    def global_update(message):
        if not global_status or global_status[-1] != message.data:
            global_status.append(message.data)

    def visual(topic, _message):
        visual_counts[topic] += 1

    for kind, topic in [('raw', '/fastlio2/lio_odom'), ('local', '/agt/odometry/local')]:
        subscriptions.append(node.create_subscription(
            Odometry, topic, lambda message, kind=kind: odom(kind, message), 100))
    subscriptions.append(node.create_subscription(
        AdapterStatus, '/agt/odometry/adapter_status', status, 20))
    subscriptions.append(node.create_subscription(
        String, '/agt/global_relocalization/status', global_update, 20))
    visual_topics = [('body_cloud', PointCloud2, args.clouds),
                     ('world_cloud', PointCloud2, args.clouds),
                     ('lio_path', PathMessage, args.path)]
    for topic, msg_type, enabled in visual_topics:
        if enabled:
            subscriptions.append(node.create_subscription(
                msg_type, '/fastlio2/'+topic,
                lambda message, topic=topic: visual(topic, message), 10))
    try:
        spawn(['ros2', 'launch', 'agt_field_test_bringup', 'field_test.launch.py',
               'robot:=bunker', 'enable_chassis:=false', 'enable_mid360:=false',
               'enable_localization:=false', 'enable_navigation:=false',
               'enable_guard:=false', sim,
               'calibration_bundle:='+str(args.calibration_bundle.resolve())], 'mount')
        if args.full_session_json:
            # Override executable lookup only, retaining the existing calibration/gates.
            prefix = output/'prefix'
            (prefix/'share/ament_index/resource_index/packages').mkdir(parents=True)
            (prefix/'share/ament_index/resource_index/packages/fastlio2').touch()
            (prefix/'lib/fastlio2').mkdir(parents=True)
            (prefix/'lib/fastlio2/lio_node').symlink_to(binary)
            (prefix/'share/fastlio2').symlink_to(root/'install/fastlio2/share/fastlio2')
            os.environ['AMENT_PREFIX_PATH'] = (
                str(prefix)+':'+os.environ.get('AMENT_PREFIX_PATH', ''))
            launch_args = json.loads(args.full_session_json.read_text())['command'][2:]
            options = dict(item.split(':=', 1) for item in launch_args if ':=' in item)
            # Saved sessions must be receive-only; never trust their hardware flags.
            for name in ('enable_chassis', 'enable_mid360', 'enable_navigation',
                         'enable_guard', 'external_chassis', 'enable_inspection',
                         'enable_camera_gimbal'):
                options[name] = 'false'
            options['use_sim_time'] = str(not args.live).lower()
            frontend = spawn(['ros2', 'launch', 'agt_field_test_bringup', 'field_test.launch.py',
                              *[name+':='+value for name, value in options.items()]], 'frontend')
        else:
            frontend = spawn([binary, '--ros-args', '-r', '__ns:=/fastlio2', '-p',
                              'config_path:='+str(config), '-p', sim], 'frontend')
            adapter_config = root/('install/agt_fastlio_adapter/share/agt_fastlio_adapter/'
                                   'config/fastlio_navigation_adapter.yaml')
            spawn([root/'install/agt_fastlio_adapter/lib/agt_fastlio_adapter/fastlio_adapter',
                   '--ros-args', '--params-file', adapter_config,
                   '-p', sim, '-p', 'body_to_base_calibration_file:='+str(config)], 'adapter')
        until = time.monotonic()+5
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.05)
        player = None
        if not args.live:
            command = ['ros2', 'bag', 'play', args.bag.resolve(), '--clock', '100',
                       '--disable-keyboard-controls', '--topics', '/livox/lidar', '/livox/imu']
            if args.qos:
                command += ['--qos-profile-overrides-path', args.qos.resolve()]
            player = spawn(command, 'player')
        start = time.monotonic()
        stalled, resume = False, None
        while time.monotonic()-start < args.limit and (player is None or player.poll() is None):
            rclpy.spin_once(node, timeout_sec=.01)
            if frontend.poll() is not None:
                raise RuntimeError('frontend exited')
            if (args.relocalize and request is None and time.monotonic()-start > 10
                    and client.service_is_ready()):
                request = client.call_async(Trigger.Request())
            if args.stall and not stalled and time.monotonic()-start > 15:
                if args.full_session_json:
                    raise ValueError('--stall requires a directly spawned frontend')
                os.kill(frontend.pid, signal.SIGSTOP)
                stalled, resume = True, time.monotonic()+args.stall
            if resume and time.monotonic() >= resume:
                os.kill(frontend.pid, signal.SIGCONT)
                resume = None
        if player:
            stop(player)
        summary = {'args': {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
                   'adapter': statuses[-1] if statuses else None, 'visual_counts': visual_counts,
                   'global_status': global_status,
                   'relocalize': str(request.result()) if request and request.done() else None}
        for kind, data in rows.items():
            with (output/(kind+'.csv')).open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['steady', 'stamp', 'age', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
                writer.writerows(data)
            steady = np.array([row for row in data if row[0] > start+5])
            if len(steady) > 2:
                summary[kind] = {
                    'count': len(data), 'hz': float((len(steady)-1)/(steady[-1, 0]-steady[0, 0])),
                    'age_ms': dict(zip(['min', 'p50', 'p95', 'p99', 'max'],
                                       (np.percentile(
                                           steady[:, 2], [0, 50, 95, 99, 100])*1000).tolist())),
                    'max_stamp_gap': float(np.diff(steady[:, 1]).max()),
                    'min_stamp_gap': float(np.diff(steady[:, 1]).min()),
                    'max_position_step': float(np.linalg.norm(
                        np.diff(steady[:, 3:6], axis=0), axis=1).max())}
        raw_stamps = {row[1] for row in rows['raw']}
        checks = {'canonical_stamps_preserved': bool(rows['local']) and all(
            row[1] in raw_stamps for row in rows['local'])}
        if args.expect_fault:
            log = (output/'frontend.log').read_text()
            checks['input_fault_latched'] = 'input_fault=1' in log
            checks['output_stopped'] = (bool(rows['raw'])
                                        and rows['raw'][-1][0] < start+args.limit-5)
        else:
            for kind in rows:
                result = summary.get(kind, {})
                checks[kind+'_10hz'] = 9 <= result.get('hz', 0) <= 11
                checks[kind+'_p95_below_200ms'] = result.get('age_ms', {}).get('p95', 1e9) < 200
                checks[kind+'_monotonic_stamps'] = result.get('min_stamp_gap', -1) > 0
        for topic, enabled in [('body_cloud', args.clouds),
                               ('world_cloud', args.clouds), ('lio_path', args.path)]:
            if enabled:
                checks[topic+'_published'] = visual_counts[topic] > 0
        summary['checks'] = checks
        summary['passed'] = all(checks.values())
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps({key: value for key, value in summary.items()
                          if key != 'global_status'}, indent=2))
        return 0 if summary['passed'] else 1
    finally:
        for child in reversed(children):
            stop(child)
        node.destroy_node()
        rclpy.shutdown()
        for stream in handles:
            stream.close()


if __name__ == '__main__':
    raise SystemExit(main())
