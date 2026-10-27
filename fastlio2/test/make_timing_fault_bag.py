#!/usr/bin/env python3
"""Create a derived timing-fault bag for an isolated ROS domain; source stays unchanged."""
import argparse
import json
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from livox_ros_driver2.msg import CustomMsg
from sensor_msgs.msg import Imu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--fault', choices=('point_wrap', 'imu_backstep'), required=True)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error('destination must not exist')
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.source), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.destination), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    topics = {'/livox/lidar', '/livox/imu'}
    for metadata in reader.get_all_topics_and_types():
        if metadata.name in topics:
            writer.create_topic(metadata)
    first = None
    injected = None
    count = 0
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic not in topics:
            continue
        if first is None:
            first = stamp
        if stamp-first > 24_000_000_000:
            break
        if injected is None and stamp-first >= 12_000_000_000:
            if args.fault == 'point_wrap' and topic == '/livox/lidar':
                message = deserialize_message(data, CustomMsg)
                message.points[0].offset_time = 4_294_000_000
                data = serialize_message(message)
                injected = dict(topic=topic, bag_stamp=stamp, point_offset_ns=4_294_000_000)
            elif args.fault == 'imu_backstep' and topic == '/livox/imu':
                message = deserialize_message(data, Imu)
                old = message.header.stamp.sec*10**9+message.header.stamp.nanosec
                message.header.stamp.sec, message.header.stamp.nanosec = divmod(old-500_000_000, 10**9)
                data = serialize_message(message)
                injected = dict(topic=topic, bag_stamp=stamp, backstep_ns=500_000_000)
        writer.write(topic, data, stamp)
        count += 1
    del writer
    if injected is None:
        raise RuntimeError('source bag too short to inject fault')
    (args.destination/'synthetic_fault.json').write_text(json.dumps(dict(
        source=str(args.source.resolve()), fault=args.fault, messages=count, injected=injected), indent=2)+'\n')


if __name__ == '__main__':
    main()
