#!/usr/bin/env python3
"""Extract sensor_msgs/msg/Image messages from a rosbag2 folder and save them as JPEG images.

Usage:
    python extract_images.py <bag_path> <topic> [--output OUTPUT_DIR] [--frame_interval N]
"""

import os
import argparse
import re

import cv2
from cv_bridge import CvBridge
import rosbag2_py
from sensor_msgs.msg import Image
from rclpy.serialization import deserialize_message


def main():
    """Parse arguments and extract images from rosbag2 file."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract sensor_msgs/msg/Image messages from a rosbag2 folder and "
            "save them as JPEG images."
        )
    )
    parser.add_argument(
        "bag_path",
        help="Path to the rosbag2 folder (not a single file)",
    )
    parser.add_argument(
        "topic",
        help="Image topic to extract (e.g. /camera/image)",
    )
    parser.add_argument(
        "--output",
        default="output_images",
        help="Output directory for JPEG images",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Save only every nth frame (default=1 saves all frames)",
    )
    args = parser.parse_args()

    # Create output directory if it does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        saved_count = 0
    else:
        saved_count = 0
        for filename in os.listdir(args.output):
            match = re.match(r"image_(\d+)\.jpg", filename)
            if match:
                index = int(match.group(1))
                if index >= saved_count:
                    saved_count = index + 1

    storage_options = rosbag2_py.StorageOptions(
        uri=args.bag_path,
        storage_id="sqlite3",
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    bridge = CvBridge()
    frame_idx = 0

    while reader.has_next():
        topic_name, data, _ = reader.read_next()
        if topic_name != args.topic:
            frame_idx += 1
            continue

        if frame_idx % args.frame_interval == 0:
            msg = deserialize_message(data, Image)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            filename = os.path.join(
                args.output,
                f"image_{saved_count:06d}.jpg",
            )
            cv2.imwrite(filename, cv_img)
            print(f"Saved {filename}")
            saved_count += 1

        frame_idx += 1

    print(
        f"Finished extracting images. "
        f"Processed {frame_idx} frames on topic '{args.topic}'."
    )


if __name__ == "__main__":
    main()
