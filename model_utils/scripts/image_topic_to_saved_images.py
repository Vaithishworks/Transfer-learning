#!/usr/bin/env python3
"""ROS 2 node that subscribes to sensor_msgs/msg/Image messages and saves every nth received image as JPEG files.

Usage:
    ./image_saver_node.py [--ros-args -p folder_path:=/tmp/images -p image_topic:=/camera/image -p n_th_frame:=15]
"""

import os
import sys
import re

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageSaverNode(Node):
    """
    ROS 2 node that subscribes to sensor_msgs/msg/Image messages and saves every
    nth received image as a JPG file in a given folder. Saved images are numbered
    continuously without overwriting existing files.
    """

    def __init__(self):
        """Initialize node, declare parameters, prepare folder, and set up subscription."""
        super().__init__("image_saver_node")
        self.declare_parameter("folder_path", "/tmp/images")
        self.folder_path = (
            self.get_parameter("folder_path")
            .get_parameter_value()
            .string_value
        )

        self.declare_parameter("image_topic", "image_topic")
        topic_name = (
            self.get_parameter("image_topic")
            .get_parameter_value()
            .string_value
        )

        self.declare_parameter("n_th_frame", 15)
        self.n_th_frame = (
            self.get_parameter("n_th_frame")
            .get_parameter_value()
            .integer_value
        )

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            self.get_logger().info(f"Created folder: {self.folder_path}")
            self.saved_count = 0
        else:
            pattern = re.compile(r"image_(\d+)\.jpg")
            existing_numbers = []
            for filename in os.listdir(self.folder_path):
                match = pattern.match(filename)
                if match:
                    existing_numbers.append(int(match.group(1)))
            if existing_numbers:
                self.saved_count = max(existing_numbers) + 1
                self.get_logger().info(
                    "Found existing images. "
                    f"Setting saved_count to {self.saved_count}"
                )
            else:
                self.saved_count = 0

        self.bridge = CvBridge()
        self.frame_count = 0
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            10,
        )

    def image_callback(self, msg: Image) -> None:
        """
        Callback for each received image message. Saves only every nth image as a
        JPG file in the specified folder.
        """
        self.frame_count += 1
        if self.frame_count % self.n_th_frame != 0:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        filename = os.path.join(
            self.folder_path,
            f"image_{self.saved_count}.jpg",
        )
        cv2.imwrite(filename, cv_image)
        self.get_logger().info(f"Saved image: {filename}")
        self.saved_count += 1


def print_help() -> None:
    """
    Print help message for using the node.
    """
    help_message = (
        "Usage: ros2 run image_topic_to_image image_node [options]\n"
        "\n"
        "Options:\n"
        "  --folder_path FOLDER   Folder path to save images (default: /tmp/images)\n"
        "  --image_topic TOPIC    Image topic name (default: image_topic)\n"
        "  --n_th_frame N         Save every Nth frame (default: 15)\n"
        "  -h, --help             Show this help message and exit\n"
    )
    print(help_message)


def main(args=None) -> None:
    """
    Entry point for the ImageSaverNode.
    """
    if args is None:
        args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        print_help()
        return

    rclpy.init(args=args)
    node = ImageSaverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
