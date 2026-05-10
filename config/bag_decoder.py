#!/usr/bin/env python3

"""
RTAB-Map Tracking Health Analyzer
=================================

Reads a ROS2 bag and computes:

1. Inlier Ratio
2. Covariance Trace
3. TF Translational Jitter
4. TF Rotational Jitter
5. Feature Count
6. Lost Events

TOPICS USED
-----------

/rtabmap/odom_info
/tf
/tf_static   (optional)

NO IMAGE TOPICS REQUIRED.

------------------------------------------------------------
INSTALL
------------------------------------------------------------

sudo apt install python3-rosbag2-py
pip install numpy pandas matplotlib scipy

------------------------------------------------------------
USAGE
------------------------------------------------------------

python3 analyze_tracking_health.py /path/to/rosbag

Example:

python3 analyze_tracking_health.py ~/bags/office_run

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

tracking_health_metrics.csv
tracking_health_plots.png

"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

from rosbag2_py import SequentialReader
from rosbag2_py import StorageOptions
from rosbag2_py import ConverterOptions

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ============================================================
# PARAMETERS
# ============================================================

WINDOW_SIZE = 20

TARGET_PARENT = "odom"
TARGET_CHILD = "base_link"


# ============================================================
# HELPERS
# ============================================================

def rotation_angle(q1, q2):
    """
    Quaternion angular difference.
    """

    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)

    r_rel = r1.inv() * r2

    return r_rel.magnitude()


def sliding_variance(signal, window):

    out = np.zeros(len(signal))

    for i in range(len(signal)):

        start = max(0, i - window + 1)

        out[i] = np.var(signal[start:i + 1])

    return out


# ============================================================
# OPEN BAG
# ============================================================

if len(sys.argv) < 2:

    print("\nUsage:")
    print("python3 analyze_tracking_health.py /path/to/rosbag\n")

    sys.exit(1)

bag_path = sys.argv[1]

storage_options = StorageOptions(
    uri=bag_path,
    storage_id='sqlite3'
)

converter_options = ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)

reader = SequentialReader()

reader.open(
    storage_options,
    converter_options
)

topic_types = reader.get_all_topics_and_types()

type_map = {
    topic.name: topic.type
    for topic in topic_types
}


# ============================================================
# STORAGE
# ============================================================

odom_info_data = []

tf_times = []
tf_positions = []
tf_quaternions = []

print("\nReading rosbag...\n")


# ============================================================
# READ BAG
# ============================================================

while reader.has_next():

    topic, data, timestamp = reader.read_next()

    t = timestamp * 1e-9

    # --------------------------------------------------------
    # RTABMAP ODOM INFO
    # --------------------------------------------------------

    if topic == "/rtabmap/odom_info":

        msg_type = get_message(type_map[topic])

        msg = deserialize_message(data, msg_type)

        matches = msg.matches
        inliers = msg.inliers

        ratio = 0.0

        if matches > 0:
            ratio = inliers / matches

        cov = np.array(msg.covariance).reshape(6, 6)

        cov_trace = np.trace(cov)

        odom_info_data.append({

            "time": t,
            "inlier_ratio": ratio,
            "features": msg.features,
            "cov_trace": cov_trace,
            "lost": int(msg.lost)

        })

    # --------------------------------------------------------
    # TF
    # --------------------------------------------------------

    elif topic == "/tf":

        msg_type = get_message(type_map[topic])

        msg = deserialize_message(data, msg_type)

        for tf_msg in msg.transforms:

            parent = tf_msg.header.frame_id
            child = tf_msg.child_frame_id

            if parent == TARGET_PARENT and child == TARGET_CHILD:

                tr = tf_msg.transform.translation
                rot = tf_msg.transform.rotation

                tf_times.append(t)

                tf_positions.append([
                    tr.x,
                    tr.y,
                    tr.z
                ])

                tf_quaternions.append([
                    rot.x,
                    rot.y,
                    rot.z,
                    rot.w
                ])


print("Finished reading rosbag.\n")


# ============================================================
# CONVERT TO NUMPY
# ============================================================

tf_positions = np.array(tf_positions)
tf_quaternions = np.array(tf_quaternions)
tf_times = np.array(tf_times)

odom_df = pd.DataFrame(odom_info_data)

if odom_df.empty:

    print(
        "Error: No /rtabmap/odom_info messages in bag. "
        "Cannot produce tracking health metrics.\n"
    )
    sys.exit(1)

# Normalize time (TF may be missing if bag has no odom->base_link /tf)
t0_candidates = [odom_df["time"].iloc[0]]
if tf_times.size > 0:
    t0_candidates.append(float(tf_times.flat[0]))
t0 = min(t0_candidates)

odom_df["time"] -= t0
if tf_times.size > 0:
    tf_times = tf_times - t0


# ============================================================
# COMPUTE TF DELTAS
# ============================================================

if len(tf_positions) < 2:

    print(
        "Warning: Fewer than 2 /tf samples for frames "
        f"'{TARGET_PARENT}' -> '{TARGET_CHILD}'. "
        "TF jitter columns will be NaN. "
        "Record that TF topic or remap frames if you need jitter.\n"
    )

    merged = odom_df.sort_values("time").copy()
    merged["trans_jitter"] = np.nan
    merged["rot_jitter"] = np.nan

else:

    delta_trans = []
    delta_rot = []

    for i in range(1, len(tf_positions)):

        # Translation increment
        dtrans = np.linalg.norm(
            tf_positions[i] - tf_positions[i - 1]
        )

        delta_trans.append(dtrans)

        # Rotation increment
        drot = rotation_angle(
            tf_quaternions[i - 1],
            tf_quaternions[i]
        )

        delta_rot.append(drot)

    delta_trans = np.array(delta_trans)
    delta_rot = np.array(delta_rot)

    tf_times_jitter = tf_times[1:]

    # ============================================================
    # JITTER
    # ============================================================

    trans_jitter = sliding_variance(
        delta_trans,
        WINDOW_SIZE
    )

    rot_jitter = sliding_variance(
        delta_rot,
        WINDOW_SIZE
    )

    # ============================================================
    # MERGE ODOM + JITTER
    # ============================================================

    jitter_df = pd.DataFrame({

        "time": tf_times_jitter,
        "trans_jitter": trans_jitter,
        "rot_jitter": rot_jitter

    })

    merged = pd.merge_asof(
        odom_df.sort_values("time"),
        jitter_df.sort_values("time"),
        on="time"
    )

csv_path = "tracking_health_metrics.csv"

merged.to_csv(csv_path, index=False)

print(f"Saved CSV: {csv_path}")


# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(
    5,
    1,
    figsize=(15, 18),
    sharex=True
)

# ------------------------------------------------------------
# Inlier Ratio
# ------------------------------------------------------------

axes[0].plot(
    merged["time"],
    merged["inlier_ratio"]
)

axes[0].set_ylabel("Inlier Ratio")
axes[0].set_title("Tracking Quality")


# ------------------------------------------------------------
# Covariance Trace
# ------------------------------------------------------------

axes[1].plot(
    merged["time"],
    merged["cov_trace"]
)

axes[1].set_ylabel("Cov Trace")
axes[1].set_title("Estimator Uncertainty")


# ------------------------------------------------------------
# Translational Jitter
# ------------------------------------------------------------

axes[2].plot(
    merged["time"],
    merged["trans_jitter"]
)

axes[2].set_ylabel("Trans Jitter")
axes[2].set_title("TF Translational Wobble")


# ------------------------------------------------------------
# Rotational Jitter
# ------------------------------------------------------------

axes[3].plot(
    merged["time"],
    merged["rot_jitter"]
)

axes[3].set_ylabel("Rot Jitter")
axes[3].set_title("TF Rotational Wobble")


# ------------------------------------------------------------
# Feature Count
# ------------------------------------------------------------

axes[4].plot(
    merged["time"],
    merged["features"]
)

axes[4].set_ylabel("Features")
axes[4].set_xlabel("Time [s]")
axes[4].set_title("Feature Count")


# ============================================================
# LOST EVENT MARKERS
# ============================================================

for ax in axes:

    for _, row in merged.iterrows():

        if row["lost"] == 1:

            ax.axvline(
                row["time"],
                linestyle='--'
            )


plt.tight_layout()

plot_path = "tracking_health_plots.png"

plt.savefig(
    plot_path,
    dpi=300
)

print(f"Saved plot: {plot_path}")

plt.show()

print("\nDone.\n")