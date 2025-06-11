import time
import os
import numpy as np
import pandas as pd
import open3d as o3d
import logging

from combining import combine_sensor_kalman_filters
from sensor_selection import select_sensors
from data_loader import load_file
from transform_coordinates import load_and_transform_scan, calculate_sensors_centroid
from clustering import dbscan_clustering, create_bounding_boxes, associate_ids_to_bboxes
from simulation import update_visualization, create_cylinder_between_points
from tracking import track_vehicles, calculate_threshold, calculate_mse, compare_mses
from trajectories_handler import add_new_point_to_trajectories

# Setup paths
current_directory = os.path.dirname(os.path.abspath(__file__))
point_cloud_directory = os.path.join(current_directory, 'filtered_sensors_data')
sensors_positions_path = os.path.join(current_directory, 'sensors_positions/pitt_sensor_positions.csv')
trajectories_path = os.path.join(current_directory, 'trajectories/pitt_trajectories.csv')
predicted_trajectories_file_path = os.path.join(current_directory, 'output/predicted_trajectories.csv')
video_frames_directory = os.path.join(current_directory, 'output/video_frames')
sensor_trajectory_file_path = os.path.join(current_directory, 'output/sensor_trajectories.csv')

# Ensure output directories exist
os.makedirs(os.path.dirname(predicted_trajectories_file_path), exist_ok=True)
os.makedirs(video_frames_directory, exist_ok=True)

# Remove previous output file if it exists
if os.path.exists(predicted_trajectories_file_path):
    os.remove(predicted_trajectories_file_path)

# Setup logging
log_file_path = os.path.join(current_directory, 'output/program.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    filemode='w'  # Overwrite the log file each time the program runs
)
logger = logging.getLogger()

# Constants
frequency = 10

'''
# Choose total number of sensors available
print("Enter the total number of sensors available: ")
try:
    num_sensors = int(input())
except ValueError:
    print("Please enter a valid number.")
    exit(1)

# Select the sensors to use
selected_sensors = select_sensors(num_sensors) '''
# For testing purposes, we can hardcode the selected sensors
selected_sensors = [0, 1, 2, 3, 4]

# Load the sensors positions
sensors_positions_df = load_file(sensors_positions_path)

# Load the trajectories
trajectories_df = load_file(trajectories_path)

# Calculate the threshold for the tracking algorithm
tracking_threshold = calculate_threshold(trajectories_df, frequency, percentage_margin=100)

# Calculate the centroid of the sensors
centroid = calculate_sensors_centroid(sensors_positions_df)

# Initialize visualization
vis = o3d.visualization.Visualizer()
vis.create_window()

# Dictionary to store kalman filters for each sensor and vehicle
sensor_kalman_filters = {sensor_id: {} for sensor_id in selected_sensors}
# Dictionary to store trajectories for each sensor and vehicle
sensor_trajectories = {sensor_id: {} for sensor_id in selected_sensors}
# Dictionary to store previous ids and bounding box centroids for each sensor
sensor_prev_ids = {sensor_id: [] for sensor_id in selected_sensors}
sensor_prev_bbox_centroids = {sensor_id: [] for sensor_id in selected_sensors}

# Dictionary to store combined trajectories after merging data from all sensors
combined_trajectories = {}
tracked_speeds = {}

# Dictionary to store vehicle colors
vehicle_colors = {}

# Load the real trajectories from trajectories_df
real_trajectories = {}
for i in range(20, 71):
    # Filter the trajectories for the current timestep
    current_trajectories = trajectories_df[trajectories_df['time'] == i-20]
    for _, row in current_trajectories.iterrows():
        vehicle_id = int(row['label']) if row['label'] != 'AV' else -1
        point = [row['x'], row['y']]

        # Transform the trajectory point
        transformed_point = np.array(point) - centroid[:2]
        # Check if this vehicle_id already exists in the dictionary
        if vehicle_id not in real_trajectories:
            real_trajectories[vehicle_id] = [transformed_point]  # Start a list for this vehicle
        else:
            real_trajectories[vehicle_id].append(transformed_point)  # Append the point

#Load the real speed from trajectories_df
real_speeds = {}
for i in range(20, 71):
    # Filter the trajectories for the current timestep
    current_trajectories = trajectories_df[trajectories_df['time'] == i-20]
    for _, row in current_trajectories.iterrows():
        vehicle_id = int(row['label']) if row['label'] != 'AV' else -1
        point = [row['vx'], row['vy']]

        # Transform the trajectory point
        transformed_point = np.array(point)
        # Check if this vehicle_id already exists in the dictionary
        if vehicle_id not in real_speeds:
            real_speeds[vehicle_id] = [transformed_point]  # Start a list for this vehicle
        else:
            real_speeds[vehicle_id].append(transformed_point)  # Append the point


# Function to generate a random color
def generate_random_color():
    return np.random.uniform(0, 1, 3)

# Initialize the frame index for saving screenshots
frame_index = 0

# Loop through scan indices from 20 to 70
for scan_idx in range(20, 71):
    print(f"\n--- Processing scan {scan_idx} ---")

    # Process each sensor separately
    for sensor_idx, sensor_id in enumerate(selected_sensors):
        print(f"Processing sensor {sensor_id}")

        # Get the sensor scan file
        sensor_scan = os.path.join(point_cloud_directory, f'sensor_{sensor_id}_{scan_idx:02d}.csv')
        if not os.path.exists(sensor_scan):
            print(f"Scan file not found: {sensor_scan}")
            continue

        # Load and transform the scan
        transformed_xyz, object_ids = load_and_transform_scan(sensor_scan, sensors_positions_df, centroid, sensor_id)

        if transformed_xyz is None or len(transformed_xyz) == 0:
            print(f"No valid data for sensor {sensor_id}, scan {scan_idx}")
            continue

        # Create a point cloud from the transformed points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed_xyz)

        # Downsample the point cloud
        print(f"Number of points before downsampling: {len(pcd.points)}")
        pcd = pcd.voxel_down_sample(voxel_size=0.2) #TODO find optimal value if any
        print(f"Number of points after downsampling: {len(pcd.points)}")

        # Perform DBSCAN clustering
        clusters, labels = dbscan_clustering(pcd, scan_idx)

        # Create bounding boxes
        bounding_boxes, bbox_centroids = create_bounding_boxes(clusters, scan_idx)

        # Associate object IDs to bounding boxes
        bbox_ids = associate_ids_to_bboxes(bbox_centroids, object_ids, transformed_xyz)

        # Track vehicles using Kalman filter
        if sensor_prev_bbox_centroids[sensor_id]:
            matches, exited_vehicles, entered_vehicles, kalman_states = track_vehicles(
                sensor_prev_bbox_centroids[sensor_id],
                bbox_centroids,
                sensor_prev_ids[sensor_id],
                bbox_ids,
                tracking_threshold,
                frequency,
                sensor_kalman_filters[sensor_id]
            )

            if matches:
                print(f"Sensor {sensor_id} - Matches: {pd.DataFrame(matches)}")
                for prev_id, current_id in matches:
                    if prev_id != current_id:
                        logger.error(f"Scan: {scan_idx}, Sensor {sensor_id} - {prev_id} != {current_id}")
            if exited_vehicles:
                print(f"Sensor {sensor_id} - Exited vehicles: {pd.DataFrame(exited_vehicles)}")
                logger.info(f"Scan: {scan_idx}, Sensor {sensor_id} - Exited vehicles: {exited_vehicles}")
            if entered_vehicles:
                print(f"Sensor {sensor_id} - Newly entered vehicles: {pd.DataFrame(entered_vehicles)}")
                logger.info(f"Scan: {scan_idx}, Sensor {sensor_id} - Newly entered vehicles: {entered_vehicles}")


            # Update trajectories for this sensor
            for prev_id, current_id in matches:
                if current_id in bbox_ids:
                    current_centroid = kalman_states[bbox_ids.index(current_id)]
                    # If the vehicle ID is not in the dictionary, add it
                    if prev_id not in sensor_trajectories[sensor_id]:
                        sensor_trajectories[sensor_id][prev_id] = []

                    # Append the current centroid to the trajectory
                    sensor_trajectories[sensor_id][prev_id].append(current_centroid)


        # Update previous IDs and centroids for this sensor
        sensor_prev_ids[sensor_id] = bbox_ids
        sensor_prev_bbox_centroids[sensor_id] = bbox_centroids

    # Combine trajectories from all sensors
    combined_kalman_filters = combine_sensor_kalman_filters(sensor_kalman_filters, selected_sensors,30 ) #TODO adjust max distance if needed

    if len(combined_kalman_filters) != 6:
        logger.warning(f"Scan: {scan_idx}, strange stuff with the vehicles: length is: {len(combined_kalman_filters)}")

    add_new_point_to_trajectories(combined_kalman_filters, combined_trajectories, tracked_speeds, tracking_threshold * 5)

    # ----- Visualization -----
    # Combine all point clouds for visualization
    all_transformed_xyz = []
    all_object_ids = []

    for sensor_id in selected_sensors:
        sensor_scan = os.path.join(point_cloud_directory, f'sensor_{sensor_id}_{scan_idx:02d}.csv')
        if os.path.exists(sensor_scan):
            transformed_xyz, object_ids = load_and_transform_scan(sensor_scan, sensors_positions_df, centroid,
                                                                  sensor_id)
            if transformed_xyz is not None and len(transformed_xyz) > 0:
                all_transformed_xyz.append(transformed_xyz)
                all_object_ids.extend(object_ids)

    if all_transformed_xyz:
        # Convert to numpy arrays for easier handling
        all_transformed_xyz = np.vstack(all_transformed_xyz)

        # Create combined point cloud for visualization
        pcd_combined = o3d.geometry.PointCloud()
        pcd_combined.points = o3d.utility.Vector3dVector(all_transformed_xyz)
        pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.3)

        # Create visualization elements
        vis_elements = []

        # Get clusters for visualization (reuse clustering code)
        clusters, _ = dbscan_clustering(pcd_combined, scan_idx)
        bboxes, _ = create_bounding_boxes(clusters, scan_idx)
        vis_elements.extend(bboxes)

        # Draw trajectories
        all_trajectories = {**combined_trajectories, **real_trajectories}

        for vehicle_id, points in all_trajectories.items():
            if len(points) > 1:  # Ensure at least two time steps
                # Check if points are 2D or 3D
                if len(points[0]) == 2:
                    # Convert 2D points to 3D by adding a z coordinate (set to 0)
                    points_array = np.asarray(points)
                    points_3d = np.column_stack((points_array[:, 0], np.zeros(len(points)), -points_array[:, 1]))
                else:
                    points_3d = np.asarray(points)

                # Generate a color for this vehicle if not already assigned
                if vehicle_id not in vehicle_colors:
                    vehicle_colors[vehicle_id] = generate_random_color()

                color = vehicle_colors[vehicle_id]

                for idx in range(len(points_3d) - 1):
                    point1 = points_3d[idx]
                    point2 = points_3d[idx + 1]
                    # Create a cylinder between consecutive points
                    cylinder = create_cylinder_between_points(point1, point2, color, radius=0.1)
                    vis_elements.append(cylinder)

        # Update visualization

        update_visualization(vis, pcd_combined, vis_elements)

        # Save screenshot
        screenshot_path = os.path.join(video_frames_directory, f"frame_{frame_index:04d}.png")
        vis.capture_screen_image(screenshot_path)
        frame_index += 1


# Calculate MSE for the combined trajectories
mses, avg_frame_mse = calculate_mse(real_trajectories, combined_trajectories, tracked_speeds, real_speeds)
print(f"MSE values: {mses}")
logger.info(f"MSE values: {mses}")
compare_mses(real_trajectories, sensor_trajectories, selected_sensors, mses, avg_frame_mse)

# Save predicted trajectories to CSV
trajectory_data = []
for vehicle_id, points in combined_trajectories.items():
    for n, point in enumerate(points):
        if point[0] is not np.inf:
            trajectory_data.append([n + 20, vehicle_id, *point])

if trajectory_data:
    df_trajectories = pd.DataFrame(trajectory_data, columns=['scan', 'vehicle_id', 'x', 'y'])
    df_trajectories.to_csv(predicted_trajectories_file_path, mode='a', header=True, index=False)

# Save sensor trajectories to CSV
sensor_trajectory_data = []

for sensor_id, trajectories in sensor_trajectories.items():
    for vehicle_id, points in trajectories.items():
        for scan_num, point in enumerate(points):
            # Check if the point contains valid data
            if point[0] is not np.inf:
                # Add sensor_id, vehicle_id, scan number, and position coordinates
                sensor_trajectory_data.append([sensor_id, vehicle_id, scan_num + 20, point[0], point[1], point[2]])

if sensor_trajectory_data:
    df_sensor_trajectories = pd.DataFrame(
        sensor_trajectory_data,
        columns=['sensor_id', 'vehicle_id', 'scan', 'x', 'y', 'z']
    )
    df_sensor_trajectories.to_csv(sensor_trajectory_file_path, index=False)


# Close visualization
vis.destroy_window()
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
print("Processing complete.")