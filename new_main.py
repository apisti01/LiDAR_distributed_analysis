import time
import os
import numpy as np
import pandas as pd
import open3d as o3d
from sensor_selection import select_sensors
from data_loader import load_file, load_point_clouds_from_sensors
from transform_coordinates import load_and_transform_scan, calculate_sensors_centroid
from clustering import dbscan_clustering, create_bounding_boxes, associate_ids_to_bboxes
from simulation import update_visualization, create_cylinder_between_points
from tracking import track_vehicles, calculate_threshold, calculate_mse, KalmanFilter

# Setup paths
current_directory = os.path.dirname(os.path.abspath(__file__))
point_cloud_directory = os.path.join(current_directory, 'filtered_sensors_data')
sensors_positions_path = os.path.join(current_directory, 'sensors_positions/pitt_sensor_positions.csv')
trajectories_path = os.path.join(current_directory, 'trajectories/pitt_trajectories.csv')
predicted_trajectories_file_path = os.path.join(current_directory, 'output/predicted_trajectories.csv')
video_frames_directory = os.path.join(current_directory, 'output/video_frames')

# Ensure output directories exist
os.makedirs(os.path.dirname(predicted_trajectories_file_path), exist_ok=True)
os.makedirs(video_frames_directory, exist_ok=True)

# Remove previous output file if it exists
if os.path.exists(predicted_trajectories_file_path):
    os.remove(predicted_trajectories_file_path)

# Constants
frequency = 10

# Choose total number of sensors available
print("Enter the total number of sensors available: ")
try:
    num_sensors = int(input())
except ValueError:
    print("Please enter a valid number.")
    exit(1)

# Select the sensors to use
selected_sensors = select_sensors(num_sensors)

# Load the sensors positions
sensors_positions_df = load_file(sensors_positions_path)

# Load the trajectories
trajectories_df = load_file(trajectories_path)

# Calculate the threshold for the tracking algorithm
tracking_threshold = calculate_threshold(trajectories_df, frequency, percentage_margin=25)

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


# Function to generate a random color
def generate_random_color():
    return np.random.uniform(0, 1, 3)


# Function to combine trajectory data from all sensors
def combine_sensor_trajectories(sensor_trajectories):
    """
    Combine trajectory data from all sensors using averaging.

    Args:
        sensor_trajectories: Dictionary of dictionaries with trajectories by sensor and vehicle

    Returns:
        Dictionary with combined trajectories for each vehicle
    """
    combined = {}

    # Get all unique vehicle IDs across all sensors
    all_vehicle_ids = set()
    for sensor_id in sensor_trajectories:
        all_vehicle_ids.update(sensor_trajectories[sensor_id].keys())

    # For each vehicle, average its position from all sensors that tracked it
    for vehicle_id in all_vehicle_ids:
        points_by_frame = {}

        # Collect points from all sensors for this vehicle
        for sensor_id in sensor_trajectories:
            if vehicle_id in sensor_trajectories[sensor_id]:
                for frame_idx, point in enumerate(sensor_trajectories[sensor_id][vehicle_id]):
                    if frame_idx not in points_by_frame:
                        points_by_frame[frame_idx] = []
                    points_by_frame[frame_idx].append(point)

        # Average the points for each frame
        combined_points = []
        for frame_idx in sorted(points_by_frame.keys()):
            if points_by_frame[frame_idx]:
                # Calculate average point
                avg_point = np.mean(points_by_frame[frame_idx], axis=0)
                # Calculate variance if more than one sensor detected this vehicle
                if len(points_by_frame[frame_idx]) > 1:
                    variance = np.var(points_by_frame[frame_idx], axis=0)
                    print(f"Vehicle {vehicle_id}, Frame {frame_idx}: Variance: {variance}")
                combined_points.append(avg_point)

        if combined_points:
            combined[vehicle_id] = combined_points

    return combined


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
        pcd = pcd.voxel_down_sample(voxel_size=0.2)
        print(f"Number of points after downsampling: {len(pcd.points)}")

        # Perform DBSCAN clustering
        clusters, labels = dbscan_clustering(pcd, scan_idx)

        # Create bounding boxes
        bounding_boxes, bbox_centroids = create_bounding_boxes(clusters, scan_idx)

        # Associate object IDs to bounding boxes
        bbox_ids = associate_ids_to_bboxes(bbox_centroids, object_ids, transformed_xyz)

        # Track vehicles using Kalman filter
        if sensor_prev_bbox_centroids[sensor_id]:
            matches, exited_vehicles, entered_vehicles, predicted_centroids = track_vehicles(
                sensor_prev_bbox_centroids[sensor_id],
                bbox_centroids,
                sensor_prev_ids[sensor_id],
                bbox_ids,
                tracking_threshold,
                frequency
            )

            print(f"Sensor {sensor_id} - Matches: {matches}")
            print(f"Sensor {sensor_id} - Exited vehicles: {exited_vehicles}")
            print(f"Sensor {sensor_id} - Newly entered vehicles: {entered_vehicles}")

            # Update trajectories for this sensor
            for prev_id, current_id in matches:
                if current_id in bbox_ids:
                    current_centroid = bbox_centroids[bbox_ids.index(current_id)]
                    # If the vehicle ID is not in the dictionary, add it
                    if prev_id not in sensor_trajectories[sensor_id]:
                        sensor_trajectories[sensor_id][prev_id] = []

                    # Append the current centroid to the trajectory
                    sensor_trajectories[sensor_id][prev_id].append(current_centroid)

        # Update previous IDs and centroids for this sensor
        sensor_prev_ids[sensor_id] = bbox_ids
        sensor_prev_bbox_centroids[sensor_id] = bbox_centroids

    # Combine trajectories from all sensors
    combined_trajectories = combine_sensor_trajectories(sensor_trajectories)

    # Calculate MSE against ground truth trajectories
    predicted_trajectories_xy = {vehicle_id: [point[:2] for point in points] for vehicle_id, points in
                                 combined_trajectories.items()}
    calculate_mse(predicted_trajectories_xy, real_trajectories, tracking_threshold, scan_idx - 20)

    # Save predicted trajectories to CSV
    trajectory_data = []
    for vehicle_id, points in combined_trajectories.items():
        for point in points:
            trajectory_data.append([scan_idx, vehicle_id, *point])

    if trajectory_data:
        df_trajectories = pd.DataFrame(trajectory_data, columns=['scan', 'vehicle_id', 'x', 'y', 'z'])
        header = not os.path.exists(predicted_trajectories_file_path)
        df_trajectories.to_csv(predicted_trajectories_file_path, mode='a', header=header, index=False)

    # ----- Visualization -----
    # Combine all point clouds for visualization
    '''
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
                    points_3d = np.hstack((np.asarray(points), np.zeros((len(points), 1))))
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
        '''

    time.sleep(0.1)

# Close visualization
vis.destroy_window()
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
print("Processing complete.")