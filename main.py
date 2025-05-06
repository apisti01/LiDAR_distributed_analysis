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
from tracking import track_vehicles, calculate_threshold, calculate_mse

# Set paths
current_directory = os.path.dirname(os.path.abspath(__file__))
point_cloud_directory = os.path.join(current_directory, 'filtered_sensors_data')
sensors_positions_path = os.path.join(current_directory, 'sensors_positions/pitt_sensor_positions.csv')
trajectories_path = os.path.join(current_directory, 'trajectories/pitt_trajectories.csv')
predicted_trajectories_file_path = os.path.join(current_directory, 'output/predicted_trajectories.csv')
video_frames_directory = os.path.join(current_directory, 'output/video_frames')
sensor_tracking_directory = os.path.join(current_directory, 'output/sensor_tracking')

# Create output directories if they don't exist
if not os.path.exists(video_frames_directory):
    os.makedirs(video_frames_directory)
if not os.path.exists(sensor_tracking_directory):
    os.makedirs(sensor_tracking_directory)

# Remove output file if it exists
if os.path.exists(predicted_trajectories_file_path):
    os.remove(predicted_trajectories_file_path)

frequency = 10

# Get user input for total number of sensors
print("Enter the total number of sensors available: ")
try:
    num_sensors = int(input())
except ValueError:
    print("Please enter a valid number.")
    exit(1)

# Select sensors to use
selected_sensors = select_sensors(num_sensors)

# Load the sensors positions and trajectories
sensors_positions_df = load_file(sensors_positions_path)
trajectories_df = load_file(trajectories_path)

# Calculate the centroid of the sensors
centroid = calculate_sensors_centroid(sensors_positions_df)

# Calculate threshold for tracking
tracking_threshold = calculate_threshold(trajectories_df, frequency, percentage_margin=25)

# Load the real trajectories from trajectories_df
real_trajectories = {}
for i in range(20, 71):
    current_trajectories = trajectories_df[trajectories_df['time'] == i-20]
    for _, row in current_trajectories.iterrows():
        vehicle_id = int(row['label']) if row['label'] != 'AV' else -1
        point = [row['x'], row['y']]
        transformed_point = np.array(point) - centroid[:2]
        if vehicle_id not in real_trajectories:
            real_trajectories[vehicle_id] = [transformed_point]
        else:
            real_trajectories[vehicle_id].append(transformed_point)

# Initialize visualization
vis = o3d.visualization.Visualizer()
vis.create_window()

# Dictionary to store vehicle colors for visualization
vehicle_colors = {}


# Function to generate a random color
def generate_random_color():
    return np.random.uniform(0, 1, 3)


# Dictionary to store tracking data for each sensor
sensor_tracking_data = {sensor_id: {} for sensor_id in selected_sensors}

# Process each time frame
for scan_number in range(20, 71):
    print(f"\n--- Processing scan {scan_number} ---")

    # Store tracking data for each sensor at this time frame
    current_frame_tracking = {}
    all_transformed_points = []
    all_object_ids = []

    # Process each sensor independently
    for sensor_id in selected_sensors:
        print(f"Processing sensor {sensor_id}")

        # Step 1: Load the scan for this sensor
        sensor_filename = f'sensor_{sensor_id}_{scan_number:02d}.csv'
        sensor_file_path = os.path.join(point_cloud_directory, sensor_filename)

        if not os.path.exists(sensor_file_path):
            print(f"File not found: {sensor_file_path}")
            continue

        # Step 2: Transform coordinates to global frame
        transformed_xyz, object_ids = load_and_transform_scan(
            sensor_file_path,
            sensors_positions_df,
            centroid,
            sensor_id
        )

        if transformed_xyz is None or len(transformed_xyz) == 0:
            print(f"No data for sensor {sensor_id}")
            continue

        # Store transformed points for later visualization
        all_transformed_points.append(transformed_xyz)
        all_object_ids.extend(object_ids)

        # Step 3: Create point cloud for clustering
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed_xyz)
        print("Number of points before downsampling: ", len(pcd.points))

        # Downsample the point cloud for clustering
        pcd = pcd.voxel_down_sample(voxel_size=0.2) #TODO adjust voxel size
        print("Number of points after downsampling: ", len(pcd.points))

        # Step 4: Perform DBSCAN clustering
        clusters, labels = dbscan_clustering(pcd, scan_number)

        # Step 5: Create bounding boxes for detected objects
        bounding_boxes, bbox_centroids = create_bounding_boxes(clusters, scan_number)

        # Step 6: Associate IDs to bounding boxes
        bbox_ids = associate_ids_to_bboxes(bbox_centroids, object_ids, transformed_xyz)

        # Store tracking info for this sensor
        sensor_data = {
            'centroids': bbox_centroids,
            'ids': bbox_ids
        }

        # Save sensor tracking data to file
        sensor_tracking_file = os.path.join(
            sensor_tracking_directory,
            f"sensor_{sensor_id}_scan_{scan_number}.csv"
        )

        # Create DataFrame for sensor tracking data
        sensor_df = pd.DataFrame({
            'centroid_x': [c[0] for c in bbox_centroids],
            'centroid_y': [c[1] for c in bbox_centroids],
            'centroid_z': [c[2] for c in bbox_centroids],
            'object_id': bbox_ids
        })

        # Save to CSV
        sensor_df.to_csv(sensor_tracking_file, index=False)

        # Add to current frame tracking data
        current_frame_tracking[sensor_id] = sensor_data

    # Merge all sensor data for this frame
    merged_centroids = []
    merged_ids = []

    # Combine all detection data from all sensors
    for sensor_id, data in current_frame_tracking.items():
        merged_centroids.extend(data['centroids'])
        merged_ids.extend(data['ids'])

    # Check if we have previous frame data for tracking
    if scan_number > 20:
        # Get previous frame merged data
        prev_frame_file = os.path.join(current_directory, 'output/merged_frame.csv')
        if os.path.exists(prev_frame_file):
            prev_frame = pd.read_csv(prev_frame_file)
            prev_centroids = prev_frame[['centroid_x', 'centroid_y', 'centroid_z']].values
            prev_ids = prev_frame['object_id'].values.tolist()

            # Track vehicles between frames
            matches, exited_vehicles, entered_vehicles, predicted_centroids = track_vehicles(
                prev_centroids,
                merged_centroids,
                prev_ids,
                merged_ids,
                tracking_threshold,
                frequency
            )

            print("Matches:", len(matches))
            print("Exited vehicles:", len(exited_vehicles))
            print("Entered vehicles:", len(entered_vehicles))

            # Update trajectory data
            predicted_trajectories = {}
            for prev_id, current_id in matches:
                if current_id in merged_ids:
                    current_centroid = merged_centroids[merged_ids.index(current_id)]
                    if prev_id not in predicted_trajectories:
                        predicted_trajectories[prev_id] = []
                    predicted_trajectories[prev_id].append(current_centroid)

            # Save predicted trajectories to CSV
            trajectory_data = []
            for vehicle_id, points in predicted_trajectories.items():
                for point in points:
                    trajectory_data.append([scan_number, vehicle_id, *point])

            df_trajectories = pd.DataFrame(
                trajectory_data,
                columns=['scan', 'vehicle_id', 'x', 'y', 'z']
            )
            df_trajectories.to_csv(
                predicted_trajectories_file_path,
                mode='a',
                header=not os.path.exists(predicted_trajectories_file_path),
                index=False
            )

            # Calculate MSE
            predicted_trajectories_xy = {vehicle_id: [point[:2] for point in points]
                                         for vehicle_id, points in predicted_trajectories.items()}
            calculate_mse(predicted_trajectories_xy, real_trajectories, tracking_threshold, scan_number - 20)

    # Save merged data for next frame
    merged_frame = pd.DataFrame({
        'centroid_x': [c[0] for c in merged_centroids],
        'centroid_y': [c[1] for c in merged_centroids],
        'centroid_z': [c[2] for c in merged_centroids],
        'object_id': merged_ids
    })
    merged_frame.to_csv(os.path.join(current_directory, 'output/merged_frame.csv'), index=False)

    # Visualization
    # Combine all transformed points
    all_transformed_points = np.vstack(all_transformed_points) if all_transformed_points else np.array([])

    if len(all_transformed_points) > 0:
        # Create a combined point cloud
        pcd_combined = o3d.geometry.PointCloud()
        pcd_combined.points = o3d.utility.Vector3dVector(all_transformed_points)

        # Downsample for visualization efficiency
        pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.3)

        # Visualize the point cloud and bounding boxes
        # For simplicity, we'll recalculate bounding boxes from merged data
        merged_points = np.array([merged_centroids])
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points.reshape(-1, 3))

        # Create visualization elements
        vis_elements = []

        # Visualize detection points
        for centroid in merged_centroids:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            sphere.translate(centroid)
            sphere.paint_uniform_color([1, 0, 0])  # Red for detection points
            vis_elements.append(sphere)

        # Update visualization
        update_visualization(vis, pcd_combined, vis_elements)

        # Save screenshot
        screenshot_path = os.path.join(video_frames_directory, f"frame_{scan_number - 20:04d}.png")
        vis.capture_screen_image(screenshot_path)

    time.sleep(0.1)

# Close visualization
vis.destroy_window()
print("Processing complete!")