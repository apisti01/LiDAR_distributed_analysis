import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import logging

from kalman_filter import KalmanFilter

logger = logging.getLogger()

# Modified to accept a custom kalman_filters dictionary
def track_vehicles(prev_centroids, curr_centroids, prev_ids, curr_ids, threshold, sensor_frequency,
                   kalman_filters):
    delta_time = 1 / sensor_frequency  # Time step in seconds

    # Initialize Kalman filters for vehicles in prev_ids if not already initialized
    for vehicle_id in prev_ids:
        if vehicle_id not in kalman_filters:
            kf = KalmanFilter(delta_time)
            # Initialize with the previous position and zero velocity/acceleration
            kf.X[:3] = prev_centroids[prev_ids.index(vehicle_id)]  # Initial position [x, y, z]
            kf.X[3:5] = np.zeros(2)  # Initial velocity [vx, vy]
            kf.X[5:] = np.zeros(2)  # Initial acceleration [ax, ay]
            kalman_filters[vehicle_id] = kf

    # Predict the next position of each vehicle using Kalman Filter
    predicted_centroids = []
    for i, vehicle_id in enumerate(prev_ids):
        if vehicle_id in kalman_filters:
            kf = kalman_filters[vehicle_id]
            kf.predict()  # Predict the next state
            predicted_centroids.append(kf.get_state()[:3])  # Append the predicted position [x, y, z]

    # Compute distance matrix between predicted centroids and current centroids
    distance_matrix = compute_distance_matrix(predicted_centroids, curr_centroids)

    # Solve the linear assignment problem
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    matches = []
    unmatched_prev = set(range(len(prev_centroids)))
    unmatched_curr = set(range(len(curr_centroids)))

    # Perform matching based on the assignment and threshold
    for r, c in zip(row_ind, col_ind):
        if distance_matrix[r, c] < threshold:
            matches.append((prev_ids[r], curr_ids[c]))
            unmatched_prev.discard(r)
            unmatched_curr.discard(c)

            # Update the Kalman filter for the matched vehicle
            kf = kalman_filters[prev_ids[r]]
            prev_pos = prev_centroids[r]
            curr_pos = curr_centroids[c]
            prev_velocity = kf.X[3:5]
            curr_velocity = compute_velocity(prev_pos, curr_pos, delta_time)
            acceleration = compute_acceleration(prev_velocity, curr_velocity, delta_time)

            # Update the Kalman filter with the actual measurement
            kf.update(curr_pos)
            # Update velocity and acceleration in the Kalman filter's state vector
            kf.X[3:5] = curr_velocity
            kf.X[5:] = acceleration

    # Handle unmatched vehicles
    exited_vehicles = [prev_ids[i] for i in unmatched_prev]
    entered_vehicles = [curr_ids[i] for i in unmatched_curr]

    # Remove Kalman filters for exited vehicles
    for vehicle_id in exited_vehicles:
        if vehicle_id in kalman_filters:
            del kalman_filters[vehicle_id]

    # Initialize Kalman filters for new vehicles
    for vehicle_id in entered_vehicles:
        if vehicle_id in curr_ids:  # Make sure it exists in curr_ids
            idx = curr_ids.index(vehicle_id)
            curr_pos = curr_centroids[idx]
            kf = KalmanFilter(delta_time)
            kf.X[:3] = curr_pos  # Set the initial position [x, y, z]
            kf.X[3:5] = np.zeros(2)  # Initial velocity [vx, vy]
            kf.X[5:] = np.zeros(2)  # Initial acceleration [ax, ay]
            kalman_filters[vehicle_id] = kf

    states_of_bbox = []
    for i, vehicle_id in enumerate(curr_ids):
        if vehicle_id in kalman_filters:
            kf = kalman_filters[vehicle_id]
            states_of_bbox.append(kf.get_state()[:3])
        else:
            # Handle the rare cases of a mismatch on an exited vehicle
            states_of_bbox.append([np.inf, np.inf, np.inf])  # Placeholder

    return matches, exited_vehicles, entered_vehicles, states_of_bbox


def compute_velocity(curr_position, predicted_position, delta_time):
    """
    Calculate the velocity based on the previous and current positions and the elapsed time.
    """
    vx = (predicted_position[0] - curr_position[0]) / delta_time
    vy = (predicted_position[1] - curr_position[1]) / delta_time
    return np.array([vx, vy])


def compute_acceleration(curr_velocity, predicted_velocity, delta_time):
    """
    Calculate the acceleration based on the previous and current velocities and the elapsed time.
    """
    ax = (predicted_velocity[0] - curr_velocity[0]) / delta_time
    ay = (predicted_velocity[1] - curr_velocity[1]) / delta_time
    return np.array([ax, ay])


def compute_distance_matrix(prev_boxes, curr_boxes):
    distance_matrix = np.zeros((len(prev_boxes), len(curr_boxes)))

    for i, prev in enumerate(prev_boxes):
        for j, curr in enumerate(curr_boxes):
            distance_matrix[i, j] = np.linalg.norm(np.array(prev) - np.array(curr))

    return distance_matrix


def calculate_threshold(df, sensor_frequency, percentage_margin):
    vx = df['vx']
    vy = df['vy']
    v_max = (np.sqrt(pow(vx, 2) + pow(vy, 2))).max()
    threshold = v_max * (1 / sensor_frequency)
    return threshold + threshold * (percentage_margin / 100)


def calculate_mse(real_trajectories, combined_trajectories, num_frames=30):
    """
    Calculate MSE between real and combined trajectories using linear sum assignment
    for trajectory matching and evaluate over multiple frames.

    Args:
        real_trajectories: Dictionary of real trajectories {vehicle_id: [position_points]}
        combined_trajectories: Dictionary of predicted trajectories {vehicle_id: [position_points]}
        tracking_threshold: Maximum distance for matching trajectories
        num_frames: Number of frames to analyze
    """


    # Extract valid trajectory IDs (with at least one point)
    real_ids = [id for id, traj in real_trajectories.items() if len(traj) > 0]
    pred_ids = [id for id, traj in combined_trajectories.items() if len(traj) > 0]

    if not real_ids or not pred_ids:
        print("No valid trajectories to compare")
        return {}

    # Get first points from each trajectory for initial matching
    real_first_points = [real_trajectories[id][20] for id in real_ids]
    pred_first_points = [combined_trajectories[id][0] for id in pred_ids]

    # Create distance matrix between first points
    distance_matrix = compute_distance_matrix(real_first_points, pred_first_points)

    # Use linear sum assignment to find optimal matching
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Set up data structures for MSE calculation
    matched_ids = {}
    mse_values = {}
    frame_mse = {i: [] for i in range(num_frames)}
    mse_per_frame = {}

    # Process matches based on assignment
    for r, c in zip(row_ind, col_ind):
        real_id = real_ids[r]
        pred_id = pred_ids[c]

        matched_ids[real_id] = pred_id
        mse_per_frame[real_id] = []

        real_traj = real_trajectories[real_id]
        pred_traj = combined_trajectories[pred_id]

        # Calculate MSE for each frame (up to num_frames)
        for frame in range(num_frames):
            if frame < len(real_traj)-20 and frame < len(pred_traj):
                real_point = np.array(real_traj[frame+20])
                pred_point = np.array(pred_traj[frame])

                squared_error = np.sum((real_point - pred_point) ** 2)
                mse_per_frame[real_id].append(squared_error)
                frame_mse[frame].append(squared_error)

        if mse_per_frame[real_id]:
            avg_mse = np.mean(mse_per_frame[real_id])
            mse_values[(real_id, pred_id)] = avg_mse
            print(f"Matched Real ID {real_id} with Predicted ID {pred_id} - Avg MSE: {avg_mse:.4f}")
            logger.info(f"Matched Real ID {real_id} with Predicted ID {pred_id} - Avg MSE: {avg_mse:.4f}")
        else:
            print(f"No overlapping frames for Real ID {real_id} and Predicted ID {pred_id}")
            logger.warning(f"No overlapping frames for Real ID {real_id} and Predicted ID {pred_id}")

    # Calculate average MSE per frame
    avg_frame_mse = [np.mean(errors) if errors else 0 for frame, errors in sorted(frame_mse.items())]



    # Plot individual MSE per frame for each vehicle pair
    plt.figure(figsize=(14, 8))

    # Plot each vehicle's MSE over frames
    for real_id, mse_v_values in mse_per_frame.items():
        if mse_v_values:
            plt.plot(range(len(mse_v_values)), mse_v_values, marker='.', linestyle='-', alpha=0.7,
                    label=f'Vehicle {real_id}')

    # Plot the average MSE across all vehicles
    plt.plot(range(len(avg_frame_mse)), avg_frame_mse, marker='o', linewidth=3, color='black',
             label='Average MSE')

    plt.title('Mean Squared Error per Frame - All Vehicles')
    plt.xlabel('Frame Number')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.legend()
    plt.savefig('output/mse_per_frame_all_vehicles.png')
    plt.close()



    # Plot MSE per frame
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(avg_frame_mse)), avg_frame_mse, marker='o')
    plt.title('Mean Squared Error per Frame')
    plt.xlabel('Frame Number')
    plt.ylabel('Average MSE')
    plt.grid(True)
    plt.savefig('output/mse_per_frame.png')
    plt.show()



    # Plot MSE by vehicle pair
    if mse_values:
        vehicles = [f"{real_id}-{pred_id}" for (real_id, pred_id) in mse_values.keys()]
        mse_vals = list(mse_values.values())

        plt.figure(figsize=(14, 6))
        plt.bar(vehicles, mse_vals)
        plt.title('Average MSE by Vehicle Pair')
        plt.xlabel('Vehicle Pairs (Real ID - Predicted ID)')
        plt.ylabel('Average MSE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('output/mse_by_vehicle.png')
        plt.show()

    # Calculate overall average MSE
    overall_mse = np.mean(list(mse_values.values())) if mse_values else 0
    print(f"Overall average MSE across all matched trajectories: {overall_mse:.4f}")

    return mse_values