import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial.distance import cdist

from kalman_filter import KalmanFilter


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

    # Remove Kalman filters for exited vehicles
    for vehicle_id in exited_vehicles:
        if vehicle_id in kalman_filters:
            del kalman_filters[vehicle_id]

    return matches, exited_vehicles, entered_vehicles, predicted_centroids


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


def calculate_mse(predicted_trajectories, real_trajectories, tracking_threshold, scan):
    matched_ids = {}
    mse_values = {}

    # For each real trajectory, find the closest predicted trajectory within the threshold
    for real_id, real_traj in real_trajectories.items():
        min_distance = float('inf')
        matched_pred_id = None

        # Compare the real trajectory with each predicted trajectory
        for pred_id, pred_traj in predicted_trajectories.items():
            # Calculate the distance between centroids or between entire trajectories
            if len(real_traj) > 0 and len(pred_traj) > 0:
                dist = distance.euclidean(np.mean(real_traj, axis=0), np.mean(pred_traj, axis=0))

                # If the distance is within the threshold, consider it for matching
                if dist < tracking_threshold and dist < min_distance:
                    min_distance = dist
                    matched_pred_id = pred_id

        # If a match is found, calculate the MSE and store the result
        if matched_pred_id is not None:
            matched_ids[real_id] = matched_pred_id

            if len(real_trajectories[real_id]) > scan and len(predicted_trajectories[matched_pred_id]) > (scan - 1):
                # Calculate the Mean Squared Error (MSE) between the matched trajectories
                real_traj_points = np.array(real_trajectories[real_id][scan])
                pred_traj_points = np.array(predicted_trajectories[matched_pred_id][scan - 1])

                mse = np.mean((real_traj_points - pred_traj_points) ** 2)

                mse_values[(real_id, matched_pred_id)] = mse
                print(f"Matched Real ID {real_id} with Predicted ID {matched_pred_id} - MSE: {mse:.4f}")
            else:
                print(
                    f"Skipping MSE calculation for Real ID {real_id} and Predicted ID {matched_pred_id} due to insufficient data.")

        else:
            print(f"No match found for Real ID {real_id}")

    return mse_values


# New function for associating trajectories from different sensors
def associate_trajectories(trajectories_dict, threshold):
    """
    Associate trajectories from different sensors that likely belong to the same vehicle.

    Args:
        trajectories_dict: Dictionary of dictionaries {sensor_id: {vehicle_id: trajectory}}
        threshold: Distance threshold for considering trajectories as the same vehicle

    Returns:
        Dictionary of associated trajectories with global IDs
    """
    # Flatten all trajectories
    all_trajectories = []
    for sensor_id, sensor_trajectories in trajectories_dict.items():
        for vehicle_id, trajectory in sensor_trajectories.items():
            all_trajectories.append({
                'sensor_id': sensor_id,
                'vehicle_id': vehicle_id,
                'trajectory': trajectory,
                'start_frame': trajectory[0]['scan'] if trajectory else None,
                'end_frame': trajectory[-1]['scan'] if trajectory else None
            })

    # Sort by start frame
    all_trajectories.sort(key=lambda x: x['start_frame'] if x['start_frame'] is not None else float('inf'))

    # Associate trajectories
    global_id = 0
    global_trajectories = {}
    assigned = set()

    for i, traj1 in enumerate(all_trajectories):
        if (traj1['sensor_id'], traj1['vehicle_id']) in assigned:
            continue

        # Create new global trajectory
        global_trajectories[global_id] = []
        assigned.add((traj1['sensor_id'], traj1['vehicle_id']))

        # Add points from this trajectory
        for point in traj1['trajectory']:
            global_trajectories[global_id].append(point)

        # Check for other trajectories to merge
        for j, traj2 in enumerate(all_trajectories):
            if i == j or (traj2['sensor_id'], traj2['vehicle_id']) in assigned:
                continue

            # Check if trajectories can be merged
            # Simple logic: if one ends before the other starts and they're spatially close
            if traj1['end_frame'] < traj2['start_frame']:
                # traj1 ends before traj2 starts
                last_point = traj1['trajectory'][-1]['centroid']
                first_point = traj2['trajectory'][0]['centroid']

                if np.linalg.norm(last_point - first_point) < threshold:
                    # Merge traj2 into the global trajectory
                    for point in traj2['trajectory']:
                        global_trajectories[global_id].append(point)
                    assigned.add((traj2['sensor_id'], traj2['vehicle_id']))

            elif traj2['end_frame'] < traj1['start_frame']:
                # traj2 ends before traj1 starts
                last_point = traj2['trajectory'][-1]['centroid']
                first_point = traj1['trajectory'][0]['centroid']

                if np.linalg.norm(last_point - first_point) < threshold:
                    # Merge traj1 into the global trajectory (prepend traj2)
                    new_points = []
                    for point in traj2['trajectory']:
                        new_points.append(point)
                    new_points.extend(global_trajectories[global_id])
                    global_trajectories[global_id] = new_points
                    assigned.add((traj2['sensor_id'], traj2['vehicle_id']))

        # Sort the global trajectory by scan number
        global_trajectories[global_id].sort(key=lambda x: x['scan'])
        global_id += 1

    return global_trajectories