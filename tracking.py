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
    plt.close()



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
        plt.close()

    # Calculate overall average MSE
    overall_mse = np.mean(list(mse_values.values())) if mse_values else 0
    print(f"Overall average MSE across all matched trajectories: {overall_mse:.4f}")
    logger.info(f"Overall average MSE across all matched trajectories: {overall_mse:.4f}")

    return mse_values, avg_frame_mse

def compare_mses(real_trajectories, sensor_trajectories, selected_sensors, combined_mses, combined_avg_frame_mse,
                 num_frames=30):
    """
    Compare MSE between real trajectories and individual sensor trajectories.
    Calculate cardinality errors and trajectory matching for each sensor.

    Args:
        real_trajectories: Dictionary of real trajectories {vehicle_id: [position_points]}
        sensor_trajectories: Dictionary of sensor trajectories {sensor_id: {vehicle_id: [position_points]}}
        selected_sensors: List of sensor IDs to analyze
        combined_mses: MSE values from combined trajectories for comparison
        combined_avg_frame_mse: Average frame MSE from combined trajectories
        num_frames: Number of frames to analyze
    """

    print("\n=== SENSOR-WISE MSE COMPARISON ===")
    logger.info("=== SENSOR-WISE MSE COMPARISON ===")

    sensor_results = {}
    all_sensor_avg_mses = []


    for sensor_id in selected_sensors:
        print(f"\n--- Processing Sensor {sensor_id} ---")
        logger.info(f"--- Processing Sensor {sensor_id} ---")

        if sensor_id not in sensor_trajectories or not sensor_trajectories[sensor_id]:
            print(f"No trajectories found for sensor {sensor_id}")
            logger.warning(f"No trajectories found for sensor {sensor_id}")
            continue

        sensor_traj = {vehicle_id: [[point[0], -point[2]] for point in trajectory] for vehicle_id, trajectory in sensor_trajectories[sensor_id].items()}

        # Extract valid trajectory IDs
        real_ids = [id for id, traj in real_trajectories.items() if len(traj) > 0]
        sensor_ids = [id for id, traj in sensor_traj.items() if len(traj) > 0]

        # Calculate cardinality error
        cardinality_error = abs(len(real_ids) - len(sensor_ids))
        print(f"Sensor {sensor_id} - Real trajectories: {len(real_ids)}, Sensor trajectories: {len(sensor_ids)}")
        print(f"Sensor {sensor_id} - Cardinality error: {cardinality_error}")
        logger.info(f"Sensor {sensor_id} - Cardinality error: {cardinality_error}")

        if not real_ids or not sensor_ids:
            print(f"No valid trajectories to compare for sensor {sensor_id}")
            continue

        # Get first points from each trajectory for initial matching
        real_first_points = [real_trajectories[id][20] for id in real_ids]
        sensor_first_points = [sensor_traj[id][0] for id in sensor_ids]

        # Create distance matrix between first points
        distance_matrix = compute_distance_matrix(real_first_points, sensor_first_points)

        # Use linear sum assignment to find optimal matching
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        # Calculate MSE for matched trajectories
        sensor_mse_values = {}
        sensor_frame_mse = {i: [] for i in range(num_frames)}
        sensor_mse_per_frame = {}
        trajectory_lengths = {}
        early_termination_count = 0
        catastrophic_failures_sensor = 0

        for r, c in zip(row_ind, col_ind):
            real_id = real_ids[r]
            sensor_vehicle_id = sensor_ids[c]

            real_traj = real_trajectories[real_id]
            sensor_vehicle_traj = sensor_traj[sensor_vehicle_id]

            sensor_mse_per_frame[real_id] = []

            # Track trajectory lengths
            real_length = len(real_traj)-20
            sensor_length = len(sensor_vehicle_traj)
            trajectory_lengths[real_id] = {
                'real': real_length,
                'sensor': sensor_length,
                'min_length': min(real_length, sensor_length)
            }

            # Check for early termination
            if sensor_length < real_length:
                early_termination_count += 1
                print(
                    f"Sensor {sensor_id} - Vehicle {real_id}: Early termination (sensor: {sensor_length}, real: {real_length})")
                logger.warning(f"Sensor {sensor_id} - Vehicle {real_id}: Early termination")

            # Calculate MSE for overlapping frames
            overlapping_frames = 0
            catastrophic_failures = 0
            for frame in range(num_frames):

                if frame < len(real_traj)-20 and frame < len(sensor_vehicle_traj):
                    real_point = np.array(real_traj[frame+20])  # Use only x, y coordinates
                    sensor_point = np.array(sensor_vehicle_traj[frame])  # Use only x, y coordinates

                    if sensor_point[0] < 999999:
                        squared_error = np.sum((real_point - sensor_point) ** 2)
                        sensor_mse_per_frame[real_id].append(squared_error)
                        sensor_frame_mse[frame].append(squared_error)
                        overlapping_frames += 1
                    else:
                        catastrophic_failures += 1

            if sensor_mse_per_frame[real_id]:
                avg_mse = np.mean(sensor_mse_per_frame[real_id])
                sensor_mse_values[(real_id, sensor_vehicle_id)] = avg_mse
                print(
                    f"Sensor {sensor_id} - Matched Real ID {real_id} with Sensor Vehicle ID {sensor_vehicle_id} - Avg MSE: {avg_mse:.4f} ({overlapping_frames} frames) - Catastrophic failures: {catastrophic_failures}")
                logger.info(
                    f"Sensor {sensor_id} - Matched Real ID {real_id} with Sensor Vehicle ID {sensor_vehicle_id} - Avg MSE: {avg_mse:.4f} ({overlapping_frames} frames) - Catastrophic failures: {catastrophic_failures}")

            catastrophic_failures_sensor += catastrophic_failures

        # Calculate average MSE per frame for this sensor
        sensor_avg_frame_mse = [np.mean(errors) if errors else 0 for frame, errors in
                                sorted(sensor_frame_mse.items())]

        # Calculate overall statistics for this sensor
        if sensor_mse_values:
            sensor_overall_mse = np.mean(list(sensor_mse_values.values()))
            all_sensor_avg_mses.append(sensor_overall_mse)
        else:
            sensor_overall_mse = float('inf')

        # Store results
        sensor_results[sensor_id] = {
            'mse_values': sensor_mse_values,
            'avg_frame_mse': sensor_avg_frame_mse,
            'overall_mse': sensor_overall_mse,
            'cardinality_error': cardinality_error,
            'early_terminations': early_termination_count,
            'trajectory_lengths': trajectory_lengths,
            'num_matched': len(sensor_mse_values),
            'catastrophic_failures': catastrophic_failures_sensor
        }

        print(f"Sensor {sensor_id} - Overall MSE: {sensor_overall_mse:.4f}")
        print(f"Sensor {sensor_id} - Early terminations: {early_termination_count}")
        logger.info(
            f"Sensor {sensor_id} - Overall MSE: {sensor_overall_mse:.4f}, Early terminations: {early_termination_count}, Catastrophic failures: {catastrophic_failures_sensor}")

    # Create comparison plots

    # Plot 1: Overall MSE comparison
    plt.figure(figsize=(10, 6))
    sensor_ids_plot = []
    sensor_mses_plot = []

    for sensor_id in selected_sensors:
        if sensor_id in sensor_results and sensor_results[sensor_id]['overall_mse'] != float('inf'):
            sensor_ids_plot.append(f"Sensor {sensor_id}")
            sensor_mses_plot.append(sensor_results[sensor_id]['overall_mse'])

    # Add combined MSE for comparison
    if combined_mses:
        combined_overall_mse = np.mean(list(combined_mses.values()))
        sensor_ids_plot.append("Combined")
        sensor_mses_plot.append(combined_overall_mse)

    plt.bar(sensor_ids_plot, sensor_mses_plot,
            color=['lightblue' if 'Sensor' in x else 'orange' for x in sensor_ids_plot])
    plt.title('Overall MSE Comparison')
    plt.ylabel('Average MSE')
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('output/overall_mse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Frame-by-frame MSE comparison
    plt.figure(figsize=(12, 6))
    for sensor_id in selected_sensors:
        if sensor_id in sensor_results and sensor_results[sensor_id]['avg_frame_mse']:
            plt.plot(range(len(sensor_results[sensor_id]['avg_frame_mse'])),
                     sensor_results[sensor_id]['avg_frame_mse'],
                     marker='o', label=f'Sensor {sensor_id}', alpha=0.7)

    if combined_avg_frame_mse:
        plt.plot(range(len(combined_avg_frame_mse)), combined_avg_frame_mse,
                 marker='s', linewidth=3, label='Combined', color='black')

    plt.title('MSE per Frame Comparison')
    plt.xlabel('Frame Number')
    plt.ylabel('Average MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/frame_mse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    logger.info("=== SUMMARY STATISTICS ===")

    if all_sensor_avg_mses:
        best_sensor_mse = min(all_sensor_avg_mses)
        worst_sensor_mse = max(all_sensor_avg_mses)
        avg_sensor_mse = np.mean(all_sensor_avg_mses)

        print(f"Best individual sensor MSE: {best_sensor_mse:.4f}")
        print(f"Worst individual sensor MSE: {worst_sensor_mse:.4f}")
        print(f"Average individual sensor MSE: {avg_sensor_mse:.4f}")

        if combined_mses:
            combined_overall_mse = np.mean(list(combined_mses.values()))
            print(f"Combined approach MSE: {combined_overall_mse:.4f}")

            improvement_over_avg = ((avg_sensor_mse - combined_overall_mse) / avg_sensor_mse) * 100
            improvement_over_best = ((best_sensor_mse - combined_overall_mse) / best_sensor_mse) * 100

            print(f"Combined approach improvement over average sensor: {improvement_over_avg:.2f}%")
            print(f"Combined approach improvement over best sensor: {improvement_over_best:.2f}%")

            logger.info(
                f"Combined approach improvement: {improvement_over_avg:.2f}% over average, {improvement_over_best:.2f}% over best")

    # Print detailed results for each sensor
    for sensor_id, results in sensor_results.items():
        print(f"\nSensor {sensor_id} detailed results:")
        print(f"  - Matched trajectories: {results['num_matched']}")
        print(f"  - Overall MSE: {results['overall_mse']:.4f}")
        print(f"  - Cardinality error: {results['cardinality_error']}")
        print(f"  - Early terminations: {results['early_terminations']}")

        # Print trajectory length statistics
        if results['trajectory_lengths']:
            avg_real_length = np.mean([info['real'] for info in results['trajectory_lengths'].values()])
            avg_sensor_length = np.mean([info['sensor'] for info in results['trajectory_lengths'].values()])
            avg_overlap = np.mean([info['min_length'] for info in results['trajectory_lengths'].values()])

            print(f"  - Average real trajectory length: {avg_real_length:.1f}")
            print(f"  - Average sensor trajectory length: {avg_sensor_length:.1f}")
            print(f"  - Average overlap length: {avg_overlap:.1f}")

    # Log detailed results for each sensor
    for sensor_id, results in sensor_results.items():
        logger.info(f"Sensor {sensor_id} detailed results:")
        logger.info(f"  - Matched trajectories: {results['num_matched']}")
        logger.info(f"  - Overall MSE: {results['overall_mse']:.4f}")
        logger.info(f"  - Cardinality error: {results['cardinality_error']}")
        logger.info(f"  - Early terminations: {results['early_terminations']}")
        logger.info(f"  - Catastrophic failures: {results['catastrophic_failures']}")

        # Log trajectory length statistics
        if results['trajectory_lengths']:
            avg_real_length = np.mean([info['real'] for info in results['trajectory_lengths'].values()])
            avg_sensor_length = np.mean([info['sensor'] for info in results['trajectory_lengths'].values()])
            avg_overlap = np.mean([info['min_length'] for info in results['trajectory_lengths'].values()])

            logger.info(f"  - Average real trajectory length: {avg_real_length:.1f}")
            logger.info(f"  - Average sensor trajectory length: {avg_sensor_length:.1f}")
            logger.info(f"  - Average overlap length: {avg_overlap:.1f}")

    return sensor_results