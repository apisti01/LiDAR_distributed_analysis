import numpy as np
import logging
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from kalman_filter import KalmanFilter

logger = logging.getLogger()

def covariance_intersection(means, covariances):
    """
    Perform covariance intersection for multiple Kalman filters.

    Args:
        means: List of state means from different filters
        covariances: List of covariance matrices from different filters

    Returns:
        combined_mean: Combined state estimate
        combined_cov: Combined covariance matrix
    """
    if len(means) == 0:
        return None, None

    if len(means) == 1:
        return means[0], covariances[0]

    # Get dimensions from the first state
    n = means[0].shape[0]

    # Initialize combined inverse covariance and weighted mean
    C_inv = np.zeros((n, n))
    weighted_mean = np.zeros(n)

    # Find optimal weights using convex optimization
    # For simplicity, we'll use equal weights, but more sophisticated
    # methods could be used to minimize trace or determinant of result

    # Compute weighted combination
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        try:
            cov_inv = np.linalg.inv(cov)
            C_inv +=  cov_inv
            weighted_mean += np.dot(cov_inv, mean)
        except np.linalg.LinAlgError:
            # Handle singular covariance matrices by slightly increasing diagonal
            adjusted_cov = cov + np.eye(n) * 1e-5
            cov_inv = np.linalg.inv(adjusted_cov)
            C_inv += cov_inv
            weighted_mean += np.dot(cov_inv, mean)

    # Compute final combined covariance and mean
    combined_cov = np.linalg.inv(C_inv)
    combined_mean = np.dot(combined_cov, weighted_mean)

    return combined_mean, combined_cov


def mahalanobis_distance(x1, P1, x2, P2):
    """
    Calculate Mahalanobis distance between two Gaussian distributions.

    Args:
        x1, x2: State vectors (means)
        P1, P2: Covariance matrices

    Returns:
        distance: Mahalanobis distance
    """
    # Get position components (x, y, z) for distance calculation
    pos1 = x1[:3]
    pos2 = x2[:3]

    # Combined covariance for position components
    S = P1[:3, :3] + P2[:3, :3]

    # Ensure S is not singular
    S_adjusted = S + np.eye(3) * 1e-5

    # Difference vector
    diff = pos1 - pos2

    # Mahalanobis distance
    try:
        S_inv = np.linalg.inv(S_adjusted)
        distance = np.sqrt(np.dot(diff.T, np.dot(S_inv, diff)))
    except np.linalg.LinAlgError:
        # Fallback to Euclidean distance if inversion fails
        distance = np.linalg.norm(diff)

    return distance


def associate_tracks(sensor_kalman_filters, selected_sensors, max_distance=5.0):
    """
    Associate tracks across multiple sensors based on position.

    Args:
        sensor_kalman_filters: Dictionary mapping sensor_id to a dictionary of
                              {local_vehicle_id: kalman_filter} for each sensor
        selected_sensors: List of sensor IDs to consider
        max_distance: Maximum Mahalanobis distance for track association

    Returns:
        grouped_tracks: List of lists, where each inner list contains tuples of
                      (sensor_id, local_vehicle_id, kalman_filter) representing
                      the same physical vehicle across different sensors
    """
    # Create a flat list of all tracks with their source info
    all_tracks = []
    for sensor_id in selected_sensors:
        if sensor_id in sensor_kalman_filters:
            for vehicle_id, kf in sensor_kalman_filters[sensor_id].items():
                all_tracks.append((sensor_id, vehicle_id, kf))

    # If no tracks, return empty result
    if not all_tracks:
        return []

    # Start with each track in its own group
    grouped_tracks = [[track] for track in all_tracks]

    # Iteratively merge groups
    new_groups = []

    while grouped_tracks:
        current_group = grouped_tracks.pop(0)

        i = 0
        while i < len(grouped_tracks):
            candidate_group = grouped_tracks[i]

            # Check if these groups can be merged (no sensor overlap and close position)
            can_merge = True
            sensor_ids_current = {track[0] for track in current_group}
            sensor_ids_candidate = {track[0] for track in candidate_group}

            # Cannot merge if same sensor appears in both groups
            if sensor_ids_current.intersection(sensor_ids_candidate):
                i += 1
                continue

            # Check distances between all pairs across groups
            distance_check_passed = False
            for track1 in current_group:
                for track2 in candidate_group:
                    _, _, kf1 = track1
                    _, _, kf2 = track2

                    # Calculate Mahalanobis distance between state estimates
                    distance = mahalanobis_distance(kf1.X, kf1.P, kf2.X, kf2.P)

                    if distance <= max_distance:
                        distance_check_passed = True
                        break

                if distance_check_passed:
                    break

            if distance_check_passed:
                # Merge groups
                current_group.extend(candidate_group)
                grouped_tracks.pop(i)
            else:
                i += 1

        new_groups.append(current_group)

    return new_groups


def combine_sensor_kalman_filters(sensor_kalman_filters, selected_sensors, max_distance=5.0):
    """
    Combine Kalman filters from multiple sensors using covariance intersection,
    based on position-based track association.

    Args:
        sensor_kalman_filters: Dictionary mapping sensor_id to a dictionary of
                              {local_vehicle_id: kalman_filter} for each sensor
        selected_sensors: List of sensor IDs to consider
        max_distance: Maximum Mahalanobis distance for track association

    Returns:
        combined_filters: Dictionary mapping global_vehicle_id to combined Kalman filter
    """
    # Associate tracks based on position
    grouped_tracks = associate_tracks(sensor_kalman_filters, selected_sensors, max_distance)

    # Combine filters for each group
    combined_filters = {}

    for i, track_group in enumerate(grouped_tracks):
        global_vehicle_id = f"global_vehicle_{i}"

        # Extract means and covariances from each filter in this group
        means = []
        covariances = []

        for sensor_id, local_vehicle_id, kf in track_group:
            means.append(kf.X)
            covariances.append(kf.P)

        # Apply covariance intersection
        combined_mean, combined_cov = covariance_intersection(means, covariances)

        # Create a new Kalman filter with the combined state and covariance
        dt = track_group[0][2].dt  # Use dt from first filter in group

        combined_kf = KalmanFilter(dt)
        combined_kf.X = combined_mean
        combined_kf.P = combined_cov

        # Store the combined filter
        combined_filters[global_vehicle_id] = combined_kf

        # For debugging/reference, store which local tracks were combined
        combined_kf.source_tracks = [(s_id, v_id) for s_id, v_id, _ in track_group]

        # Log if vehicle IDs in the group are not all the same
        v_ids = [v_id for _, v_id, _ in track_group]
        if len(set(v_ids)) > 1:
            logger.info(f"Track group {global_vehicle_id} combined different vehicle IDs: {v_ids}")
    return combined_filters


# Example usage
if __name__ == "__main__":
    from kalman_filter import KalmanFilter

    # Example initialization with different local vehicle IDs for each sensor
    selected_sensors = ["sensor1", "sensor2", "sensor3"]
    sensor_kalman_filters = {sensor_id: {} for sensor_id in selected_sensors}

    # Create some test Kalman filters
    # Sensor 1 sees two vehicles with local IDs "A" and "B"
    sensor_kalman_filters["sensor1"]["A"] = KalmanFilter(0.1)
    sensor_kalman_filters["sensor1"]["B"] = KalmanFilter(0.1)

    # Sensor 2 sees two vehicles with local IDs "X" and "Y"
    sensor_kalman_filters["sensor2"]["X"] = KalmanFilter(0.1)
    sensor_kalman_filters["sensor2"]["Y"] = KalmanFilter(0.1)

    # Sensor 3 sees one vehicle with local ID "1"
    sensor_kalman_filters["sensor3"]["1"] = KalmanFilter(0.1)

    # Simulate some measurements
    # Vehicle at position (10, 20, 0) - seen by sensor1 as "A" and sensor2 as "X"
    sensor_kalman_filters["sensor1"]["A"].update(np.array([10, 20, 0]))
    sensor_kalman_filters["sensor2"]["X"].update(np.array([10.5, 19.8, 0]))

    # Vehicle at position (30, 15, 0) - seen by sensor1 as "B" and sensor3 as "1"
    sensor_kalman_filters["sensor1"]["B"].update(np.array([30, 15, 0]))
    sensor_kalman_filters["sensor3"]["1"].update(np.array([29.7, 15.2, 0]))

    # Vehicle at position (5, 40, 0) - seen only by sensor2 as "Y"
    sensor_kalman_filters["sensor2"]["Y"].update(np.array([5, 40, 0]))

    # Combine the filters
    combined_filters = combine_sensor_kalman_filters(sensor_kalman_filters, selected_sensors)

    # Print results
    for vehicle_id, kf in combined_filters.items():
        position = kf.get_state()
        source_tracks = getattr(kf, 'source_tracks', [])
        print(f"{vehicle_id} combined position: {position}")
        print(f"  Created from tracks: {source_tracks}")