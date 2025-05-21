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
            C_inv += cov_inv
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


def create_distance_matrix(filters1, filters2):
    """
    Create a distance matrix between two sets of Kalman filters.

    Args:
        filters1: Dictionary of {vehicle_id: kalman_filter} for first set
        filters2: Dictionary of {vehicle_id: kalman_filter} for second set
        max_distance: Maximum allowable distance, beyond which will be set to infinity

    Returns:
        distance_matrix: 2D numpy array of distances
        ids1: List of vehicle IDs from filters1 in order
        ids2: List of vehicle IDs from filters2 in order
    """
    ids1 = list(filters1.keys())
    ids2 = list(filters2.keys())

    n, m = len(ids1), len(ids2)

    # Initialize distance matrix with infinity
    distance_matrix = np.full((n, m), np.inf)

    # Calculate distances between each pair of filters
    for i, id1 in enumerate(ids1):
        kf1 = filters1[id1]
        for j, id2 in enumerate(ids2):
            kf2 = filters2[id2]
            distance = mahalanobis_distance(kf1.X, kf1.P, kf2.X, kf2.P)

            distance_matrix[i, j] = distance

    return distance_matrix, ids1, ids2


def combine_sensor_kalman_filters(sensor_kalman_filters, selected_sensors, max_distance=5.0):
    """
    Combine Kalman filters from multiple sensors using consecutive application of
    the Hungarian algorithm with linear sum assignment.

    Args:
        sensor_kalman_filters: Dictionary mapping sensor_id to a dictionary of
                              {local_vehicle_id: kalman_filter} for each sensor
        selected_sensors: List of sensor IDs to consider in order
        max_distance: Maximum Mahalanobis distance for track association

    Returns:
        combined_filters: Dictionary mapping global_vehicle_id to combined Kalman filter
    """
    if len(selected_sensors) < 2:
        logger.warning("Need at least 2 sensors to perform combination")
        return {}

    # Start with the first sensor's data
    current_filters = sensor_kalman_filters[selected_sensors[0]]

    # Track the source of each combined filter
    source_tracks = {v_id: [(selected_sensors[0], v_id)] for v_id in current_filters}

    # Consecutively combine with each additional sensor
    for i in range(1, len(selected_sensors)):
        sensor_id = selected_sensors[i]

        if sensor_id not in sensor_kalman_filters or not sensor_kalman_filters[sensor_id]:
            logger.warning(f"No data for sensor {sensor_id}, skipping")
            continue

        next_filters = sensor_kalman_filters[sensor_id]

        # Create distance matrix between current combined filters and next sensor's filters
        dist_matrix, current_ids, next_ids = create_distance_matrix(
            current_filters, next_filters)

        # New set of combined filters after this matching
        new_filters = {}
        new_source_tracks = {}

        # Track which vehicles from current and next filters have been matched
        current_matched = set()
        next_matched = set()

        # Apply Hungarian algorithm for optimal assignment
        if dist_matrix.size > 0:  # Only if there are elements to match
            try:
                # The linear_sum_assignment function finds the minimum cost assignment
                row_indices, col_indices = linear_sum_assignment(dist_matrix)

                # Process the matches
                for row_idx, col_idx in zip(row_indices, col_indices):
                    if dist_matrix[row_idx, col_idx] > max_distance:
                        # Skip distant matches
                        continue

                    current_id = current_ids[row_idx]
                    next_id = next_ids[col_idx]

                    # Mark these vehicles as matched
                    current_matched.add(current_id)
                    next_matched.add(next_id)

                    # Combine the two matched filters using covariance intersection
                    kf1 = current_filters[current_id]
                    kf2 = next_filters[next_id]

                    means = [kf1.X, kf2.X]
                    covariances = [kf1.P, kf2.P]

                    combined_mean, combined_cov = covariance_intersection(means, covariances)

                    # Create a new Kalman filter with combined state
                    dt = kf1.dt  # Use dt from first filter
                    global_id = f"combined_{current_id}"

                    combined_kf = KalmanFilter(dt)
                    combined_kf.X = combined_mean
                    combined_kf.P = combined_cov

                    # Update the combined filters dictionary
                    new_filters[global_id] = combined_kf

                    # Track the source of this combined filter
                    new_source_tracks[global_id] = source_tracks[current_id] + [(sensor_id, next_id)]

            except ValueError as e:
                logger.error(f"Error in linear_sum_assignment: {e}")

        # Add unmatched vehicles from current filters
        for current_id in current_ids:
            if current_id not in current_matched:
                global_id = f"unmatched_{current_id}"
                new_filters[global_id] = current_filters[current_id]
                new_source_tracks[global_id] = source_tracks[current_id]

        # Add unmatched vehicles from next sensor
        for next_id in next_ids:
            if next_id not in next_matched:
                global_id = f"unmatched_sensor{i}_{next_id}"
                new_filters[global_id] = next_filters[next_id]
                new_source_tracks[global_id] = [(sensor_id, next_id)]

        # Update current filters for next iteration
        current_filters = new_filters
        source_tracks = new_source_tracks

        # If no filters are left, stop processing
        if not current_filters:
            logger.warning(f"No filters left after combining with sensor {sensor_id}")
            return {}

    # Create the final combined filters with proper global IDs - only keep vehicles seen by multiple sensors
    final_filters = {}
    filter_counter = 0

    for filter_id, kf in current_filters.items():
        # Check if this vehicle was seen by at least two different sensors
        sensors_seen = set(sensor_id for sensor_id, _ in source_tracks[filter_id])

        if len(sensors_seen) >= 2:
            global_vehicle_id = f"global_vehicle_{filter_counter}"
            filter_counter += 1

            final_filters[global_vehicle_id] = kf

            # Store source tracks for reference
            kf.source_tracks = source_tracks[filter_id]

            # Log if vehicle IDs in the group are not all the same
            v_ids = [v_id for _, v_id in source_tracks[filter_id]]
            if len(set(v_ids)) > 1:
                logger.error(f"Track group {global_vehicle_id} combined different vehicle IDs: {v_ids}")
            if len(v_ids) < 4:
                logger.warning(f"Track group {global_vehicle_id} has fewer than 4 vehicle IDs: {v_ids}")

    return final_filters


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

    # Combine the filters using the consecutive Hungarian algorithm approach
    combined_filters = combine_sensor_kalman_filters(sensor_kalman_filters, selected_sensors)

    # Print results
    for vehicle_id, kf in combined_filters.items():
        position = kf.get_state()
        source_tracks = getattr(kf, 'source_tracks', [])
        print(f"{vehicle_id} combined position: {position}")
        print(f"  Created from tracks: {source_tracks}")