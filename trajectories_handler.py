from numpy import inf
from scipy.optimize import linear_sum_assignment

from tracking import compute_distance_matrix

def add_new_point_to_trajectories(combined_kalman_filters, combined_trajectories, threshold):

    if not combined_trajectories:
        # if first iteration and trajectories empty, initialize them with first state
        for filter_id, kf in combined_kalman_filters.items():
            # Initialize the trajectory for this sensor
            combined_trajectories[filter_id] = []
            # Append the initial state to the trajectory
            combined_trajectories[filter_id].append([kf.get_state()[0], -kf.get_state()[2]])
    else:
        # get all the states from the kalman filters
        states = []
        for _, kf in combined_kalman_filters.items():
            states.append([kf.get_state()[0], -kf.get_state()[2]])

        #get last points of the trajectories
        last_points = []
        for _, traj in combined_trajectories.items():
            last_points.append(traj[-1])

        # calculate the distance between the last points and the current states
        distance_matrix =  compute_distance_matrix(last_points, states)

        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        unmatched_prev = set(range(len(last_points)))
        unmatched_curr = set(range(len(states)))

        for r, c in zip(row_ind, col_ind):
            if distance_matrix[r, c] < threshold:
                unmatched_prev.discard(r)
                unmatched_curr.discard(c)

                combined_trajectories[list(combined_trajectories.keys())[r]].append(states[c])

        # Handle unmatched vehicles
        for i in unmatched_curr:
            name = list(combined_kalman_filters.keys())[i]
            # Try to find a unique name for the new trajectory
            original_name = name
            counter = 0
            while name in combined_trajectories:
                counter += 1
                name = f"{original_name}{counter}"

            # Create new trajectory with unique name
            combined_trajectories[name] = []
            combined_trajectories[name].append(states[i])

        for i in unmatched_prev:
            # If the vehicle is not in the current states, makes it infinity so it doesn't match in the future
            if i in combined_trajectories:
                combined_trajectories[list(combined_trajectories.keys())[i]].append([inf, inf])

        print("entered vehicles:", len(unmatched_curr))
        print("exited vehicles:", len(unmatched_prev))