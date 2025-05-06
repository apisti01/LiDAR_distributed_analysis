def get_user_input():
    """
    Get the number of sensors and frames from user input.

    Returns:
        tuple: (num_sensors, num_frames)
    """
    # Ask for number of sensors
    while True:
        try:
            num_sensors = int(input("Enter the number of sensors: "))
            if num_sensors <= 0:
                print("Number of sensors must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")

    # Ask for number of frames
    while True:
        try:
            num_frames = int(input("Enter the number of frames: "))
            if num_frames <= 0:
                print("Number of frames must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")

    return num_sensors, num_frames


def load_data(num_sensors, num_frames):
    """
    Load data based on the number of sensors and frames.

    Args:
        num_sensors (int): Number of sensors
        num_frames (int): Number of frames

    Returns:
        data: The loaded data (format depends on implementation)
    """
    # Placeholder for data loading logic
    print(f"Loading data for {num_sensors} sensors and {num_frames} frames...")

    # Implement your data loading logic here

    return None


def main():
    """Main function to execute the data loading process."""
    num_sensors, num_frames = get_user_input()
    data = load_data(num_sensors, num_frames)

    # Process or display the data as needed
    print("Data loading complete.")


if __name__ == "__main__":
    main()