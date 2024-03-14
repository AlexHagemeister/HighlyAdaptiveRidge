# ALL THE IMPORTS!!
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from highly_adaptive_lasso import HAL # Placeholder for actual import
from highly_adaptive_ridge import HAR # Placeholder for actual import
from data_generators import DataGenerator # Placeholder for actual import
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Define the simulation parameters
n_simulations = 100
data_sizes = np.arange(100, 1100, 100)  # From 100 to 1000 in steps of 100
results = {
    "HAL": {"time": [], "mse": []},
    "HAR": {"time": [], "mse": []}
}

# Placeholder for data generator initialization (to be replaced with actual initialization)
data_generator = DataGenerator()

# Simulation function (to be expanded with actual model fitting and evaluation)
def run_simulation(data_size, n_features, noise=0.1, random_state=42):
    """
    Runs a single simulation for given data size and number of features.

    Parameters:
    - data_size: int, the number of samples to generate.
    - n_features: int, the number of features to generate.
    - noise: float, the standard deviation of the gaussian noise applied to the output.
    - random_state: int, random state for reproducibility.

    Returns:
    - hal_time: float, computation time for HAL model.
    - har_time: float, computation time for HAR model.
    - hal_mse: float, mean squared error for HAL model.
    - har_mse: float, mean squared error for HAR model.
    """
    # Generate regression data
    X, y = make_regression(n_samples=data_size, n_features=n_features, noise=noise, random_state=random_state)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Initialize models
    hal = HAL() 
    har = HAR()  
    
    # Fit models and record computation time
    start_time = time.time()
    hal.fit(X_train, y_train)
    hal_time = time.time() - start_time
    
    start_time = time.time()
    har.fit(X_train, y_train)
    har_time = time.time() - start_time
    
    # Evaluate models
    hal_mse = mean_squared_error(y_test, hal.predict(X_test))
    har_mse = mean_squared_error(y_test, har.predict(X_test))
    
    return hal_time, har_time, hal_mse, har_mse



def plot_simulation_results(data_sizes, results):
    """
    Plots the simulation results comparing HAL and HAR models.

    Parameters:
    - data_sizes: Array of data sizes used in the simulations.
    - results: Dictionary containing the simulation results for HAL and HAR models.
    """
    plt.figure(figsize=(12, 6))

    # Plot for Computation Time Comparison
    plt.subplot(1, 2, 1)
    plt.plot(data_sizes, results["HAL"]["time"], label="HAL Time", marker='o')
    plt.plot(data_sizes, results["HAR"]["time"], label="HAR Time", marker='x')
    plt.title("Computation Time Comparison")
    plt.xlabel("Data Size")
    plt.ylabel("Time (seconds)")
    plt.legend()

    # Plot for Model Accuracy (MSE) Comparison
    plt.subplot(1, 2, 2)
    plt.plot(data_sizes, results["HAL"]["mse"], label="HAL MSE", marker='o')
    plt.plot(data_sizes, results["HAR"]["mse"], label="HAR MSE", marker='x')
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Data Size")
    plt.ylabel("Mean Squared Error")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Main simulation loop
n_features_multiplier = 2  # This will set the number of features to be twice the number of samples
for data_size in data_sizes:
    n_features = data_size * n_features_multiplier  # Calculate number of features based on data_size
    hal_time_total, har_time_total, hal_mse_total, har_mse_total = 0, 0, 0, 0
    for _ in range(n_simulations):
        hal_time, har_time, hal_mse, har_mse = run_simulation(data_size, n_features)
        hal_time_total += hal_time
        har_time_total += har_time
        hal_mse_total += hal_mse
        har_mse_total += har_mse
    
    # Record average metrics
    results["HAL"]["time"].append(hal_time_total / n_simulations)
    results["HAL"]["mse"].append(hal_mse_total / n_simulations)
    results["HAR"]["time"].append(har_time_total / n_simulations)
    results["HAR"]["mse"].append(har_mse_total / n_simulations)


# Visualize the simulation results
plot_simulation_results(data_sizes, results)

