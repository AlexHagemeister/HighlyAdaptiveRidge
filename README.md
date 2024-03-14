# Highly Adaptive Models Comparison Project

## Overview

This project aims to empirically compare the performance of two statistical models: the Highly Adaptive Lasso (HAL) and Highly Adaptive Ridge (HAR) in high-dimensional data analysis. Both models are variants of regularization techniques used in regression and machine learning to prevent overfitting by adding a penalty to the loss function. The focus is on evaluating these models in terms of computational efficiency, prediction accuracy, and the effectiveness of regularization under various conditions.

## Current Status

Preliminary results showing dramatic decrease in compute to train HAR vs HAL while seeming to maintain similar prediction accuracy. 
Next: building upon the simulation environment for more exhaustive testing and comparison, along with more detailed (and disaggregated) visualizations of key eval metrics.

## Goals

1. **Empirical Comparison**: Conduct a thorough empirical comparison between HAL and HAR models to assess their performance on high-dimensional datasets.
2. **Computation Time**: Evaluate and compare the computation times required by each model to fit the data, providing insights into their efficiency.
3. **Prediction Accuracy**: Use metrics such as Mean Squared Error (MSE) to compare the prediction accuracy of the models across different dataset sizes and conditions.
4. **Regularization Effectiveness**: Examine how cross-validation techniques control the regularization parameters in both models, focusing on the L1 norm control in HAR despite its explicit use of L2 regularization.
5. **Scalability**: Assess how each model scales with increasing data dimensions, offering insights into their applicability to real-world, high-dimensional datasets.

## Methodology

The project utilizes a simulation-based approach to generate synthetic datasets with controllable features such as the number of samples, number of features, and the level of noise. The simulation involves:

- Generating datasets using the `make_regression` function from scikit-learn.
- Fitting both HAL and HAR models to these datasets.
- Evaluating model performance using cross-validation and calculating MSE on a test set.
- Repeating the process for various dataset sizes to gather comprehensive performance data.

## Results Analysis

The simulation results will be analyzed and visualized to compare the computation time and MSE of HAL and HAR models. This analysis aims to provide a clear understanding of each model's strengths and limitations, particularly regarding their efficiency and accuracy in handling high-dimensional data.

## Project Structure

- `highly_adaptive_lasso.py`: Implementation of the Highly Adaptive Lasso model.
- `highly_adaptive_ridge.py`: Implementation of the Highly Adaptive Ridge model.
- `data_generators.py`: Functions to generate synthetic datasets for the simulation.

## Running the Simulation

(Building in progress, disregard) To run the simulation, ensure you have Python installed along with necessary libraries such as `numpy`, `matplotlib`, and `scikit-learn`. Execute the `simulation_1.0.py` script in your environment to begin the comparison.

## Future Work

As the project progresses, additional models and comparison metrics may be introduced to expand the scope of the analysis. Further optimizations and enhancements to the simulation process will also be considered to improve the accuracy and relevance of the findings.
