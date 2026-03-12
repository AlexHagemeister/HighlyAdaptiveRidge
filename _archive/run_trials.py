import numpy as np
import time
from sklearn.model_selection import train_test_split
from data_generators import DataGenerator, SmoothDataGenerator, JumpDataGenerator, SinusoidalDataGenerator
from highly_adaptive_lasso import HAL
from highly_adaptive_ridge import HAR
from kernel_har import KernelHAR
from typing import*

from sklearn.metrics import mean_squared_error

import numpy as np

class RunTrials:

    @classmethod
    def run_trials(cls, d, sample_sizes, num_trials, dgp):
        """
        Run trials on synthetic data for all sample sizes, methods, and DGPs, and return
        aggregated results
        PARAMS:
            num_features (p): int, dimension of the covariates 
            sample_sizes: list of ints, sample sizes to test
            num_trials: int, number of trials for each sample size
            dgp: dgp class to use for generating data
            methods: dict of methods to test

        RETURNS:
            results_list: list of dicts, each containing the results of a single trial
            structured as follows:
            {
                'Sample Size': int,
                'Method': str,
                'training_time': float,
                'MSE': float
            }
        """
        # # Unpack keyword arguments with default values
        # dgp: Callable = kwargs.get('dgp', SmoothDataGenerator())
        # num_trials: int = kwargs.get('num_trials', 5) 
        # p: int = kwargs.get('num_features', 1) 
        # sample_sizes: np.ndarray = kwargs.get('sample_sizes', np.arange(start=100, stop=1000, step=100))
        # methods: Dict[str, callable] = kwargs.get('methods', {'HAR': HAR(), 'KernelHAR': KernelHAR(), 'HAL': HAL()})

        methods = {'HAR': HAR()}
        results_list = []

        # run 'num_trials' trials for each sample size for all methods
        for n in sample_sizes:
            metrics = {m: {'training_times': [], 'mses': []} for m in methods}

            for _ in range(num_trials):
                X, Y = dgp.generate_data(n, d)
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.9, random_state=42)

                for method_name, method in methods.items():
                    results_list.append({
                        'Sample Size': n,
                        'Method': method_name,
                        **cls.run_trial(method, X_train, Y_train, X_test, Y_test)
                    })

        return results_list

    @classmethod
    def run_trial(cls, method, X_train, Y_train, X_test, Y_test):
        """
        Helper method. Run a single trial for a given method.
        """
        # fit the model and record training time
        start_time = time.time()
        method.fit(X_train, Y_train)
        training_time = time.time() - start_time
        
        # predict and calculate MSE
        predictions = method.predict(X_test)
        mse = mean_squared_error(Y_test, predictions)
        
        return {"training_time": training_time, "MSE": mse}
