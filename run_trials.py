import numpy as np
import time
from sklearn.model_selection import train_test_split
from data_generators import DataGenerator, SmoothDataGenerator, JumpDataGenerator, SinusoidalDataGenerator
from highly_adaptive_lasso import HAL
from highly_adaptive_ridge import HAR
from sklearn.metrics import mean_squared_error

import numpy as np

class RunTrials:

    @classmethod
    def run_trial(cls, method, X_train, Y_train, X_test, Y_test):
        """
        Run a single trial for a given method.
        """
        start_time = time.time()
        method.fit(X_train, Y_train)
        training_time = time.time() - start_time
        
        predictions = method.predict(X_test)
        mse = mean_squared_error(Y_test, predictions)
        
        return {"training_time": training_time, "MSE": mse}

    @classmethod
    def run_trials(cls, d, sample_sizes, num_trials, dgp):
        """
        Run trials for all sample sizes, for both HAL and HAR methods, and return
        aggregated results including mean training time, training time variance,
        mean MSE, and MSE variance for easy DataFrame conversion.
        PARAMS:
        d: int, dimension of the covariates
        sample_sizes: list of ints, sample sizes to test
        num_trials: int, number of trials for each sample size
        dgp: dgp class to use for generating data
        """

        methods = {'HAL': HAL(), 'HAR': HAR()}
        results_list = []

        # run num_trials trials for each sample size for both HAL and HAR
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

