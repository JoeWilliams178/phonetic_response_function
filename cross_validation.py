import numpy as np
from numpy.linalg import inv
from scipy.stats import pearsonr
from scipy.stats import mode
import scipy.io.wavfile as wav
from response_data import Response
from regularisation_matrices import Matrices
from interpret_results import Interpretation

import matplotlib.pyplot as plt


class CrossVal:
    def __init__(self):
        """data: eeg data for all trials and participants of a certain condition
        lambda_errors: r and mse for all trials in the cross-validation for a single ridge parameter
        averaged_lambda_errors: averaged r and mse for each ridge parameter after cross-validation
        """
        self.data = None
        self.lambda_errors = np.zeros(shape=29, dtype=np.ndarray)
        self.averaged_lambda_errors = np.zeros(shape=9, dtype=np.ndarray)
        self.electrodes = None

    def final_model(self, trfs, desired_freq, lag_min, lag_max, ppt, trial):
        sum_trfs = 0
        for trf in trfs:
            sum_trfs += trf
        average_trf = sum_trfs / len(trfs)
        # plt.plot(average_trf)
        # plt.show()

        sampling_rate, signal_data, time_array = cross.read_stimulus(condition, trial)
        sm = Matrices()
        # env_matrix = sm.construct_phoneme_features_matrix('sense', 29, time_array, sampling_rate, desired_freq,
        #                                              lag_min, lag_max)

        env_matrix = sm.construct_envelope_matrix(signal_data, time_array, sampling_rate,
                                                        desired_freq,
                                                        lag_min, lag_max)

        prediction = np.matmul(env_matrix.transpose(), average_trf)
        actual = self.data[ppt][trial]
        error = self.calculate_rank_error(prediction, actual)
        return error

    def optimal_lambda(self):
        """from averaged errors for each ridge parameter. find the most suitable ridge_paremeter value by maximising r
        and minimising mse

        variables:
        lambda_value: index of the current ridge_parameter derive trfs
        average_error: stores the error for every electrode
        """

        best_lambda = np.zeros(self.electrodes)
        for electrode in range(0, self.electrodes):
            r, mse = 0, 10000
            lambda_index = -1
            for index, error in enumerate(self.averaged_lambda_errors):
                # err = np.load(f'average_error{error}.npy', allow_pickle=True)
                new_r = error[electrode][0]
                new_mse = error[electrode][1]
                if new_r > r and new_mse < mse:
                    lambda_index = index
                    r = new_r
                    mse = new_mse
            best_lambda[electrode] = lambda_index

        return mode(best_lambda)[0]
        # return int(np.mean(best_lambda))

    def calculate_average_error(self, lambda_value):
        """averages the ridge parameter r values and mse for all test sets after cross-validation. to produce two values
        for the specified ridge parameter These values are created for each electrode

        arguments and variables:
        lambda_value: index of the current ridge_parameter derive trfs
        average_error: stores the error for every electrode
        """

        number_of_trials = 29
        average_error = np.ndarray(shape=(self.electrodes), dtype=tuple)
        for electrode in range(0, self.electrodes):
            r, mse = 0, 0
            for error in self.lambda_errors:
                # err = np.load(f'error{error}.npy', allow_pickle=True)
                electrode_error = error[electrode]
                r += electrode_error[0]
                mse += electrode_error[1]

            # print(r / len(self.lambda_errors))
            average_error[electrode] = (r / 29, mse / 29)

        self.averaged_lambda_errors[lambda_value] = average_error
        # np.save(f'average_error{lambda_value}', average_error)

    def calculate_rank_error(self, prediction, actual_response):
        """for a given predicted and actual response find the pearsons r and the mse for each of the 32
        electrodes

        arguments and variables:
        prediction: predicted response data by electrode
        actual_response: eeg recorded response electrode by data
        errors: r and mse for each electrode stored as a tuple in an np array

        returns:
        r and mse for each electrode
        """
        errors = np.zeros(shape=(self.electrodes), dtype=tuple)
        for electrode, data in enumerate(actual_response.transpose()):
            r = pearsonr(data, prediction[:, electrode])
            mse = np.mean((data - prediction[:, electrode]) ** 2)
            errors[electrode] = (r[0], mse)
        return errors

    def build_prediction(self, ppt, trial_trf, test_set, prediction, lambda_value, condition):
        """cycles through all of the trials of a certain condition for a specific participant. using each trial as a
        test trial and the others to train the data by averaging the trf value for each lambda. result is 29 predictions
        with pearsons r values and MSE for each electrode for each lambda value. This is M-fold cross validation.
        Subsequently averages the errors for each lambda value.

        arguments and variables:
        ppt: participant being assessed
        trial_trf: array of trf for each lambda value
        test_set: set being used to test rather than train
        prediction:average trf  - initial 0 then trials cycled through
        lambda_value: index for lambda in trial_trf
        condition: condition being assessed
        trfs: all temporal response functions produced for each lambda value
        error: tuple containing r and mse for the current training sets

        returns:
        the trf array
        """
        if test_set == -1:
            self.calculate_average_error(lambda_value)
            if lambda_value != 8:
                lambda_value += 1
                test_set = trial_trf.shape[1] - 1
                self.build_prediction(ppt, trial_trf, test_set, 0, lambda_value, condition)
            else:
                return
        else:
            for trial in range(0, trial_trf.shape[1]):
                if trial != test_set:
                    w = trial_trf[lambda_value][trial]
                    prediction = prediction + w
                    del w
            prediction = prediction / (trial_trf.shape[1] - 1)
            sm = Matrices()
            sampling_rate, signal_data, time_array = self.read_stimulus(condition, trial)
            # env_matrix = sm.construct_phoneme_features_matrix('sense', trial, time_array, sampling_rate, desired_freq,
            #                                              lag_min, lag_max)
            env_matrix = sm.construct_envelope_matrix(signal_data, time_array, sampling_rate,
                                                                    desired_freq,
                                                                    lag_min, lag_max)

            predicted_response = np.matmul(env_matrix.transpose(), prediction)
            del env_matrix
            del sampling_rate
            del signal_data
            del time_array
            del sm

            actual_response = self.data[ppt][test_set]
            error = self.calculate_rank_error(predicted_response, actual_response)
            del predicted_response
            del actual_response
            self.lambda_errors[test_set] = error
            # np.save(f'error{test_set}', error)

            test_set = test_set - 1
            self.build_prediction(ppt, trial_trf, test_set, prediction, lambda_value, condition)

    def smoothing_matrix(self, dimension):
        """matrix created from the stimulus identity matrix. Meant to add bias to surrounding data points

        arguments and variables:
        dimension: the size of the matrix
        identity: the modified identity matrix
        indexes: tuples holding coordinates of the elements in the matrix with a value of 2
        arr_up: column locations one to the right of elements with a value of 2
        arr_down: column locations one to the left of elements with a value of 2

        returns:
        the modified smoothing matrix
        """

        identity = np.identity(dimension)
        identity[identity != 0] = 2
        identity[0][0], identity[-1][-1] = 1, 1
        identity[0][1], identity[-1][-2] = -1, -1
        indexes = np.where(identity == 2)
        arr_up = indexes[1] + 1
        arr_down = indexes[1] - 1
        identity[indexes[0], arr_up] = -1
        identity[indexes[0], arr_down] = -1

        return identity

    def trial_parameter(self, autocovariance, stimulus_response, ridge_parameters):
        """for a single trial and participant. Calculates trf values for all of the different ridge values being
         assessed. This is calculated for each electrode.

        arguments and variables:
        autocovariance: the stimulus autocovariance envelope. e.g. envelope or spectrogram
        stimulus_response: stimulus matrix transpose multiplied by the corresponding response data
        smoothing: list of ridge lambda values
        identity: bias matrix derived from the autocovariance identity matrix
        ridge: final ridge is inv(autocovariance + lambda * bias matrix)
        w: trf produced
        trfs: all temporal response functions produced for each lambda value.

        returns:
        the trf array
        """
        identity = self.smoothing_matrix(autocovariance.shape[0])
        trfs = np.zeros(shape=len(ridge_parameters), dtype=np.ndarray)
        for index, rp in enumerate(ridge_parameters):
            ridge = rp * identity

            ridge = autocovariance + ridge
            ridge = inv(ridge)
            trf = np.matmul(ridge, stimulus_response)

            trfs[index] = trf
        return trfs

    def read_stimulus(self, condition, trial):
        """matrix created from the stimulus identity matrix. Meant to add bias to surrounding data points

         arguments and variables:
         dimension: the size of the matrix
         identity: the modified identity matrix
         indexes: tuples holding coordinates of the elements in the matrix with a value of 2
         arr_up: column locations one to the right of elements with a value of 2
         arr_down: column locations one to the left of elements with a value of 2

         returns:
         the modified smoothing matrix
         """
        trial_name = f'trials/{condition}_sentence_block_{trial}'
        recording = f'{trial_name}.wav'
        sampling_rate, signal_data = wav.read(recording)
        signal_data = signal_data[int(low_trim * sampling_rate): int(high_trim * sampling_rate)]
        time_array = np.arange(0, len(signal_data)) / sampling_rate
        return sampling_rate, signal_data, time_array


if __name__ == '__main__':
    """select the appropriate values for the variables listed below

    cycles through each condition, the reads in all trial data for that condition trimmed to the appropriate length.
    cycles through each participant and trial for the current condition. Construct matrices for each trial, run
    cross-validation (build_prediction) and identify the optimal ridge parameter. build final trf using all trials and 
    the optimal ridge parameter and compare response against the untested trials actual response"""

    conditions = ('sense', 'nonsense', 'sense_control', 'nonsense_control')
    ridge_parameters = [10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]
    desired_freq = 1000
    lag_min, lag_max = 0, 200
    low_trim = 1.28
    high_trim = 15.36
    participants = 18
    trials = 30
    low_trim_points = int(low_trim * desired_freq)
    response_data = Response()
    cross = CrossVal()
    cross.electrodes = 32
    check = []
    interpretation = Interpretation()

    for cnd_index, condition in enumerate(conditions):
        """experimented with reading trials in individually but was very slow"""
        cross.data = None
        cross.data, skipped = response_data.load_specific_condition(cnd_index, lag_max, low_trim_points)
        for ppt in range(0, participants):
            ppt_trial_data = None
            if ppt in skipped:
                continue
            else:
                # cross.lambda_errors = np.zeros(shape=29, dtype=np.ndarray)
                # cross.averaged_lambda_errors = np.zeros(shape=9, dtype=np.ndarray)
                print(ppt)
                cross_val_trials = trials - 1
                trial_trf = np.zeros(shape=(len(ridge_parameters), cross_val_trials), dtype=np.ndarray)
                for trial in range(0, trials - 1):
                    print(trial)
                    ppt_trial_data = cross.data[ppt][trial]

                    sampling_rate, signal_data, time_array = cross.read_stimulus(condition, trial)
                    stim_mat = Matrices()
                    # env_matrix = stim_mat.construct_phoneme_features_matrix(condition, trial, time_array, sampling_rate,
                    #                                                    desired_freq,
                    #                                                    lag_min, lag_max)
                    env_matrix = stim_mat.construct_envelope_matrix(signal_data, time_array, sampling_rate,
                                                                    desired_freq,
                                                                    lag_min, lag_max)

                    autocovariance = np.matmul(env_matrix, env_matrix.transpose())
                    s_t_r = np.matmul(env_matrix, ppt_trial_data)
                    del env_matrix
                    del sampling_rate
                    del signal_data
                    del time_array
                    del stim_mat
                    cross_val_trfs = cross.trial_parameter(autocovariance, s_t_r, ridge_parameters)
                    trial_trf[:, trial] = cross_val_trfs

                print('finished trial cycle')
                initial_prediction_value, initial_lambda_index = 0, 0
                first_test_index = trials - 2
                cross.build_prediction(ppt, trial_trf, first_test_index, initial_prediction_value, initial_lambda_index,
                                       condition)
                print('finished build')
                final_lambda = cross.optimal_lambda()
                final_trial = trials - 1
                check.append(trial_trf[int(final_lambda), :])
                r_values = cross.final_model(trial_trf[int(final_lambda), :], desired_freq, lag_min, lag_max, ppt,
                                             final_trial)
                print('r tests')
                interpretation.r_averages(r_values, condition, ppt)
