import scipy.io as sio
import numpy as np
from scipy import stats


class Response:

    def __init__(self):
        self.conditions = ('sensical', 'nonsensical', 'sensical_ctrl', 'nonsensical_ctrl')

    def read_data(self, cnd, ppt):
        """reads in EEG data in matrix format.

        arguments and variables:
        cnd: condition being read
        ppt: participant number
        data:  32x15360x30 (number of channels x number of time points x number of trials)

        returns:
        read data
        """

        d = sio.loadmat(
            '../Preprocessed_EEG_data_individual_trials/trial_data_ppt_' + str(ppt) + '_block_' + self.conditions[cnd] + '.mat')
        data = d['data_for_python']
        return data

    def load_all_data(self):
        """all stored EEG data stored in the appropriate location

        variable:
        total: all EEG data

        returns:
        all EEG data for all trials and participants under every experimental condition
        """

        number_of_ppts = 18
        total = np.ndarray(shape=(number_of_ppts, len(self.conditions)), dtype=np.ndarray)
        for i, cnd in enumerate(self.conditions):
            for ppt in range(1, number_of_ppts + 1):
                data = self.read_data(i, ppt)
                total[ppt - 1][i] = data

        return total

    def load_specific_condition(self, cnd, lag, trim):
        """gathers EEG data from specified conditions for all participants.

        variable:
        collected_data: all EEG data associated with the single specified condition

        returns:
        all EEG data for all trials and participants under a single experimental condition. Trimmed
        to match length of stimulus data and normalised.
        """

        number_of_ppts = 18
        number_of_trials = 30
        skipped = []
        collected_data = np.ndarray(number_of_ppts, dtype=np.ndarray)
        for ppt in range(1, number_of_ppts + 1):
            trials = np.zeros(number_of_trials, dtype=np.ndarray)
            data = self.read_data(cnd, ppt)
            if data.shape[2] != 30:
                skipped.append(ppt - 1)
            else:
                for trial in range(0, data.shape[2]):
                    trials[trial] = self.trim_normalise(data[:, :, trial], lag, trim)
            collected_data[ppt - 1] = trials
        return collected_data, skipped

    def load_specific_participants(self, participants):
        """gathers EEG data from specified participants for all conditions

        variable:
        collected_data: all EEG data associated with specified participants

        returns:
        all EEG data for all trials and  experimental conditions for the selected participants
        """

        if type(participants) == int:
            rows = 1
        else:
            rows = len(participants)

        collected_data = np.ndarray(shape=(rows, len(self.conditions)), dtype=np.ndarray)
        for i, cnd in enumerate(self.conditions):
            if rows == 1:
                data = self.read_data(cnd, participants)
                collected_data[participants - 1][i] = data
            else:
                for ppt in participants:
                    data = self.read_data(cnd, ppt)
                    collected_data[ppt - 1][i] = data

        return collected_data

    def load_specific_condition_trial(self, condition, trial):

        number_of_ppts = 18
        collected_data = []
        for ppt in range(1, number_of_ppts + 1):
            if condition == 1 and (ppt == 4 or ppt == 7):
                continue
            if condition == 2 and (ppt == 7 or ppt == 8):
                continue
            data = self.read_data(condition, ppt)
            data = data[:, :, trial]
            collected_data.append(data)

        return np.asarray(collected_data)

    def load_specific_ppt_condition_trial(self, ppt, condition, trial, lag=None, trim=None):
        if condition == 1 and (ppt == 4 or ppt == 7):
            return
        if condition == 2 and (ppt == 7 or ppt == 8):
            return
        else:
            data = self.read_data(condition, ppt + 1)
            data = data[:, :, trial]

        if trim != None:
            data = self.trim_normalise(data, lag, trim)

        return data

    def trim_normalise(self, data, lag, trim):
        electrodes = data.shape[0]
        length = data[0, lag: - trim].size
        normalised = np.zeros(shape=(length, electrodes))
        for electrode in range(0, electrodes):
             trim_data = data[electrode, lag: -trim]
             normalised[:, electrode] = stats.zscore(trim_data)
             normalised[:, electrode ] = trim_data

        return normalised
    def check_condition_index(self, condition):
        """returns the index location of the requested condition from Response.conditions
        """
        return self.conditions.index(condition)

    def check_condition(self, index):
        """returns the condition at the requested index from Response.conditions
        """
        return self.conditions[index]


if __name__ == '__main__':
    low_trim = 1.28
    high_trim = 15.36
    participants = 18
    trials = 30
    low_trim_points = int(low_trim * 1000)

    response = Response()
    conditions = ('sense', 'nonsense', 'sense_control', 'nonsense_control')

    # for index, cnd in enumerate(conditions):
    #     for ppt in range(1, 19):
    #         data = response.read_data(index, ppt)
    #         if data.shape[2] != 30:
    #             print(f'here: {cnd}, {index}, {ppt}, {data.shape}')
    #             """highlights issue with certain block would ignore however some are larger than 30 which is
    #             more than the number of trials so will remove all in the sake of caution"""

    data, skipped = response.load_specific_condition(0, 200, low_trim_points)
    print(data.shape)
    print(data[0][29].shape)
    fistppt = data[0][28]
    secondppt = data[1][28]
    print(np.array_equal(fistppt, secondppt))



