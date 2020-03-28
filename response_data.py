import scipy.io as sio
import numpy as np


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
            'Preprocessed_EEG_data_individual_trials/trial_data_ppt_' + str(ppt) + '_block_' + cnd + '.mat')
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
                data = self.read_data(cnd, ppt)
                total[ppt - 1][i] = data

        return total

    def load_specific_condition(self, cnd):
        """gathers EEG data from specified conditions for all participants

        variable:
        collected_data: all EEG data associated with the single specified condition

        returns:
        all EEG data for all trials and participants under a single experimental condition
        """

        number_of_ppts = 18
        collected_data = np.ndarray(number_of_ppts, dtype=np.ndarray)
        for ppt in range(1, number_of_ppts + 1):
            data = self.read_data(cnd, ppt)
            collected_data[ppt - 1] = data

        return collected_data

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
                    print(ppt)
                    data = self.read_data(cnd, ppt)
                    collected_data[ppt - 1][i] = data

        return collected_data

    def check_condition_index(self, condition):
        """returns the index location of the requested condition from Response.conditions
        """
        return self.conditions.index(condition)

    def check_condition(self, index):
        """returns the condition at the requested index from Response.conditions
        """
        return self.conditions[index]


if __name__ == '__main__':

    response = Response()
    print(response.check_condition_index('nonsensical'))
    print(response.check_condition(0))



