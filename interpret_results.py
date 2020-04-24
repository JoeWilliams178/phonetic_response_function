import numpy as np

class Interpretation:
    condition = None
    averages = None

    def r_averages(self, r_values, condition, ppt):
        if self.condition == condition:
            self.averages[ppt] = r_values
            if ppt == 17:
                for number, participant in enumerate(self.averages):
                    electrodes = np.zeros(shape=32)
                    for index, electrode in enumerate(participant):
                        electrodes[index] = electrode[0]
                    print(number, np.max(electrodes), np.where(electrodes == np.max(electrodes)))

        else:
            self.condition = condition
            self.averages = None
            self.averages = np.zeros(shape=18, dtype=np.ndarray)

            self.averages[ppt] = r_values



        #
        # for electrode in r_values:
        #     av_r += electrode[0]
        #
        # av_r = av_r/32
        # print('average', av_r)

