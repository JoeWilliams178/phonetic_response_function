import numpy as np
from statistics import mode

class Interpretation:
    condition = None
    averages = None
    maximums = np.zeros(shape=18, dtype=np.ndarray)
    max_index = np.zeros(shape=18, dtype=np.ndarray)

    def r_averages(self, r_values, condition, ppt):
        if self.condition == condition:
            print(ppt)
            self.averages[ppt] = r_values

            if type(self.averages[ppt]) != int:
                av_r = 0
                max_e = []
                for index, electrode in enumerate(self.averages[ppt]):
                    av_r += electrode[0]
                    max_e.append(electrode[0])

                av_r = av_r/32
                print(ppt,':', av_r)
                print('max:', ppt, max(max_e))


        else:
            self.condition = condition
            self.averages = None
            self.averages = np.zeros(shape=18, dtype=np.ndarray)
            self.averages[ppt] = r_values


    # def most_popular(self, number, participant):
    #     electrodes = np.zeros(shape=32)
    #     for index, electrode in enumerate(participant):
    #         electrodes[index] = electrode[0]
    #
    #     ordered_electrodes = np.zeros(shape=32)
    #     for electrode in range (0, 32):
    #         ordered_electrodes[electrode] = np.where(electrodes == np.max(electrodes))[0][0]
    #         electrodes[np.where(electrodes == np.max(electrodes))[0][0]] = 0
    #         # print(max_temp[maximum] = np.max(electrodes)
    #
    #     print(ordered_electrodes)

        #
        # self.maximums[number] = max_temp
        # self.max_index[number] = max_index
        # if number == 17:
        #     all_ppt_index = []
        #     for ppt in self.max_index:
        #         for value in ppt:
        #             all_ppt_index.append(int(value))
        #
        #     print(all_ppt_index)

            # from collections import Counter

            # printing initial ini_list
            # print("initial list", str(all_ppt_index))

            # sorting on bais of frequency of elements
            # result = [item for items, c in Counter(all_ppt_index).most_common()
                      # for item in [items] * c]

            # printing final result
            # print("final list", str(result))

            # from collections import Counter
            # from itertools import groupby
            #
            #
            # freqs = groupby(Counter(all_ppt_index).most_common(), lambda x: x[1])
            # print('here', freqs)
            # print([val for val, count in next(freqs)[1]])
            # # prints [3, 4, 6]
            # while all_ppt_index:
            #     if type(mode(all_ppt_index)) == int:
            #         modes.append(all_ppt_index.pop(mode(all_ppt_index)))
            #     else:
            #         modes.append(all_ppt_index.pop(mode(all_ppt_index)[0]))






        # print(number, ':', max_temp, max_index)




