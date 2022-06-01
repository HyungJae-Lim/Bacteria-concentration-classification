import pandas as pd
import numpy as np

class ExtractLabel:
    def __init__(self, data_path):
        '''
        Training model for classification task.

        :param
            data_path: path of csv file after pre-processing.
        '''
        self.data_path = data_path

    def make_date(self):
        '''
        This function is for making date column for training which is robust on date.

        :return:
            DataFrame : data with date.
        '''

        data = pd.read_csv(self.data_path)

        data['date'] = data['file_name'].str.split('/').str[-3].str[:6].astype(int)

        return data

    def make_label(self):
        '''
        This function is for labeling dataset.

        *class : (each concentration - 1)
        *label_first : label for first classification which classify low(1, 2, 3, 4) and high(5, 6, 7).
                       low(1, 2, 3, 4) concentrations are 0 and high(5, 6, 7) concentrations are 1.
        *label_second : label for second classification which classify 5 ,6 and 7 concentrations.
                        label is same as class (NAN for low(1, 2, 3, 4) concentration).
        *label_third : label for third classification which classify low(4) concentration and very low(1, 2, 3) concentration.
                       low(4) concentration is 1 and very low(1, 2, 3) concentration is 0 (NAN for high(5, 6, 7) concentrations).
        *label_fourth : label for fourth classification which classify very low(1, 2, 3) concentrations.
                        label is same as class (NAN for low(4) and high(5, 6, 7) concentrations).
        *real_label : label for regression task.
        *log_label : log10(real_label).

        :return:
            DataFrame : data with labels for classification.
        '''

        data = self.make_date()


        #classification label

        data['class'] = data['file_name'].str.split('/').str[-1].str.split('_').str[0].str[2].astype(int) - 1

        data['label_first'] = 0
        data.loc[data['class'] == 6, 'label_first'] = 1
        data.loc[data['class'] == 5, 'label_first'] = 1
        data.loc[data['class'] == 4, 'label_first'] = 1

        data.loc[data['class'] == 6, 'label_second'] = 6
        data.loc[data['class'] == 5, 'label_second'] = 5
        data.loc[data['class'] == 4, 'label_second'] = 4

        data.loc[data['class'] == 3, 'label_third'] = 1
        data.loc[data['class'] == 2, 'label_third'] = 0
        data.loc[data['class'] == 1, 'label_third'] = 0
        data.loc[data['class'] == 0, 'label_third'] = 0

        data.loc[data['class'] == 2, 'label_fourth'] = 2
        data.loc[data['class'] == 1, 'label_fourth'] = 1
        data.loc[data['class'] == 0, 'label_fourth'] = 0

        # regression label
        label_1 = data['file_name'].str.split(")").str[0].str.split("=").str[1].astype(float)
        label_2 = data['file_name'].str.split(")").str[1].str.split("=").str[1].astype(float)

        data.loc[data['class'] == 0, 'real_label'] = label_1
        data.loc[data['class'] == 1, 'real_label'] = label_2
        data.loc[data['class'] == 2, 'real_label'] = (label_2 * 10)
        data.loc[data['class'] == 3, 'real_label'] = (label_2 * 100)
        data.loc[data['class'] == 4, 'real_label'] = (label_2 * 1000)
        data.loc[data['class'] == 5, 'real_label'] = (label_2 * 10000)
        data.loc[data['class'] == 6, 'real_label'] = (label_2 * 100000)

        data['log_label'] = np.log10(data['real_label'])

        data_path = "../train/data_with_labels.csv"

        data.to_csv(data_path, index=False)

        print("Finish extracting labels !")

        return data_path