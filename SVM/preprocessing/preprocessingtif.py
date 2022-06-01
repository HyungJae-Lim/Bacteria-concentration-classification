import pandas as pd
import numpy as np
import os
from skimage import io
from scipy.stats import norm



class FeatureExtractor:
    '''
    This class is for extracting features from input files.
    '''

    def read_file(self, file_path, channel_first):
        '''
        Changing file format to npy array.
        :param:
            file_path: path of tif file
            channel_first : boolen type for shape of data.
        '''
        img = io.imread(file_path)
        img = np.asarray(img)
        img = img.astype(float) # img shape : (60,600,600)

        if channel_first == False:
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 0, 1)

        self.tif = img
        self.height = img.shape[1]
        self.width = img.shape[2]
        self.channel = img.shape[0]


    def extract_peak_tail(self):
        '''
        Extracting peak and tail features from delta consisted of variation from previous frame to after frame.

        * peak : The frequency of delta '1'
        * tail : The maximum variation


        :return:
            int : peak
            int : tail
        '''

        tif = self.tif

        delta_img = np.abs(tif[1:, :, :] - tif[0:-1, :, :])
        unique_elements, counts_elements = np.unique(delta_img, return_counts=True)
        peak = counts_elements[1]
        tail = np.max(unique_elements)

        return peak, tail

    def extract_div60(self):
        '''
        Extracting div60 feature from tif file.

        * div_60 : Average variation per pixel


        :return:
            float : div_60
        '''

        tif = self.tif
        height = self.height
        width = self.width
        channel = self.channel

        tif_mean = np.mean(tif, axis=0)
        sub_mean = tif - tif_mean
        square_tif = np.power(sub_mean, 2)
        square_sum = np.sum(square_tif)
        div_360000 = np.divide(square_sum, (height * width))
        div_60 = np.divide(div_360000, channel)

        return div_60

    def extract_five_features(self):
        '''
        Extracting five features from averaging per frame.

        * tif_max : Max value from averaging per frame.
        * tif_min : Min value from averaging per frame.
        * one_sigma1 : Adding standard deviation to mean from averaging per frame.
        * one_sigma2 : Subtracting standard deviation from mean from averaging per frame.
        * outlier_cnt : Counting number of values which is lie outside a band consisted of standard deviation around the mean.

        :return:
            float : tif_max, tif_min, one_sigma1, one_sigma2
            int : outlier_cnt
        '''

        tif = self.tif

        height = self.height
        width = self.width
        channel = self.channel

        graph = [np.sum(tif[frame]) / (height * width) for frame in range(channel)]

        tif_max = np.max(graph)
        tif_min = np.min(graph)

        tif_mean, tif_std = norm.fit(graph)

        one_sigma1 = tif_mean + tif_std
        one_sigma2 = tif_mean - tif_std

        outlier_cnt = 0

        for frame in range(channel):
            if (graph[frame] > one_sigma1) or (graph[frame] < one_sigma2):
                outlier_cnt +=1
        return tif_max, tif_min, one_sigma1, one_sigma2, outlier_cnt

    def save_features(self, channel_first=True, train=False):
        '''
        Saving features after extracting eight features from tif files.
        :param:
            channel_first : if data is channel_last, use "False".
            train : test mode is default. if you want to train the model, use "True".
        :return:
            path : path of csv file.
        '''

        if train == True:
            input_dir = "../train/data/"
        else:
            input_dir = "../test/inputs/"


        df = pd.DataFrame(columns=['file_name', 'max', 'min', 'one_sigma1', 'one_sigma2', 'outlier_cnt_mean', 'div60','peak', 'tail'])

        for path, _, files in os.walk(input_dir):
            for file in files:
                file_name = os.path.join(path, file)
                self.read_file(file_name, channel_first)
                max_, min_, sigma1, sigma2, outlier_cnt_mean = self.extract_five_features()
                div60 = self.extract_div60()
                peak, tail = self.extract_peak_tail()

                df = df.append(pd.DataFrame([[file_name, max_, min_, sigma1, sigma2, outlier_cnt_mean, div60, peak, tail]],
                                        columns=['file_name', 'max', 'min', 'one_sigma1', 'one_sigma2', 'outlier_cnt_mean', 'div60',
                                                 'peak', 'tail']), ignore_index=True, sort=False)

                print(file_name,"is in progress")

        if train == True:
            df.to_csv("../train/data.csv", index=False)
            results_path = "../train/data.csv"
        else:
            df.to_csv("../test/features.csv",index=False)
            results_path = "../test/features.csv"

        return results_path
