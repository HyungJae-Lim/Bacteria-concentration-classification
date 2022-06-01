from sklearn.externals import joblib
import pandas as pd
import os
import numpy as np


class Regression():
    def __init__(self, data_csv_path):
        '''
        This class is for predicting a real value of tif files.
        :param:
            data_csv_path: path of data csv
        '''

        self.data_csv_path = data_csv_path


    def pred(self, model, data):
        '''
        This function is for predicting values with log10.

        :param model: path of a trained model
        :param data: data consisted of class

        :return:
            DataFrame : results of regression
        '''

        features = ['peak', 'tail', 'max', 'min', 'one_sigma1', 'one_sigma2', 'outlier_cnt_mean', 'div60']

        X_test = data[features].values

        model = joblib.load(model)
        pred_log = model.predict(X_test)


        predict_results = pd.concat([data.reset_index(drop=True), pd.DataFrame(pred_log, columns=['pred_log'])], axis=1)
        predict_results['pred_real'] = np.power(10, predict_results['pred_log'])

        return predict_results


    def save_results_regression(self, raspberry=False):
        '''
        Saving regression results.

        :return:
            path : path of final results.
        '''

        data = pd.read_csv(self.data_csv_path)

        if raspberry == True:
            model_path = "./models/regression/svr/"
            filename = "./results/regression/svr_regression_results.csv"

        else:
            model_path = "./models/regression/lgb"
            filename = "./results/regression/lgb_regression_results.csv"


        for i in os.listdir(model_path):
            clss = i.split("_")[0]
            model = os.path.join(model_path, i)

            if clss == "one":
                test = data[data["new_class"] == 0 ]
                one_class_results = self.pred(model, test)
            elif clss == "two":
                test = data[data["new_class"]== 1 ]
                two_class_results = self.pred(model, test)
            elif clss == "three":
                test = data[data["new_class"]== 2 ]
                three_class_results = self.pred(model, test)
            elif clss == "four":
                test = data[data["new_class"] == 3 ]
                four_class_results = self.pred(model, test)
            elif clss == "five":
                test = data[data["new_class"] == 4 ]
                five_class_results = self.pred(model, test)
            elif clss == "six":
                test = data[data["new_class"] == 5 ]
                six_class_results = self.pred(model, test)
            elif clss == "seven":
                test = data[data["new_class"] == 6 ]
                seven_class_results = self.pred(model, test)

        final_results = pd.concat([one_class_results, two_class_results, three_class_results,
                                       four_class_results, five_class_results, six_class_results, seven_class_results], axis=0)

        final_results.to_csv(filename, index=False)

        print("Finish regression !")

        return filename
