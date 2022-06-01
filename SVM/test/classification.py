import pandas as pd
from sklearn import preprocessing
import pickle

class Classification:
    def __init__(self, data_csv_path, scaling):
        '''
        This class is for classifying concentration.
        :param:
            data_csv_path: path of data csv
            scaling : flag for scaling
        '''

        self.data_csv_path = data_csv_path
        self.scaling = scaling

    def first_classification(self):
        '''
        Classify low(class:1, 2, 3, 4) and high(5, 6, 7) concentration.

        :return: The results of low and high concentration.
        '''

        data = pd.read_csv(self.data_csv_path)

        model_path = "./models/classification/first_classification.sav"
        features = ["peak", "tail", 'max', 'min', 'one_sigma1', 'one_sigma2', 'outlier_cnt_mean']

        loaded_model = pickle.load(open(model_path, 'rb'))

        X_test = data[features]

        if self.scaling[0] == True:
            X_test = preprocessing.scale(X_test)

        prediction = pd.DataFrame(loaded_model.predict(X_test),columns=['pred_first'])
        res_first = pd.concat([data,prediction],axis=1)

        res_first.to_csv("./results/classification/first_classification.csv",index=False)

        return res_first


    def second_classification(self):
        '''
        Classify high(5, 6, 7) concentration.

        :return:
            csv : results of classification.
        '''

        data = self.first_classification()
        model_path = "./models/classification/second_classification.sav"
        features = ["peak", "tail", 'max', 'min','one_sigma1', 'one_sigma2', 'outlier_cnt_mean', 'div60']

        loaded_model = pickle.load(open(model_path, 'rb'))

        test = data[data["pred_first"] == 1]

        X_test = test[features]
        y_test = test["file_name"]

        if self.scaling[1] == True:
            X_test = preprocessing.scale(X_test)

        prediction = pd.concat([y_test.reset_index(drop=True),pd.DataFrame(loaded_model.predict(X_test), columns=['pred_second']).reset_index(drop=True)],axis=1)
        res_second = pd.merge(data,prediction,how="outer",on=["file_name"])
        res_second.to_csv("./results/classification/second_classification.csv", index=False)

        return res_second


    def third_classification(self):
        '''
        Classify low(4) and very low(1,2,3) concentration.

        :return:
            csv : results of classification.
        '''

        data = self.second_classification()
        model_path = "./models/classification/third_classification.sav"
        features = [ 'peak', 'tail', 'max', 'one_sigma2', 'div60']


        loaded_model = pickle.load(open(model_path, 'rb'))

        test = data[data["pred_first"] == 0 ]

        X_test = test[features]
        y_test = test["file_name"]

        if self.scaling[2] == True:
            X_test = preprocessing.scale(X_test)

        prediction = pd.concat([y_test.reset_index(drop=True),
                                pd.DataFrame(loaded_model.predict(X_test), columns=['pred_third']).reset_index(drop=True)], axis=1)

        res_third = pd.merge(data, prediction, how="outer", on=["file_name"])

        res_third.to_csv("./results/classification/third_classification.csv", index=False)

        return res_third

    def fourth_classification(self):
        '''
        Classify low(1,2,3) concentration.

        :return:
            csv : results from classification
        '''

        data = self.third_classification()
        model_path = "./models/classification/fourth_classification.sav"
        features =  ['tail', 'max', 'one_sigma2', 'div60']

        loaded_model = pickle.load(open(model_path, 'rb'))

        test = data[data["pred_third"] == 0]

        X_test = test[features]
        y_test = test["file_name"]

        if self.scaling[3] == True:
            X_test = preprocessing.scale(X_test)

        prediction = pd.concat([y_test.reset_index(drop=True),
                                pd.DataFrame(loaded_model.predict(X_test), columns=['pred_fourth']).reset_index(drop=True)], axis=1)

        res_fourth = pd.merge(data, prediction, how="outer", on=["file_name"])

        res_fourth.to_csv("./results/classification/fourth_classification.csv", index=False)

        return res_fourth

    def save_results_classification(self):
        '''
        This function is for making a new data set from classification results

        :return:
            path : results csv file path from all classifications
        '''

        data = self.fourth_classification()

        data['new_class'] = 0

        for i in range(len(data)):
            if data['pred_first'].iloc[i] == 1:
                data['new_class'].iloc[i] = data['pred_second'].iloc[i]
            elif data['pred_third'].iloc[i] == 1:
                data['new_class'].iloc[i] = 3
            else:
                data['new_class'].iloc[i] = data['pred_fourth'].iloc[i]

        final_path = "./results/classification/final_classification.csv"

        data.to_csv(final_path, index=False)

        print("Finish classification !")

        return final_path


