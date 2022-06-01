import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.svm import SVR
import lightgbm as lgb
import pickle
import numpy as np
from math import sqrt


class Training:
    def __init__(self, data_path):
        '''

        :param
            data_path:
        '''
        self.data_path = data_path


    def classification_train_and_save(self, train_date, test_date, features, number, num_depth, num_random, scaling=False):
        '''
        Training a model and saving.

        :param
            date: using for training(do not use for test dataset).
            features: using for training.
            number: "first" for first classification, "second" for second classification, "third" for third classification and "fourth" for last classification.
                    (details in manual)
            num_depth: number of max depth.
            num_random: fix seed for random.
            scaling: standard scaling is not default. if want to scale data, use  "True".
        :return:
        '''

        data = pd.read_csv(self.data_path)
        label = "label_" + number

        if number == "second":
            data = data[data['class']>3]
        elif number == "third":
            data = data[data['class']<4]
        elif number == "fourth":
            data = data[data['class']<3]

        train = data[data['date'].isin(train_date)]
        test = data[data['date'].isin(test_date)]

        X_train = train[features]
        y_train = train[label]

        X_test = test[features]

        if scaling == True:
            X_train = preprocessing.scale(X_train)
            X_test = preprocessing.scale(X_test)

        model = DecisionTreeClassifier(max_depth=num_depth, random_state=num_random)
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predict_results = pd.concat([test.reset_index(drop=True), pd.DataFrame(y_pred, columns=["pred_"+number])], axis=1)
        acc = accuracy_score(predict_results['label_'+number],predict_results['pred_'+number])

        print("Teat date :",pd.unique(test['date']))
        print("Class :", pd.unique(test['class']))
        print("Using features :", features)
        print("Test acc : %.2f%%" % (acc * 100))

        cm=confusion_matrix(predict_results['label_'+number],predict_results['pred_'+number],labels=pd.unique(train[label]))
        print("confusion matrix")
        print(cm)

        predict_results.to_csv("./train/results_train/classification/classification_results_"+number+".csv", index=False)

        filename = "./models/classification/"+ number + "_classification.sav"

        pickle.dump(model, open(filename, 'wb'))

        return predict_results


    def regression_train_and_save(self, train_date, valid_date, test_date, number, features, raspberry=False):
        '''
        Training a model and saving.

        :param
            train_date: date of train.
            valid_date: date of valid only use if raspberry=True.
            test_date: date of test.
            number: each class of concentration.
                    e.g. "one" for 10^1 class, "two" for 10^2 class, "three" for 10^3 class, "four" for 10^4 class,
                         "five" for 10^5 class, "six" for 10^6 class, "seven" for 10^7 class.
            features: using for training.
            raspberry: for using raspberry pi, use "True"(detail in manual).
        '''

        data = pd.read_csv(self.data_path)

        if number == "one":
            data = data[data['class'] == 0]
        elif number == "two":
            data = data[data['class'] == 1]
        elif number == "three":
            data = data[data['class'] == 2]
        elif number == "four":
            data = data[data['class'] == 3]
        elif number == "five":
            data = data[data['class'] == 4]
        elif number == "six":
            data = data[data['class'] == 5]
        elif number == "seven":
            data = data[data['class'] == 6]

        if raspberry == True:
            train = data[data['date'].isin(train_date+valid_date)]
        else:
            train = data[data['date'].isin(train_date)]
        test = data[data['date'].isin(test_date)]

        X_train = train[features].values
        y_train = train['log_label'].values

        X_test = test[features].values

        if raspberry == True:
            filename = "./models/regression/svr/" + number + "_class_svr_regression.pkl"
            model = SVR(gamma='scale', C=1.0, epsilon=0.2)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        else:
            valid = data[data['date'].isin(valid_date)]

            X_valid = valid[features].values
            y_valid = valid['log_label'].values

            d_train = lgb.Dataset(X_train, label=y_train)
            d_valid = lgb.Dataset(X_valid, label=y_valid)
            watchlist = [d_train, d_valid]

            filename = "./models/regression/lgb/" + number + "_class_lgb_regression.pkl"

            params = {'application': 'regression',
                      'boosting': 'gbdt',
                      'metric': 'mean_absolute_percentage_error',
                      'num_leaves': 2,
                      'max_depth': 2,
                      'learning_rate': 0.001,
                      'subsample': 0.8,
                      'feature_fraction': 0.5,
                      'lambda_l1': 0.6,
                      'lambda_l2': 0.6,
                      'verbosity': -1
                      }

            print("training LGB:")
            model = lgb.train(params,
                              train_set=d_train,
                              verbose_eval= 100,
                              num_boost_round= 50000,
                              early_stopping_rounds= 500,
                              valid_sets=watchlist)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        predict_results = pd.concat([test.reset_index(drop=True), pd.DataFrame(y_pred, columns=['pred_log'])], axis=1)
        predict_results['pred_real'] = np.power(10, predict_results['pred_log'])

        if raspberry == True:
            results_csv_path = "./train/results_train/regression/svr/regression_results_"+ number +".csv"
        else:
            results_csv_path = "./train/results_train/regression/lgb/regression_results_"+ number +".csv"

        predict_results.to_csv(results_csv_path, index=False)

        rmse = sqrt(mean_squared_error(predict_results['real_label'], predict_results['pred_real']))

        print("RMSE:",rmse)

        joblib.dump(model, filename)

        return predict_results
