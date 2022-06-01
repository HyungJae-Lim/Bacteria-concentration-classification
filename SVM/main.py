from preprocessing import preprocessingtif
from test import classification
from test import regression


'''
    Author: NOTA
    Date created: 07/12/2019
    Date last modified: 07/16/2019
    Python Version: 3.5.2
'''


if __name__ == '__main__':
    # extracted_feature_path = preprocessingtif.FeatureExtractor().save_features()
    extracted_feature_path = "./test/features_only_cam3.csv"
    scaling = [False, True, False, True]
    classified_files = classification.Classification(extracted_feature_path, scaling).save_results_classification()
    for i in (True, False):
        regression.Regression(classified_files).save_results_regression(raspberry=i)
   