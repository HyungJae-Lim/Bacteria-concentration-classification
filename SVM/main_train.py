from preprocessing import preprocessingtif
from preprocessing import label_extractor
from train import train

if __name__ == '__main__':
    # extracted_feature_path = preprocessingtif.FeatureExtractor().save_features(train=True)
    # data_path = label_extractor.ExtractLabel(extracted_feature_path).make_label()

    data_path ="./train/only_cam3.csv"
    ## classification
    classification_list = ["first", "second", "third", "fourth"]

    features = [["peak", "tail", 'max', 'min', 'one_sigma1', 'one_sigma2', 'outlier_cnt_mean'],
                ["peak", "tail", 'max', 'min','one_sigma1', 'one_sigma2', 'outlier_cnt_mean', 'div60'],
                [ 'peak', 'tail',  'max','one_sigma2', 'div60'],
                ['tail', 'max', 'one_sigma2', 'div60'],
                ]

    scaling = [False, True, False, True]

    train_date = [190508, 190509, 190514, 190515, 190516, 190517, 190520, 190521,190523,190530, 190603]
    test_date = [190513, 190522, 190610]


    max_depth = [3, 3, 8, 15]

    seed = [379, 379, 604, 520]

    for idx,item in enumerate(classification_list):
        train.Training(data_path).classification_train_and_save(train_date=train_date, test_date=test_date,
                                                                features=features[idx],
                                                                number=item,
                                                                num_depth=max_depth[idx],
                                                                num_random=seed[idx],
                                                                scaling=scaling[idx])


    ### regression

    clss_list = ["one", "two", "three", "four", "five", "six", "seven"]

    test_date = [190513, 190522, 190610]
    train_date = [190508, 190509, 190514, 190515, 190516, 190517, 190520, 190521,190523]
    valid_date = [190530, 190603]

    features = ["peak", "tail", 'max', 'min','one_sigma1', 'one_sigma2', 'outlier_cnt_mean', 'div60']

    raspberry = [False, True]

    for raspb in raspberry:
        for clss in clss_list:
            train.Training(data_path).regression_train_and_save(train_date=train_date, valid_date=valid_date, test_date=test_date,
                                                     number=clss, features=features, raspberry=raspb)
