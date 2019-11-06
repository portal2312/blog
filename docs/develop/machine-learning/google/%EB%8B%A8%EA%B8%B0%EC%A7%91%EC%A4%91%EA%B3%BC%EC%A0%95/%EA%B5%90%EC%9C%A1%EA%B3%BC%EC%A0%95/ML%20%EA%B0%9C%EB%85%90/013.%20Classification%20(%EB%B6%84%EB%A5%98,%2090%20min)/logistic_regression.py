"""로지스틱 회귀

- 이전 실습의 주택 가격 중간값 예측 모델을 이진 분류 모델로 재편한다
- 이진 분류 문제에서 로지스틱 회귀와 선형 회귀의 효과를 비교한다

이전 실습과 동일하게 캘리포니아 주택 데이터 세트를 사용하되, 이번에는
특정 지역의 거주 비용이 높은지 여부를 예측하는 이진 분류 문제로 바꿔 보겠습니다.
또한 기본 특성으로 일단 되돌리겠습니다.

## 이진 분류 문제로 전환
데이터 세트의 타겟은 숫자(연속 값) 특성인 median_house_value 입니다.
이 연속 값에 임계값을 적용하여 부울 라벨을 만들 수 있습니다.

특정 지역을 나타내는 특성이 주어질 때 거주 비용이 높은 지역인지를 예측하려고 합니다.
데이터 학습 및 평가를 위한 타겟을 준비하기 위해, 분류 임계값을 주택 가격 중앙값에 대한 75번째 백분위수(약 265,000)로 정의하겠습니다.
주택 가격이 임계값보다 높으면 라벨이 1로, 그렇지 않으면 라벨이 0으로 지정됩니다.
"""
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))


# 아래 코드가 이전 실습에 비해 다른 점을 확인하세요.
# median_house_value를 타겟으로 사용하는 대신 median_house_value_is_high라는 이진 타겟을 새로 만들고 있습니다.
def preprocess_features(california_housing_dataframe):
    """Prepares input features from California housing data set.

    Args:
      california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
      A DataFrame that contains the features to be used for the model, including
      synthetic features.
    """
    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
            california_housing_dataframe["total_rooms"] /
            california_housing_dataframe["population"])
    return processed_features


def preprocess_targets(california_housing_dataframe):
    """Prepares target features (i.e., labels) from California housing data set.

    Args:
      california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
      A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Create a boolean categorical feature representing whether the
    # median_house_value is above a set threshold.
    output_targets["median_house_value_is_high"] = (
            california_housing_dataframe["median_house_value"] > 265000).astype(float)
    return output_targets


# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
# Training examples summary:
#        latitude  longitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  rooms_per_person
# count   12000.0    12000.0             12000.0      12000.0         12000.0     12000.0     12000.0        12000.0           12000.0
# mean       35.6     -119.6                28.6       2635.6           538.0      1424.1       500.0            3.9               2.0
# std         2.1        2.0                12.6       2164.9           417.5      1138.3       382.1            1.9               1.2
# min        32.5     -124.3                 1.0          2.0             1.0         3.0         1.0            0.5               0.1
# 25%        33.9     -121.8                18.0       1457.0           295.8       786.8       280.0            2.6               1.5
# 50%        34.2     -118.5                29.0       2128.5           434.0      1166.0       409.0            3.5               1.9
# 75%        37.7     -118.0                37.0       3148.0           647.0      1719.0       604.0            4.8               2.3
# max        42.0     -114.3                52.0      37937.0          5471.0     35682.0      5189.0           15.0              55.2
print("Validation examples summary:")
display.display(validation_examples.describe())
# Validation examples summary:
#        latitude  longitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  rooms_per_person
# count    5000.0     5000.0              5000.0       5000.0          5000.0      5000.0      5000.0         5000.0            5000.0
# mean       35.6     -119.5                28.6       2662.9           542.8      1442.7       504.2            3.9               2.0
# std         2.1        2.0                12.4       2215.7           430.9      1170.5       390.3            1.9               1.1
# min        32.6     -124.3                 1.0         11.0             3.0         9.0         3.0            0.5               0.0
# 25%        33.9     -121.8                18.0       1470.8           299.0       795.0       284.0            2.6               1.5
# 50%        34.2     -118.5                28.0       2125.5           433.0      1169.0       409.0            3.6               1.9
# 75%        37.7     -118.0                37.0       3157.0           652.2      1730.2       607.0            4.8               2.3
# max        42.0     -114.5                52.0      32627.0          6445.0     28566.0      6082.0           15.0              52.0
print("Training targets summary:")
display.display(training_targets.describe())
# Training targets summary:
#        median_house_value_is_high
# count                     12000.0
# mean                          0.3
# std                           0.4
# min                           0.0
# 25%                           0.0
# 50%                           0.0
# 75%                           1.0
# max                           1.0
print("Validation targets summary:")
display.display(validation_targets.describe())
# Validation targets summary:
#        median_house_value_is_high
# count                      5000.0
# mean                          0.2
# std                           0.4
# min                           0.0
# 25%                           0.0
# 50%                           0.0
# 75%                           0.0
# max                           1.0


# 선형 회귀의 성능 측정
# 로지스틱 회귀가 효과적인 이유를 확인하기 위해, 우선 선형 회귀를 사용하는 단순 모델을 학습시켜 보겠습니다.
# 이 모델에서는 {0, 1} 집합에 속하는 값을 갖는 라벨을 사용하며 0 또는 1에 최대한 가까운 연속 값을 예측하려고 시도합니다.
# 또한 출력을 확률로 해석하려고 하므로 (0, 1) 범위 내에서 출력되는 것이 이상적입니다.
# 그런 다음 임계값 0.5를 적용하여 라벨을 결정합니다.
#
# 아래 셀을 실행하여 LinearRegressor로 선형 회귀 모델을 학습시킵니다.
def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
      input_features: The names of the numerical input features to use.
    Returns:
      A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_linear_regressor_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.

    Returns:
      A `LinearRegressor` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["median_house_value_is_high"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value_is_high"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value_is_high"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    return linear_regressor


linear_regressor = train_linear_regressor_model(
    learning_rate=0.000001,
    steps=200,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
# Training model...
# RMSE (on training data):
#   period 00 : 0.45
#   period 01 : 0.45
#   period 02 : 0.44
#   period 03 : 0.44
#   period 04 : 0.44
#   period 05 : 0.44
#   period 06 : 0.44
#   period 07 : 0.44
#   period 08 : 0.44
#   period 09 : 0.44
# Model training finished.

# 작업 1: 예측의 LogLoss 계산 가능성 확인
# 예측을 조사하여 LogLoss를 계산하는 데 사용될 수 있는지 확인합니다.
#
# LinearRegressor는 L2 손실을 사용하므로 출력이 확률로 해석되는 경우 오분류에 효과적으로 페널티를 부여하지 못합니다.
# 예를 들어 음성 예가 양성으로 분류되는 확률이 0.9인 경우와 0.9999인 경우에는
# 커다란 차이가 있어야 하지만 L2 손실은 두 경우를 분명하게 구분하지 않습니다.
#
# 반면, LogLoss는 이러한 "신뢰 오차"에 훨씬 큰 페널티를 부여합니다. LogLoss는 다음과 같이 정의됩니다.
# LogLoss=∑(x,y)∈D−y⋅log(ypred)−(1−y)⋅log(1−ypred)
#
# 하지만 우선 예측 값을 가져오는 것이 먼저입니다.
# LinearRegressor.predict를 사용하여 값을 가져올 수 있습니다.
#
# 예측과 타겟이 주어지면 LogLoss를 계산할 수 있는지 확인해 보세요.
predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                  validation_targets["median_house_value_is_high"],
                                                  num_epochs=1,
                                                  shuffle=False)

validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

_ = plt.hist(validation_predictions)


# 작업 2: 로지스틱 회귀 모델을 학습시키고 검증세트로 LogLoss 계산
# 로지스틱 회귀를 사용하려면 LinearRegressor 대신 LinearClassifier를 사용합니다. 아래 코드를 완성하세요.
#
# NOTE: LinearClassifier 모델에서 train() 및 predict()를 실행할 때는
# 반환된 dict의 "probabilities" 키를 통해 예측된 실수값 확률에 액세스할 수 있습니다(예: predictions["probabilities"]).
# Sklearn의 log_loss 함수를 사용하면 이러한 확률로 LogLoss를 편리하게 계산할 수 있습니다.
def train_linear_classifier_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear classification model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.

    Returns:
      A `LinearClassifier` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear classifier object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    # YOUR CODE HERE: Construct the linear classifier.
    #
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["median_house_value_is_high"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value_is_high"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value_is_high"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on training data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        # Logistic Regression 의 손실 함수인 로그 손실(LogLoss) 함수 사용하기
        # (LogLoss 함수: 이진 로지스틱 회귀에 사용되는 손실 함수)
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()

    return linear_classifier


linear_classifier = train_linear_classifier_model(
    learning_rate=0.000005,
    steps=500,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
# Training model...
# LogLoss (on training data):
#   period 00 : 0.59
#   period 01 : 0.58
#   period 02 : 0.57
#   period 03 : 0.55
#   period 04 : 0.54
#   period 05 : 0.54
#   period 06 : 0.53
#   period 07 : 0.53
#   period 08 : 0.53
#   period 09 : 0.53
# Model training finished.


# 작업 3: 검증 세트로 정확성 계산 및 ROC 곡선 도식화
# 분류에 유용한 몇 가지 측정항목은 모델 정확성, ROC 곡선 및 AUC(ROC 곡선 아래 영역)입니다.
# 이러한 측정항목을 조사해 보겠습니다.
#
# LinearClassifier.evaluate 는 정확성 및 AUC 등의 유용한 측정항목을 계산합니다.
# AUC(ROC 곡선 아래 영역): 무작위로 선택한 긍정 예가
#                        실제로 긍정일 가능성이 무작위로 선택한 부정 예가
#                        긍정일 가능성보다 높다고 분류자가 신뢰할 확률입니다.
evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
# {'accuracy': 0.7412,
#  'accuracy_baseline': 0.74380004,
#  'auc': 0.7292675,
#  'auc_precision_recall': 0.44418782,
#  'average_loss': 0.5302442,
#  'label/mean': 0.2562,
#  'loss': 0.5302442,
#  'precision': 0.4679803,
#  'prediction/mean': 0.27995563,
#  'recall': 0.074160814,
#  'global_step': 500}

print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])  # AUC: 가능한 모든 분류 임계값을 고려하는 평가 측정항목
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])  # 정확성
# AUC on the validation set: 0.74
# Accuracy on the validation set: 0.75

# LinearClassifier.predict 및 Sklearn 의 roc_curve 등으로 계산되는 클래스 확률을 사용하여
# ROC 곡선을 도식화하는 데 필요한 참양성률 및 거짓양성률을 가져올 수 있습니다.
# predict_validation_input_fn = lambda: my_input_fn(validation_examples,
#                                                   validation_targets["median_house_value_is_high"],
#                                                   num_epochs=1,
#                                                   shuffle=False)
validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
# Get just the probabilities for the positive class.
validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])
# print(validation_probabilities)
# array([0.3019569 , 0.09113835, 0.08383843, ..., 0.65122783, 0.1643126 ,
#        0.06399128], dtype=float32)

# FPR, TPR, 임계치 = metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities)
# print(false_positive_rate)
# array([0.        , 0.        , 0.00134445, ..., 0.99596666, 1.        ,
#        1.        ])
# print(true_positive_rate)
# array([0.00000000e+00, 7.80640125e-04, 7.80640125e-04, ...,
#        9.99219360e-01, 9.99219360e-01, 1.00000000e+00])
# print(thresholds)
# array([1.9917972e+00, 9.9179721e-01, 9.7625685e-01, ..., 2.7387883e-04,
#        1.5546746e-06, 8.9336703e-07], dtype=float32)

plt.plot(false_positive_rate, true_positive_rate, label="our model")  # plt.plot(X, Y, label=None), ROC 그래프 그리기
plt.plot([0, 1], [0, 1], label="random classifier")
_ = plt.legend(loc=2)

# 작업 2에서 학습시킨 모델의 학습 설정을 조정하여 AUC 를 개선할 수 있는지 확인해 보세요.
# 어떤 측정항목을 개선하면 다른 측정항목이 악화되는 경우가 종종 나타나므로, 적절하게 균형이 맞는 설정을 찾아야 합니다.
# 모든 측정항목이 동시에 개선되는지 확인해 보세요.
#
# 효과적인 해결 방법 중 하나는
# 과적합이 나타나지 않는 범위 내에서 더 오랫동안 학습하는 것. 이렇게 하려면 단계 수, 배치 크기 또는 둘 모두를 늘리면 됩니다. (조기 중단)
#
# 모든 측정항목이 동시에 개선되므로 손실 측정항목은 AUC 와 정확성 모두를 적절히 대변합니다.
#
# AUC 를 몇 단위만 개선하려 해도 굉장히 많은 추가 반복이 필요합니다.
# 이는 흔히 나타나는 상황이지만, 이렇게 작은 개선이라도 비용을 투자할 가치는 충분합니다.
linear_classifier = train_linear_classifier_model(
    learning_rate=0.000003,
    steps=20000,
    batch_size=500,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
# Training model...
# LogLoss (on training data):
#   period 00 : 0.50
#   period 01 : 0.48
#   period 02 : 0.48
#   period 03 : 0.47
#   period 04 : 0.47
#   period 05 : 0.47
#   period 06 : 0.47
#   period 07 : 0.47
#   period 08 : 0.47
#   period 09 : 0.47
# Model training finished.
# AUC on the validation set: 0.82
# Accuracy on the validation set: 0.78

validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities)
# array([0.00000000e+00, 2.68889486e-04, 2.68889486e-04, ...,
#        9.97042216e-01, 1.00000000e+00, 1.00000000e+00]),
# array([0.00000000e+00, 0.00000000e+00, 7.80640125e-04, ...,
#        9.99219360e-01, 9.99219360e-01, 1.00000000e+00]),
# array([1.9987952e+00, 9.9879515e-01, 9.9276358e-01, ..., 2.3239900e-05,
#        7.6034219e-08, 1.0486755e-10], dtype=float32)
plt.plot(false_positive_rate, true_positive_rate, label="our model")
plt.plot([0, 1], [0, 1], label="random classifier")
_ = plt.legend(loc=2)
