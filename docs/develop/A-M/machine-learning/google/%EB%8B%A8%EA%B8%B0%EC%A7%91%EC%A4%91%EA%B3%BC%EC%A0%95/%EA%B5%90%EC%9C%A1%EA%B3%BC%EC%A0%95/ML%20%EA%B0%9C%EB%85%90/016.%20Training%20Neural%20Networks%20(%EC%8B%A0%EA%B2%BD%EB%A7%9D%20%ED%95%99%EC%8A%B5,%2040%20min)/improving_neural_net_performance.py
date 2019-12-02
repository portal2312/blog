"""신경망 성능 개선하기
특성을 정규화하고 다양한 최적화 알고리즘을 적용하여 신경망의 성능을 높입니다.

참고: 이 실습에서 설명하는 최적화 방식은 신경망에 국한된 것이 아니며 대부분의 모델 유형을 개선하는 데 효과적입니다.

- 설정
- 신경망 학습
- 선형 조정: 입력값을 -1, 1 범위에 들어오도록 정규화하는 것이 권장되는 표준 방식
- 작업 1: 선형 조정을 사용하여 특성 정규화: 입력(특성)값을 -1, 1 척도로 정규화
- 작업 2: 다른 Optimizer 사용해 보기: Adagrad 및 Adam 옵티마이저를 사용하고 성능을 비교
- 작업 3: 대안적 정규화 방식 탐색: 다양한 특성에 대안적인 정규화를 시도하여 성능 높이기
- 선택 과제: 위도 및 경도 특성만 사용
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
        ["longitude",
         "latitude",
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
    # Scale the target to be in units of thousands of dollars.
    output_targets["median_house_value"] = (
            california_housing_dataframe["median_house_value"] / 1000.0)
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
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())

# 신경망 학습
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
    """Trains a neural network model.

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


def train_nn_regression_model(
        my_optimizer,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a neural network regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.

    Returns:
      A tuple `(estimator, training_losses, validation_losses)`:
        estimator: the trained `DNNRegressor` object.
        training_losses: a `list` containing the training loss values taken during training.
        validation_losses: a `list` containing the validation loss values taken during training.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a DNNRegresor object.
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["median_house_value"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value"],
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
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
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

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    return dnn_regressor, training_rmse, validation_rmse


_ = train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
# Training model...
# RMSE (on training data):
#   period 00 : 155.65
#   period 01 : 132.52
#   period 02 : 115.92
#   period 03 : 110.83
#   period 04 : 102.41
#   period 05 : 101.48
#   period 06 : 100.64
#   period 07 : 105.34
#   period 08 : 101.64
#   period 09 : 99.82
# Model training finished.
# Final RMSE (on training data):   99.82
# Final RMSE (on validation data): 101.05


# 선형 조정
# 입력값을 -1, 1 범위에 들어오도록 정규화하는 것이 권장되는 표준 방식입니다.
# 이렇게 하면 SGD 에서 한 차원으로 너무 크거나 다른 차원으로 너무 작은 단계를 밟을 때 고착을 방지하는 데 도움이 됩니다.
# 수치 최적화 분야에 익숙하다면 프리컨디셔너를 사용한다는 개념과 관련이 있음을 알 수 있습니다.
def linear_scale(series):
    """
    Examples:
        min_val = 0
        max_val = 4
        scale = 2.0
        x = 1
        return -0.5 (0~4 의 1, -1~1 의 -0.5 은 서로 비중이 같다)
    """
    min_val = series.min()  # 32.54
    max_val = series.max()  # 41.95
    scale = (max_val - min_val) / 2.0
    # 1.0 은 선형 조정 최대 값을 나타냄
    return series.apply(lambda x: ((x - min_val) / scale) - 1.0)
    # 2250    34.0  -->  -0.7
    # 7990    33.9  -->  -0.7
    # 7634    34.0  -->  -0.7
    # 651     32.8  -->  -0.9
    # 12050   38.6  -->   0.3
    #         ...   -->  ...
    # 7753    33.9  -->  -0.7
    # 831     32.8  -->  -1.0
    # 14474   38.1  -->   0.2
    # 1926    33.2  -->  -0.9
    # 9448    34.3  -->  -0.6
    # Name: latitude, Length: 17000, dtype: float64


# 작업 1: 선형 조정을 사용하여 특성 정규화
# 입력값을 -1, 1 척도로 정규화합니다.
#
# 5분 정도 시간을 내어 새로 정규화한 데이터를 학습하고 평가해 보세요. 어느 정도까지 성능을 높일 수 있나요?
#
# 경험적으로 보면, 입력 특성이 대략 같은 척도일 때 NN의 학습 효율이 가장 높습니다.
#
# 정규화된 데이터의 상태를 확인하세요. 실수로 특성 하나를 정규화하지 않으면 어떠한 결과가 나타날까요?
def normalize_linear_scale(examples_dataframe):
    """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
    #
    # Your code here: normalize the inputs.
    #
    # 정규화에 최소값과 최대값이 사용되므로 데이터 세트 전체에 한 번에 적용되도록 조치해야 합니다.
    processed_features = pd.DataFrame()
    processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
    processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
    processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
    processed_features["total_rooms"] = linear_scale(examples_dataframe["total_rooms"])
    processed_features["total_bedrooms"] = linear_scale(examples_dataframe["total_bedrooms"])
    processed_features["population"] = linear_scale(examples_dataframe["population"])
    processed_features["households"] = linear_scale(examples_dataframe["households"])
    processed_features["median_income"] = linear_scale(examples_dataframe["median_income"])
    processed_features["rooms_per_person"] = linear_scale(examples_dataframe["rooms_per_person"])
    return processed_features

normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
# 여기에서는 모든 데이터가 단일 DataFrame 에 있으므로 문제가 없습니다.
# 데이터 세트가 여러 개인 경우에는 학습 세트에서 추출한 정규화 매개변수를 테스트 세트에 동일하게 적용하는 것이 좋습니다.
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

_ = train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.005),
    steps=2000,
    batch_size=50,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)
# Training model...
# RMSE (on training data):
#   period 00 : 167.83
#   period 01 : 111.44
#   period 02 : 98.67
#   period 03 : 83.64
#   period 04 : 77.60
#   period 05 : 75.07
#   period 06 : 73.27
#   period 07 : 72.11
#   period 08 : 71.47
#   period 09 : 70.97
# Model training finished.
# Final RMSE (on training data):   70.97
# Final RMSE (on validation data): 69.75

# 작업 2: 다른 옵티마이저 사용해 보기
# ** `Adagrad` 및 `Adam` 옵티마이저를 사용하고 성능을 비교합니다.**
#
# 대안 중 하나는 `Adagrad` 옵티마이저입니다.
# `Adagrad` 의 핵심 개념은 모델의 각 계수에 대해 학습률을 적응적으로 조정하여 유효 학습률을 단조적으로 낮춘다는 것입니다.
# 이 방식은 볼록 문제에는 적합하지만 비볼록 문제 신경망 학습에는 이상적이지 않을 수 있습니다.
# `Adagrad`를 사용하려면 `GradientDescentOptimizer` 대신 `AdagradOptimizer`를 지정합니다.
# `Adagrad`를 사용하는 경우 학습률을 더 높여야 할 수 있습니다.
#
# 비볼록 최적화 문제에 대해서는 `Adagrad`보다 `Adam`이 효율적일 수 있습니다.
# `Adam`을 사용하려면 `tf.train.AdamOptimizer` 메소드를 호출합니다.
# 이 메소드는 선택적으로 몇 가지 초매개변수를 인수로 취하지만 이 솔루션에서는 인수 중 하나(`learning_rate`)만 지정합니다.
# 프로덕션 설정에서는 선택적 초매개변수를 신중하게 지정하고 조정해야 합니다.
_, adagrad_training_losses, adagrad_validation_losses = train_nn_regression_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.5),
    steps=500,
    batch_size=100,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)
# Training model...
# RMSE (on training data):
#
#   period 00 : 99.67
#   period 01 : 76.18
#   period 02 : 73.34
#   period 03 : 72.90
#   period 04 : 70.59
#   period 05 : 70.16
#   period 06 : 71.48
#   period 07 : 69.52
#   period 08 : 69.44
#   period 09 : 69.45
# Model training finished.
# Final RMSE (on training data):   69.45
# Final RMSE (on validation data): 69.70

_, adam_training_losses, adam_validation_losses = train_nn_regression_model(
    my_optimizer=tf.train.AdamOptimizer(learning_rate=0.009),
    steps=500,
    batch_size=100,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)
# Training model...
# RMSE (on training data):
#   period 00 : 222.64
#   period 01 : 147.85
#   period 02 : 116.42
#   period 03 : 111.32
#   period 04 : 101.30
#   period 05 : 82.31
#   period 06 : 72.07
#   period 07 : 70.35
#   period 08 : 70.80
#   period 09 : 69.48
# Model training finished.
# Final RMSE (on training data):   69.48
# Final RMSE (on validation data): 69.26

# 작업 3: 대안적 정규화 방식 탐색
# 다양한 특성에 대안적인 정규화를 시도하여 성능을 더욱 높입니다.
#
# 변환된 데이터의 요약 통계를 자세히 조사해 보면 선형 조정으로 인해 일부 특성이 -1에 가깝게 모이는 것을 알 수 있습니다.
#
# 예를 들어 여러 특성의 중앙값이 0.0이 아닌 -0.8 근처입니다.
# - pandas.DataFrame.hist:
#   DataFrame의 히스토그램을 만듭니다.
#
#   히스토그램은 데이터 분포를 나타냅니다.
#   이 함수는 `DataFrame`의 각 시리즈에서 `matplotlib.pyplot.hist()`를 호출하여 열당 하나의 막대 그래프를 만듭니다.
_ = training_examples.hist(bins=20, figsize=(18, 12), xlabelsize=2)
plt.show()


# 이러한 특성을 추가적인 방법으로 변환하면 성능이 더욱 향상될 수 있습니다.
#
# 예를 들어 로그 조정이 일부 특성에 도움이 될 수 있습니다.
# 또는 극단값을 잘라내면 척도의 나머지 부분이 더 유용해질 수 있습니다.
#
# 아래에는 몇 가지 가능한 정규화 함수가 추가로 포함되어 있습니다.
#
# 이러한 함수를 사용하거나 직접 추가해 보세요.
# 단, 타겟을 정규화하는 경우 예측을 비정규화해야 손실 측정항목을 서로 비교할 수 있습니다.
def log_normalize(series):
    return series.apply(lambda x: math.log(x+1.0))


def clip(series, clip_to_min, clip_to_max):
    """
    series 마다 값들중 이상한(최대&최소 범위 안에 속하지 않는) 값 존재시 최대&최소로 조정하기
    """
    return series.apply(lambda x: (
        min(max(x, clip_to_min), clip_to_max)))


def z_score_normalize(series):
    mean = series.mean()
    std_dv = series.std()
    return series.apply(lambda x: (x - mean) / std_dv)


def binary_threshold(series, threshold):
    return series.apply(lambda x: (1 if x > threshold else 0))


# 해결 방법:
#
# 지금까지 살펴본 내용은 데이터를 다루는 방법 중 일부에 불과합니다.
# 다른 변환이 더 좋은 효과를 보일 수도 있습니다.
def normalize(examples_dataframe):
    """Returns a version of the input `DataFrame` that has all its features normalized."""
    #
    # YOUR CODE HERE: Normalize the inputs.
    #
    processed_features = pd.DataFrame()

    # `households`, `median_income`, `total_bedrooms`: 모두 로그 공간에서 정규 분포를 나타냅니다.
    # Old:
    # processed_features["households"] = linear_scale(examples_dataframe["households"])
    # processed_features["median_income"] = linear_scale(examples_dataframe["median_income"])
    # processed_features["total_bedrooms"] = linear_scale(examples_dataframe["total_bedrooms"])
    # New:
    processed_features["households"] = log_normalize(examples_dataframe["households"])  # 169.0  -->  5.13
    processed_features["median_income"] = log_normalize(examples_dataframe["median_income"])
    processed_features["total_bedrooms"] = log_normalize(examples_dataframe["total_bedrooms"])

    # `latitude`, `longitude`, `housing_median_age`: 이전과 같이 선형 조정을 사용하는 방법이 더 좋을 수 있습니다.
    # Old:
    # processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
    # processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
    # processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
    # New:
    processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])  # 36.8 --> -0.1
    processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
    processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])

    # `population`, `totalRooms`, `rooms_per_person`
    #   - 극단적인 이상점이 몇 개 있습니다.
    #   - 이러한 점은 지나치게 극단적이므로 로그 정규화도 도움이 되지 않습니다. 따라서 삭제하기로 하겠습니다.
    # Old:
    # processed_features["population"] = linear_scale(examples_dataframe["population"])
    # processed_features["total_rooms"] = linear_scale(examples_dataframe["total_rooms"])
    # processed_features["rooms_per_person"] = linear_scale(examples_dataframe["rooms_per_person"])
    # New:
    processed_features["population"] = linear_scale(clip(examples_dataframe["population"], 0, 5000))  # 781.0  -->  -0.7
    processed_features["rooms_per_person"] = linear_scale(clip(examples_dataframe["rooms_per_person"], 0, 5))
    processed_features["total_rooms"] = linear_scale(clip(examples_dataframe["total_rooms"], 0, 10000))

    return processed_features


normalized_dataframe = normalize(preprocess_features(california_housing_dataframe))
#               households    median_income       total_bedrooms          latitude           longitude   housing_median_age           population   rooms_per_person           total_rooms
# 12945    169.0  -->  5.1    5.1  -->  1.8    181.0   -->   5.2   37.4  -->   0.0   -121.8  -->  -0.5      33.0  -->   0.3    781.0   -->  -0.7      1.1  --> -0.6      835.0  -->  -0.8
# 9946     242.0  -->  5.5    2.4  -->  1.2    220.0   -->   5.4   36.8  -->  -0.1   -119.8  -->   0.1      35.0  -->   0.3    474.0   -->  -0.8     2.4  -->  -0.1    1129.0  -->   -0.8
# 13963    709.0  -->  6.6    3.2  -->  1.4    733.0   -->   6.6   38.0  -->   0.2   -122.0  -->   0.5      16.0  -->  -0.4   1447.0   -->  -0.4     2.1  -->  -0.2    3077.0  -->   -0.4
# 8795     411.0  -->  6.0    6.0  -->  1.9    455.0   -->   6.1   34.2  -->  -0.7   -118.6  -->   0.1      33.0  -->   0.3   1116.0   -->  -0.6     2.6  -->   0.0    2896.0  -->   -0.4
# 11989    167.0  -->  5.1    3.8  -->  1.6    180.0   -->   5.2   38.6  -->   0.3   -121.4  -->   0.4      34.0  -->   0.3    359.0   -->  -0.9     3.4  -->   0.4    1226.0  -->   -0.8
# ...        ...  -->  ...    ...  -->  ...      ...   -->   ...    ...  -->   ...      ...  -->   ...       ...  -->   ...      ...   -->   ...     ...  -->   ...       ...  -->    ...
# 15598    443.0  -->  6.1    1.9  -->  1.1    524.0   -->   6.3   38.0  -->   0.2   -122.3  -->   0.6      29.0  -->   0.1   1357.0   -->  -0.5     1.4  -->  -0.4    1899.0  -->   -0.6
# 5680     691.0  -->  6.5    1.8  -->  1.0    758.0   -->   6.6   33.8  -->  -0.7   -118.2  -->   0.2      30.0  -->   0.1   2951.0   -->   0.2     0.9  -->  -0.6    2734.0  -->   -0.5
# 14038   1091.0  -->  7.0    2.9  -->  1.3   1213.0   -->   7.1   38.0  -->   0.1   -122.0  -->   0.5      22.0  -->  -0.2   2804.0   -->   0.1     1.8  -->  -0.3    5175.0  -->    0.0
# 5634     143.0  -->  5.0    2.9  -->  1.4    142.0   -->   5.0   34.0  -->  -0.7   -118.2  -->   0.2      35.0  -->   0.3    720.0   -->  -0.7     0.9  -->  -0.6     661.0  -->   -0.9
# 247     1105.0  -->  7.0    3.5  -->  1.5   1444.0   -->   7.3   33.9  -->  -0.7   -116.5  -->   0.6      13.0  -->  -0.5   3189.0   -->   0.3     2.4  -->  -0.1    7559.0  -->    0.5
#
# [17000 rows x 9 columns]
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

_ = train_nn_regression_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.15),
    steps=1000,
    batch_size=50,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)
# Training model...
# RMSE (on training data):
#   period 00 : 89.11
#   period 01 : 77.52
#   period 02 : 73.09
#   period 03 : 71.00
#   period 04 : 69.69
#   period 05 : 69.07
#   period 06 : 68.11
#   period 07 : 67.74
#   period 08 : 67.66
#   period 09 : 66.80
# Model training finished.
# Final RMSE (on training data):   66.80
# Final RMSE (on validation data): 68.73


#  ## 선택 과제: 위도 및 경도 특성만 사용
#
# **특성으로 위도와 경도만 사용하는 NN 모델을 학습시킵니다.**
#
# 부동산 업계에서는 오로지 입지만이 주택 가격을 결정하는 중요한 요소라고 말합니다.
# 특성으로 위도와 경도만 사용하는 모델을 학습시켜 이 주장을 검증해 보겠습니다.
#
# 이 방법이 성공하려면 NN이 위도 및 경도로부터 복잡한 비선형성을 학습할 수 있어야 합니다.
#
# **NOTE:** 실습 앞부분에서 충분했던 것보다 더 많은 레이어를 포함하는 네트워크 구조가 필요할 수 있습니다.
#
# YOUR CODE HERE: Train the network using only latitude and longitude
#
def location_location_location(examples_dataframe):
    processed_features = pd.DataFrame()
    processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
    processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
    return processed_features


lll_dataframe = location_location_location(preprocess_features(california_housing_dataframe))
lll_training_examples = lll_dataframe.head(12000)
lll_validation_examples = lll_dataframe.tail(5000)
_ = train_nn_regression_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.05),
    steps=500,
    batch_size=50,
    hidden_units=[10, 10, 5, 5, 5],  # [10, 10]
    training_examples=lll_training_examples,
    training_targets=training_targets,
    validation_examples=lll_validation_examples,
    validation_targets=validation_targets)
# Training model...
# RMSE (on training data):
#   hidden_units = [10, 10] --> [10, 10, 5, 5, 5]
#   period 00 : 224.45 --> 122.74
#   period 01 : 200.50 --> 108.51
#   period 02 : 166.24 --> 105.26
#   period 03 : 132.98 --> 102.98
#   period 04 : 111.37 --> 101.21
#   period 05 : 108.10 --> 100.68
#   period 06 : 107.17 --> 100.38
#   period 07 : 106.69 --> 100.90
#   period 08 : 106.17 --> 100.25
#   period 09 : 105.89 --> 100.18
# Model training finished.
# Final RMSE (on training data):   105.89 --> 100.18
# Final RMSE (on validation data): 105.44 --> 101.16
