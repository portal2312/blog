"""검증
- 단일 특성이 아닌 여러 특성을 사용하여 모델의 효과를 더욱 높인다
- 모델 입력 데이터의 문제를 디버깅한다
- 테스트 데이터 세트를 사용하여 모델이 검증 데이터에 과적합되었는지 확인한다
https://colab.research.google.com/notebooks/mlcc/validation.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=validation-colab&hl=ko#scrollTo=65sin-E5NmHN
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

# california_housing_dataframe = california_housing_dataframe.reindex(
#     np.random.permutation(california_housing_dataframe.index))

training_examples = training_targets = validation_examples = validation_targets = linear_regressor = None


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
    # Scale the target to be in units of thousands of dollars.
    output_targets["median_house_value"] = (
            california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.

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


# 이제 여러 입력 특성을 다뤄야 하므로 특성 열을 구성하는 코드를 모듈화하여 별도의 함수로 만들겠습니다.
# 지금은 모든 특성이 숫자이므로 코드가 비교적 단순하지만, 이후 실습에서 다른 유형의 특성을 사용하면서 이 코드를 확장해 나갈 것입니다.)
def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
      input_features: The names of the numerical input features to use.
    Returns:
      A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model of multiple features.

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

    # 1. Create input functions.
    my_label = 'median_house_value'
    training_input_fn = lambda: my_input_fn(
        training_examples, training_targets[my_label],
        batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(
        training_examples, training_targets[my_label],
        num_epochs=1,
        shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(
        validation_examples, validation_targets[my_label],
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
            steps=steps_per_period,
        )
        # 2. Take a break and compute predictions.
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

    # 값 확인하기
    df = pd.DataFrame()
    df['training_rmse'] = pd.Series(training_rmse)
    df['validation_rms'] = pd.Series(validation_rmse)
    print(df)
    print(df.descibe())

    return linear_regressor


def work1():
    global training_examples, training_targets, validation_examples, validation_targets

    training_examples = preprocess_features(california_housing_dataframe.head(12000))
    training_examples.describe()
    training_targets = preprocess_targets(california_housing_dataframe.head(12000))
    training_targets.describe()

    validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
    validation_examples.describe()
    validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
    validation_targets.describe()


def work2():
    global training_examples, training_targets, validation_examples, validation_targets

    plt.figure(figsize=(13, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Validation Data")

    ax.set_autoscaley_on(False)  # 플롯 명령에 Y 축의 자동 확장을 적용할지 여부를 설정합니다.
    ax.set_ylim([32, 43])  # y 축보기 제한을 설정합니다.
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(validation_examples["longitude"],
                validation_examples["latitude"],
                cmap="coolwarm",  # color map
                c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())  # color

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Training Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(training_examples["longitude"],
                training_examples["latitude"],
                cmap="coolwarm",
                c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
    # Plot y versus x as lines and/or markers.
    # 선 및 / 또는 표식으로 y 대 x를 플롯합니다.
    _ = plt.plot()


def work3():
    """데이터 가져오기 및 전처리 코드로 돌아가서 버그가 있는지 확인
    문제점을 찾아서 수정했으면 위의 latitude / longitude 도식화 셀을 다시 실행하여 타당성 검사 결과가 좋아졌는지 확인합니다.
    여기에서 중요한 교훈을 얻을 수 있습니다.
    ML의 디버깅은 코드 디버깅이 아닌 데이터 디버깅인 경우가 많습니다.
    데이터가 잘못되었다면 가장 발전한 최신 ML 코드라도 문제를 일으킬 수밖에 없습니다.
    """
    features = ['latitude', 'longitude']
    print(validation_examples[features])
    print(training_examples[features])
    # 데이터를 읽을 때 무작위로 섞는 부분을 살펴봅니다.
    # 학습 세트와 검증 세트를 만들 때 데이터를 무작위로 적절히 섞지 않으면, 데이터가 일정한 규칙으로 정렬된 경우 문제가 생길 수 있습니다.
    # 바로 이 문제가 발생한 것으로 생각됩니다.
    # 위도 경도가 각 head, tail 에서 가져온 거라 서로 일반적이지 않다.


def work4():
    """모델 학습 및 평가
    다음으로, 데이터 세트의 모든 특성을 사용하여 선형 회귀 모델을 학습시키고 성능을 확인합니다.
    앞에서 텐서플로우 모델에 데이터를 로드할 때 사용한 것과 동일한 입력 함수(my_input_fn, construct_feature_columns)를 정의.
    """
    global training_examples, training_targets, validation_examples, validation_targets, linear_regressor
    # 다음으로, 아래의 train_model() 코드를 완성하여 입력 함수를 설정하고 예측을 계산합니다.
    #
    # 참고: 이전 실습의 코드를 참조해도 무방하지만 적절한 데이터 세트에 대해 predict()를 호출해야 합니다.
    # 학습 데이터와 검증 데이터의 손실을 비교합니다. 원시 특성이 하나일 때는 가장 양호한 평균 제곱근 오차(RMSE)가 약 180이었습니다.
    # 앞에서 살펴본 몇 가지 방법으로 데이터를 점검하세요. 예를 들면 다음과 같습니다.
    # - 예측 값과 실제 타겟 값의 분포 비교
    # - 예측 값과 타겟 값으로 산포도 작성
    # - latitude 및 longitude를 사용하여 검증 데이터로 두 개의 산포도 작성
    #   - 색상을 실제 타겟 median_house_value에 매핑하는 도식 작성
    #   - 색상을 예측된 median_house_value에 매핑하는 도식을 작성하여 나란히 비교

    linear_regressor = train_model(
        learning_rate=0.00003,
        steps=500,
        batch_size=5,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

    # Training model...
    # RMSE (on training data):
    #   period 00 : 207.40
    #   period 01 : 189.95
    #   period 02 : 176.86
    #   period 03 : 171.05
    #   period 04 : 164.73
    #   period 05 : 161.79
    #   period 06 : 160.91
    #   period 07 : 161.09
    #   period 08 : 161.10
    #   period 09 : 162.00
    # Model training finished.


def work5():
    """테스트 데이터로 평가
    아래 셀에서 테스트 데이터 세트를 로드하고 이를 기준으로 모델을 평가합니다.
    검증 데이터에 대해 많은 반복을 수행했습니다. 이제 해당 표본의 특이성에 대한 과적합이 발생하지 않았는지 확인해야 합니다.
    테스트 세트는 여기에 있습니다.
    검증 성능과 비교하여 테스트 성능이 어떠한가요? 이 결과가 모델의 일반화 성능에 대해 무엇을 시사하나요?
    """

    california_housing_test_data = pd.read_csv(
        "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")

    test_examples = preprocess_features(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)

    predict_test_input_fn = lambda: my_input_fn(
        test_examples,
        test_targets["median_house_value"],
        num_epochs=1,
        shuffle=False)

    global linear_regressor
    test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
    test_predictions = np.array([item['predictions'][0] for item in test_predictions])

    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets))

    print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)
    # Final RMSE (on test data): 160.67
