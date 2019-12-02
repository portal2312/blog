"""신경망 소개

- 텐서플로우의 DNNRegressor 클래스를 사용하여 신경망(NN) 및 히든 레이어를 정의한다
- 비선형성을 갖는 데이터 세트를 신경망에 학습시켜 선형 회귀 모델보다 우수한 성능을 달성한다

이전 실습에서는 모델에 비선형성을 통합하는 데 도움이 되는 합성 특성을 사용했습니다.

비선형성을 갖는 대표적인 세트는 위도와 경도였지만 다른 특성도 있을 수 있습니다.

일단 이전 실습의 로지스틱 회귀 작업이 아닌 표준 회귀 작업으로 돌아가겠습니다.
즉, median_house_value 를 직접 예측할 것입니다.
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


# 신경망 구축
# NN은 DNNRegressor 클래스에 의해 정의됩니다.
#
# **hidden_units** 를 사용하여 NN의 구조를 정의합니다.
# hidden_units 인수는 정수의 목록을 제공하며, 각 정수는 히든 레이어에 해당하고 포함된 노드의 수를 나타냅니다.
# 예를 들어 아래 대입식을 살펴보세요:
# hidden_units=[3, 10]
#
# 위 대입식은 히든 레이어 2개를 갖는 신경망을 지정합니다.
# - 1번 히든 레이어는 노드 3개를 포함합니다.
# - 2번 히든 레이어는 노드 10개를 포함합니다.
#
# 레이어를 늘리려면 목록에 정수를 더 추가하면 됩니다.
# 예:
# hidden_units=[10, 20, 30, 40]은 각각 10개, 20개, 30개, 40개의 유닛을 갖는 4개의 레이어를 만듭니다.
#
# 기본적으로 모든 히든 레이어는 ReLu 활성화를 사용하며 완전 연결성을 갖습니다.

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.
  Args:
      input_features: The names of the numerical input features to use.
  Returns:
      A set of feature columns
  Examples:
      construct_feature_columns(training_examples)

      {NumericColumn(key='households', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
      NumericColumn(key='housing_median_age', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
      NumericColumn(key='latitude', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
      NumericColumn(key='longitude', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
      NumericColumn(key='median_income', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
      NumericColumn(key='population', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
      NumericColumn(key='rooms_per_person', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
      NumericColumn(key='total_bedrooms', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
      NumericColumn(key='total_rooms', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)}
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural net regression model.

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
        learning_rate,
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
      learning_rate: A `float`, the learning rate.
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
      A `DNNRegressor` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # 경사하강법, 학습율 정의하기
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)  # 최대값 정의하기
    dnn_regressor = tf.estimator.DNNRegressor(  # (new) Deep Neural Network 선위 정의하기
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,  # (new) Hidden Layers 정의하기
        optimizer=my_optimizer
    )

    # 트레이닝에 사용할 input 함수 정의하기
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["median_house_value"],
                                            batch_size=batch_size)
    # 트레이닝 예측에 사용할 input 함수 정의하기
    # - num_epochs: 전체 데이터 세트의 각 예를 한 번씩 확인한 전체 학습 단계
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value"],
                                                    num_epochs=1,  # 이미 학습된 평가자의 예측이므로 1.
                                                    shuffle=False)  # 이미 학습된 평가자의 예측이므로 섞을 필요가 없다.
    # 예측 검증에 사용할 input 함수 정의하기
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # 모델을 훈련 시키되 루프 내부에서 수행하여 주기적으로 평가할 수 있습니다.
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # 이전 상태에서 시작하여 모델을 교육하십시오.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        training_root_mean_squared_error = math.sqrt(  # 학습 평균 제곱근 오차 구하기
            metrics.mean_squared_error(training_predictions, training_targets))  # 평균 제곱 오차 구하기
        validation_root_mean_squared_error = math.sqrt(  # 검증 평균 제곱근 오차 구하기
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

    return dnn_regressor


# ## 작업 1: NN 모델 학습
#
# **RMSE를 110 미만으로 낮추는 것을 목표로 초매개변수를 조정합니다.**
#
# 다음 블록을 실행하여 NN 모델을 학습시킵니다.
#
# 많은 특성을 사용한 선형 회귀 실습에서 RMSE 이 110 정도면 상당히 양호하다고 설명한 바 있습니다.
# 더 우수한 모델을 목표로 해 보겠습니다.
#
# 이번에 수행할 작업은 다양한 학습 설정을 수정하여 검증 데이터에 대한 정확성을 높이는 것입니다.
#
# NN 에는 과적합이라는 위험이 도사리고 있습니다.
# 학습 데이터에 대한 손실과 검증 데이터에 대한 손실의 격차를 조사하면 모델에서 과적합이 시작되고 있는지를 판단하는 데 도움이 됩니다.
# 일반적으로 격차가 증가하기 시작하면 과적합의 확실한 증거가 됩니다.
#
# 매우 다양한 설정이 가능하므로, 각 시도에서 설정을 잘 기록하여 개발 방향을 잡는 데 참고하는 것이 좋습니다.
#
# 또한 괜찮은 설정을 발견했다면 여러 번 실행하여 결과의 재현성을 확인하시기 바랍니다.
# NN 가중치는 일반적으로 작은 무작위 값으로 초기화되므로 실행 시마다 약간의 차이를 보입니다.
dnn_regressor = train_nn_regression_model(
    learning_rate=0.01,
    steps=500,
    batch_size=10,
    hidden_units=[10, 2],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
# Training model...
# RMSE (on training data):
#   period 00 : 235.02
#   period 01 : 171.61
#   period 02 : 169.36
#   period 03 : 165.08
#   period 04 : 195.91
#   period 05 : 164.29
#   period 06 : 160.57
#   period 07 : 186.12
#   period 08 : 162.80
#   period 09 : 189.05
# Model training finished.
# Final RMSE (on training data):   189.05
# Final RMSE (on validation data): 189.85


def work1():
    """해결 방법
    참고: 이 매개변수 선택은 어느 정도 임의적인 것입니다.
    여기에서는 오차가 목표치 아래로 떨어질 때까지 점점 복잡한 조합을 시도하면서 학습 시간을 늘렸습니다.
    이 조합은 결코 최선의 조합이 아니며, 다른 조합이 더 낮은 RMSE를 달성할 수도 있습니다.
    오차를 최소화하는 모델을 찾는 것이 목표라면 매개변수 검색과 같은 보다 엄밀한 절차를 사용해야 합니다.
    :return:
    """
    dnn_regressor = train_nn_regression_model(
        learning_rate=0.001,
        steps=2000,
        batch_size=100,
        hidden_units=[10, 10],
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)
    # Training model...
    # RMSE (on training data):
    #   period 00 : 172.20
    #   period 01 : 163.67
    #   period 02 : 158.25
    #   period 03 : 154.34
    #   period 04 : 148.07
    #   period 05 : 140.04
    #   period 06 : 136.66
    #   period 07 : 126.15
    #   period 08 : 112.86
    #   period 09 : 113.61
    # Model training finished.
    # Final RMSE (on training data):   113.61
    # Final RMSE (on validation data): 114.26


# 작업 2: 테스트 데이터로 평가
# 검증 성능 결과가 테스트 데이터에 대해서도 유지되는지 확인합니다.
#
# 만족할 만한 모델이 만들어졌으면 테스트 데이터로 평가하고 검증 성능과 비교해 봅니다.
#
# 테스트 데이터 세트는 여기에 있습니다.
def work2():
    """해결 방법
    위 코드에서 수행하는 작업과 마찬가지로 적절한 데이터 파일을 로드하고 전처리한 후 predict 및 mean_squared_error 를 호출해야 합니다.
    모든 레코드를 사용할 것이므로 테스트 데이터를 무작위로 추출할 필요는 없습니다.
    """
    california_housing_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")

    test_examples = preprocess_features(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)

    predict_testing_input_fn = lambda: my_input_fn(test_examples,
                                                   test_targets["median_house_value"],
                                                   num_epochs=1,
                                                   shuffle=False)

    test_predictions = dnn_regressor.predict(input_fn=predict_testing_input_fn)
    test_predictions = np.array([item['predictions'][0] for item in test_predictions])

    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets))

    print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)
    # Final RMSE(on test data): 114.64

