"""특성 세트
학습 목표: 복잡한 특성 세트만큼 좋은 성능을 발휘하는 최소한의 특성 세트를 만듭니다.

지금까지는 모델에 모든 특성을 집어넣었습니다.
그러나 모델에 포함된 특성이 적을수록 리소스 사용이 감소하며 유지보수도 쉬워집니다.

이제부터는 주택 관련 특성을 최소한으로 사용하면서 데이터 세트의 모든 특성을
사용하는 모델과 동등한 성능을 발휘하는 모델을 만들 수 있는지를 살펴보겠습니다.
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
    # 인구 밀도 계산하기
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
    # Scaling feature values (데이터 정제: 특성 값 조정)
    output_targets["median_house_value"] = (
            california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets


# Choose the first 12000 (out of 17000) examples for training.
# 학습 세트 선택하기 (검증)
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

# Choose the last 5000 (out of 17000) examples for validation.
# 검증 세트 선택하기 (검증)
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


def work1():
    """효율적인 특성 세트 개발
    특성을 2~3개만 사용하면서 성능을 어디까지 올릴 수 있을까요?
    상관행렬은 각 특성을 타겟과 비교한 결과 및 각 특성을 서로 비교한 결과에 따라 쌍의 상관성을 보여줍니다.
    여기에서는 상관성을 피어슨 상관계수로 정의합니다. 이 실습을 위해 자세한 수학적 원리를 이해할 필요는 없습니다.

    상관성 값의 의미는 다음과 같습니다.
        - -1.0: 완벽한 음의 상관성
        - 0.0: 상관성 없음
        - 1.0: 완벽한 양의 상관성
    """
    correlation_dataframe = training_examples.copy()
    correlation_dataframe['target'] = training_targets['median_house_value']
    correlation_dataframe.corr()

    #
    # Your code here: add your features of choice as a list of quoted strings.
    #
    minimal_features = [
        "median_income",
        "latitude",
    ]

    minimal_training_examples = training_examples[minimal_features]
    minimal_validation_examples = validation_examples[minimal_features]

    #
    # Don't forget to adjust these parameters.
    #
    _ = train_model(
        learning_rate=0.001,
        steps=500,
        batch_size=5,
        training_examples=minimal_training_examples,
        training_targets=training_targets,
        validation_examples=minimal_validation_examples,
        validation_targets=validation_targets)


# 타겟과 상관성이 높은 특성을 찾아야 합니다.
# 또한 각 특성이 서로 독립적인 정보를 추가하도록 서로간의 상관성이 높지 않은 특성을 찾는 것이 좋습니다.
# 이 정보를 참고하여 특성을 삭제해 보세요.
# 두 가지 원시 특성의 비율과 같은 합성 특성을 추가로 만들어 볼 수도 있습니다.
# 편의를 위해 이전 실습의 학습 코드를 포함해 두었습니다.
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


def train_model(
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
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
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


def work2():
    """위도 활용 고도화
    latitude와 median_house_value로 그래프를 그리면 선형 관계가 없다는 점이 드러납니다.
    대신, 로스앤젤레스 및 샌프란시스코에 해당하는 위치 부근에 마루가 나타납니다.
    """
    plt.scatter(training_examples["latitude"], training_targets["median_house_value"])

    # 위도를 더 잘 활용할 수 있는 합성 특성을 만들어 보세요.
    # 예를 들어 latitude를 |latitude - 38|의 값에 매핑하는 특성을 만들고 이름을 distance_from_san_francisco로 지정할 수 있습니다.
    # 또는 공간을 10개의 버킷으로 나눌 수 있습니다.
    # latitude_32_to_33, latitude_33_to_34 등의 특성을 만들고
    # latitude가 해당 버킷의 범위에 포함되면 1.0 값을, 그렇지 않으면 0.0 값을 표시하면 됩니다.
    # 상관행렬을 개발에 참고하면서 적절한 특성이 발견되면 모델에 추가하세요.
    # 검증 성능을 최대 어느 정도까지 높일 수 있나요?
    selected_training_examples = select_and_transform_features(training_examples)
    selected_validation_examples = select_and_transform_features(validation_examples)

    _ = train_model(
        learning_rate=0.01,
        steps=500,
        batch_size=5,
        training_examples=selected_training_examples,
        training_targets=training_targets,
        validation_examples=selected_validation_examples,
        validation_targets=validation_targets)


LATITUDE_RANGES = zip(range(32, 44), range(33, 45))


def select_and_transform_features(source_df):
    selected_examples = pd.DataFrame()
    selected_examples["median_income"] = source_df["median_income"]
    for r in LATITUDE_RANGES:
        selected_examples["latitude_%d_to_%d" % r] = source_df["latitude"].apply(
            lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
    return selected_examples
