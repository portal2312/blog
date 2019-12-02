import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
print('데이터 가져오기\n', california_housing_dataframe)
plt.close()

calibration_data = None


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    print(f'    options: batch_size = {batch_size}, shuffle = {shuffle}, num_epochs = {num_epochs}')

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}
    # {'room_per_person': array([1.8100172, 2.00326, 2.4851851, ..., 1.0234493, 2.0966184, 2.052356], dtype=float32)}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    # <DatasetV1Adapter shapes: ((1,), ()), types: (tf.float32, tf.float32)>
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size, input_feature):
    """Trains a linear regression model.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      input_feature: A `string` specifying a column from `california_housing_dataframe`
        to use as input feature.

    Returns:
      A Pandas `DataFrame` containing targets and the corresponding predictions done
      after training the model.
    """

    periods = 10
    steps_per_period = steps / periods

    print('1. 특성 정의 및 특성 열 구성')
    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]].astype('float32')
    feature_columns = [tf.feature_column.numeric_column(my_feature)]
    print(f"""
    특성: {my_feature}
    특성 값\n{my_feature_data}
    특성 열 구성 = {feature_columns}
    """)

    print('2. 타겟 정의')
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label].astype('float32')
    print(f"""
    타겟 = {my_label}
    타겟 값\n{targets}
    """)

    # Create a linear regressor object.
    print('3. LinearRegressor 구성')
    learning_max_rate = 5.0
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, learning_max_rate)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)
    print(f"""
    학습 경사: 단계 크기 = {learning_rate}
    단계 크기 제한: {learning_max_rate}
    """)

    # Create input functions.
    print('4. 입력 함수 정의\n')
    print('4.1. 학습 입력 함수 정의\n')
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    print('4.2. 예측 입력 함수 정의\n')
    num_epochs = 1
    predict_training_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=num_epochs, shuffle=False)

    # Set up to plot the state of our model's line each period.
    plt.figure(figsize=(15, 6))  # 그래프 (가로, 세로) 크기
    nrows = 1
    ncols = 2
    plt.subplot(nrows, ncols, 1)  # nrows, ncols, index
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        print(f'    5.{period}. 모델 학습 ({steps_per_period * (period + 1)}/{steps})')
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )

        # Take a break and compute predictions.
        print(f'    6.{period}. 모델 평가 ({steps_per_period * (period + 1)}/{steps})')
        predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # Compute loss.
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        # Occasionally print the current loss.
        print("    period %02d : %0.2f" % (period, root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)
        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.

        # Y 의 default 최대 & 최소 범위 구하기
        y_extents = np.array([0, sample[my_label].max()])
        # array([  0.   , 500.001])

        # 계산된 실제 가중치(weight, W1, W2, .. Wn)
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        # array([0.02228459], dtype=float32)

        # 계산된 실제 편향(bias, b = W0)
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
        # array([0.01119963], dtype=float32)

        # X 값 구하기
        #   1. Y 의 default 최대 & 최소에 각각 - 편향 / 가중치
        x_extents = (y_extents - bias) / weight
        # array([-0.5, 22436]) = array([-1.11996271e-02, 4.99989800e+02]) / array([0.02228459], dtype=float32)

        #   2. 계산된 범위를 sample[my_feature] 최대 & 최소 범위에서 맞추기
        #   np.minimum(np.maximum(x_extents, sample[my_feature].min()), sample[my_feature].max()) 이와 같음
        x_extents = np.maximum(
            np.minimum(x_extents, sample[my_feature].max()),
            # array([-0.50257265,  6.634375]) = np.minimun(array([-5.02572645e-01,  2.24365682e+04]), 6.634375)
            sample[my_feature].min()
            # 0.06160540478644223
        )
        # array([0.0616054, 6.634375 ])

        # Y 값 구하기
        y_extents = weight * x_extents + bias
        # chart 그리기, [(X0, Y0), (X1, Y1)]
        plt.plot(x_extents, y_extents, color=colors[period])
        # (array([0.0616054, 6.634375 ]), array([0.01257248, 0.15904398]), (0.2298057, 0.298717966, 0.753683153, 1.0))
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.subplot(nrows, ncols, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    # 지정된 패딩을 제공하기 위해 서브 플로트 매개 변수를 자동으로 조정합니다.
    plt.tight_layout()
    # chart 그리기, [RMSE0, RMSE1, .. RMSE9], period 별 평균 제곱근 오차(RMSE, Root Mean Squared Error)
    plt.plot(root_mean_squared_errors)

    # Create a table with calibration data.
    # (경사하강법에 의해 마지막 값이 정확하다는 가정하에) 마지막 예측 값들만 분석 정보 보여주기
    global calibration_data
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)


def work1():
    """인구 밀도와 주택 가격 중앙 값의 관계
    total_rooms 특성과 population 특성은 모두 특정 지역의 합계를 계수합니다.

    그런데 지역마다 인구밀도가 다르다면 어떻게 될까요?
    total_rooms와 population의 비율로 합성 특성을 만들면 지역의 인구밀도와 주택 가격 중앙값의 관계를 살펴볼 수 있습니다.

    아래 셀에서 rooms_per_person이라는 특성을 만들고 train_model()의 input_feature로 사용합니다.

    학습률을 조정하여 이 단일 특성으로 성능을 어디까지 올릴 수 있을까요?
    성능이 높다는 것은 회귀선이 데이터에 잘 부합하고 최종 RMSE가 낮다는 의미입니다.
    """
    global california_housing_dataframe

    california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index))
    print('데이터 index 열을 random 하게 배열후 그에 따른 데이터도 재인덱스 하기\n', california_housing_dataframe)

    california_housing_dataframe["median_house_value"] /= 1000.0
    print('집 값(median_house_value) 자리수 1,000 단위로 변경하기\n', california_housing_dataframe)

    input_feature = 'rooms_per_person'
    # 인구 밀도 = 방 총수/ 인구
    california_housing_dataframe[input_feature] = (
        california_housing_dataframe['total_rooms'] / california_housing_dataframe['population']
    )
    print('인구 밀도(rooms_per_person, =방 총수/인구) 추가하기\n', california_housing_dataframe)

    train_model(
        learning_rate=0.00005,
        steps=500,
        batch_size=5,
        input_feature="rooms_per_person"
    )


def work2():
    """이상점 식별
    예측과 목표값을 비교한 산포도를 작성하면 모델의 성능을 시각화할 수 있습니다.
    이상적인 상태는 완벽한 상관성을 갖는 대각선이 그려지는 것입니다.
    작업 1에서 학습한 rooms-per-person 모델을 사용한 예측과 타겟에 대해 Pyplot의 scatter()로 산포도를 작성합니다.
    특이한 점이 눈에 띄나요? rooms_per_person의 값 분포를 조사하여 소스 데이터를 추적해 보세요.
    """
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    # 보정 데이터를 보면 대부분의 산포점이 직선을 이룹니다. 이 선은 수직에 가까운데, 여기에 대해서는 나중에 설명합니다.
    # 지금은 선에서 벗어난 점에 대해 집중할 때입니다. 이러한 점은 비교적 적은 편입니다.
    # rooms_per_person의 히스토그램을 그려보면 입력 데이터에서 몇 개의 이상점을 발견할 수 있습니다.
    plt.scatter(calibration_data["predictions"], calibration_data["targets"])
    plt.show()


def work3():
    """이상점 삭제
    rooms_per_person의 이상점 값을 적당한 최소값 또는 최대값으로 설정하여 모델의 적합성을 더 높일 수 있는지 살펴보세요.
    다음은 Pandas Series에 함수를 적용하는 방법을 간단히 보여주는 예제입니다.
    clipped_feature = my_dataframe["my_feature_name"].apply(lambda x: max(x, 0))
    위와 같은 clipped_feature는 0 미만의 값을 포함하지 않습니다.
    """

    # 작업 2에서 작성한 히스토그램을 보면 대부분의 값이 5 미만입니다.
    # rooms_per_person을 5에서 잘라내고 히스토그램을 작성하여 결과를 다시 확인해 보세요.
    california_housing_dataframe["rooms_per_person"] = (
        california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))

    _ = california_housing_dataframe["rooms_per_person"].hist()

    train_model(
        learning_rate=0.05,
        steps=500,
        batch_size=5,
        input_feature="rooms_per_person")

    _ = plt.scatter(calibration_data["predictions"], calibration_data["targets"])
    plt.show()
