"""신경망으로 필기 입력된 숫자 분류하기
학습 목표:
- 선형 모델과 신경망을 모두 학습시켜 기존 MNIST 데이터 세트의 필기 입력된 숫자를 분류한다
- 선형 모델과 신경망 분류 모델의 성능을 비교한다
- 신경망 히든 레이어의 가중치를 시각화한다

이번 목표는 각각의 입력 이미지를 올바른 숫자에 매핑하는 것입니다.
몇 개의 히든 레이어를 포함하며 소프트맥스 레이어가 맨 위에서 최우수 클래스를 선택하는 NN을 만들어 보겠습니다.
"""
# ## 설정
# 우선 데이터 세트를 다운로드하고, 텐서플로우 및 기타 유틸리티 모듈을 import로 불러오고, 데이터를 pandas DataFrame에 로드합니다.
# 이 데이터는 원본 MNIST 학습 데이터에서 20,000개 행을 무작위로 추출한 샘플입니다.
from __future__ import print_function

import glob
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

mnist_dataframe = pd.read_csv(
  "https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv",
  sep=",",
  header=None)  # [20000 rows x 785 columns]

# Use just the first 10,000 records for training/validation.
mnist_dataframe = mnist_dataframe.head(10000)

mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
mnist_dataframe.head()
#       0    1    2    3    4    5    6    7    8    9    10   11   12   ...  772  773  774  775  776  777  778  779  780  781  782  783  784
# 6974    3    0    0    0    0    0    0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0    0    0    0    0    0    0
# 1810    6    0    0    0    0    0    0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0    0    0    0    0    0    0
# 7333    3    0    0    0    0    0    0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0    0    0    0    0    0    0
# 3022    4    0    0    0    0    0    0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0    0    0    0    0    0    0
# 423     3    0    0    0    0    0    0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0    0    0    0    0    0    0
#
# [5 rows x 785 columns]


# 첫 번째 열은 클래스 라벨을 포함합니다. (숫자 그림을 의미, target)
# 나머지 열은 특성 값을 포함하며, 28×28=784개 픽셀 값마다 각각 하나의 특성 값이 됩니다.
# 이 784개의 픽셀 값은 대부분 0이지만, 1분 정도 시간을 들여 모두 0은 아니라는 것을 확인하시기 바랍니다.
#
# 이러한 예는 비교적 해상도가 낮고 대비가 높은 필기 입력 숫자입니다.
# 0-9 범위의 숫자 10개가 각각 표현되었으며 가능한 각 숫자에 고유한 클라스 라벨이 지정됩니다.
# 따라서 이 문제는 10개 클래스를 대상으로 하는 다중 클래스 분류 문제입니다.
#
# 이제 라벨과 특성을 해석하고 몇 가지 예를 살펴보겠습니다.
# 이 데이터 세트에는 헤더 행이 없지만 loc 를 사용하여 원래 위치를 기준으로 열을 추출할 수 있습니다.
def parse_labels_and_features(dataset):
    """Extracts labels and features.

    This is a good place to scale or transform the features if needed.

    Args:
      dataset: A Pandas `Dataframe`, containing the label on the first column and
        monochrome pixel values on the remaining columns, in row major order.
    Returns:
      A `tuple` `(labels, features)`:
        labels: A Pandas `Series`.
        features: A Pandas `DataFrame`.
    """
    labels = dataset[0]

    # DataFrame.loc index ranges are inclusive at both ends.
    features = dataset.loc[:, 1:784]  # [x, y] = :, 1:784
    # Scale the data to [0, 1] by dividing out the max value, 255.
    # 최대 값 인 255를 나눔으로써 데이터를 [0, 1]로 스케일합니다.
    features = features / 255

    return labels, features


training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])  # training
# print(training_targets)
# 6974    3
# 1810    6
# 7333    3
# 3022    4
# 423     3
#        ..
# 7910    0
# 9447    9
# 2671    3
# 9445    1
# 9569    6
# Name: 0, Length: 7500, dtype: int64
training_examples.describe()
#          1      2      3      4      5      6      7      8      9    ...    776    777    778    779    780    781    782    783    784
# count 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0  ... 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0
# mean     0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# std      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# min      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# 25%      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# 50%      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# 75%      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# max      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    1.0    0.3    0.2    1.0    0.2    0.0    0.0    0.0    0.0
#
# [8 rows x 784 columns]

validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])  # validation
validation_examples.describe()
#          1      2      3      4      5      6      7      8      9    ...    776    777    778    779    780    781    782    783    784
# count 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0  ... 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0 7500.0
# mean     0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# std      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# min      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# 25%      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# 50%      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# 75%      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
# max      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ...    1.0    0.3    0.2    1.0    0.2    0.0    0.0    0.0    0.0
#
# [8 rows x 784 columns]


#  무작위로 선택한 예 및 해당 라벨을 표시합니다.
rand_example = np.random.choice(training_examples.index)
_, ax = plt.subplots()

# matplotlib.pyplot.matshow: 새 그림 창에 배열을 매트릭스로 표시합니다.
# numpy.reshape: 무작위로 선택한 예 및 해당 라벨을 표시합니다.
# 데이터를 변경하지 않고 배열에 새로운 모양을 지정합니다.
# 28 = math.sqrt(len(training_examples.columns))
# numpy.ndarray(28x28) 생성됨
ax.matshow(training_examples.loc[rand_example].values.reshape((28, 28)))

ax.set_title("Label: %i" % training_targets.loc[rand_example])
print("Label: %i" % training_targets.loc[rand_example])
ax.grid(False)
plt.show()


# 작업 1: MNIST 용 선형 모델 구축
# 우선 비교 기준이 될 모델을 만듭니다.
# `LinearClassifier`는 k개 클래스마다 하나씩 k개의 일대다 분류자 집합을 제공합니다.
#
# 이 작업에서는 정확성을 보고하고 시간별 로그 손실을 도식화할 뿐 아니라 혼동행렬도 표시합니다.
# 혼동행렬은 다른 클래스로 잘못 분류된 클래스를 보여줍니다.
# 참조: [혼동행렬](https://en.wikipedia.org/wiki/Confusion_matrix)

# 서로 혼동하기 쉬운 숫자는 무엇일까요?
#
# 또한 `log_loss` 함수를 사용하여 모델의 오차를 추적합니다.
# 이 함수는 학습에 사용되는 `LinearClassifier` 내장 손실 함수와 다르므로 주의하시기 바랍니다.
def construct_feature_columns():
    """Construct the TensorFlow Feature Columns.

    Returns:
      A set of feature columns
    """

    # There are 784 pixels in each image.
    return set([tf.feature_column.numeric_column('pixels', shape=784)])


# 여기에서는 학습과 예측의 입력 함수를 서로 다르게 만들겠습니다.
# 각각 create_training_input_fn() 및 create_predict_input_fn()에 중첩시키고
# 이러한 함수를 호출할 때 반환되는 해당 _input_fn을 .train() 및 .predict() 호출에 전달하면 됩니다.
#
# 함수 비교:
# create_training_input_fn(): features index 기준으로 를 features, targets 를 재 indexing.
# create_predict_input_fn(): num_epochs, shuffle 없음
def create_training_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True):
    """A custom input_fn for sending MNIST data to the estimator for training.

    Args:
      features: The training features.
      labels: The training labels.
      batch_size: Batch size to use during training.

    Returns:
      A function that returns batches of training features and labels during
      training.
    """

    def _input_fn(num_epochs=None, shuffle=True):
        # Input pipelines are reset with each call to .train(). To ensure model
        # gets a good sampling of data, even when number of steps is small, we
        # shuffle all the data before creating the Dataset object

        # np.random.permutation: 순서를 무작위로 바꾸거나 치환 된 범위를 반환
        idx = np.random.permutation(features.index)
        # array([2420, 6451, 1154, ..., 2732, 3015, 8396])

        # 임의 index(=idx) 기준으로 features, targets 재가공하기
        raw_features = {"pixels": features.reindex(idx)}
        # {'pixels':       1    2    3    4    5    6    7    8    9    10   11   12   13   ...  772  773  774  775  776  777  778  779  780  781  782  783  784
        #  2420  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.4  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        #  6451  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        #  1154  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        #  7499  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        #  468   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        #  ...   ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...
        #  3500  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        #  3589  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        #  2732  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        #  3015  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        #  8396  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        #
        #  [7500 rows x 784 columns]}
        raw_targets = np.array(labels[idx])
        # array([7, 0, 6, ..., 7, 9, 2])

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))  # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        # Return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch
    # 함수를 반환
    return _input_fn


def create_predict_input_fn(features, labels, batch_size):
    """A custom input_fn for sending mnist data to the estimator for predictions.

    Args:
      features: The features to base predictions on.
      labels: The labels of the prediction examples.

    Returns:
      A function that returns features and labels for predictions.
    """

    def _input_fn():
        raw_features = {"pixels": features.values}
        # {'pixels': array([[0., 0., 0., ..., 0., 0., 0.],
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         ...,
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         [0., 0., 0., ..., 0., 0., 0.]])}
        raw_targets = np.array(labels)
        # array([5, 9, 0, ..., 2, 6, 7])

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))  # warning: 2GB limit
        ds = ds.batch(batch_size)

        # Return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch
    # 함수를 반환
    return _input_fn


def train_linear_classification_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear classification model for the MNIST digits dataset.

    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, and a confusion
    matrix.

    Args:
      learning_rate: A `float`, the learning rate to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing the training features.
      training_targets: A `DataFrame` containing the training labels.
      validation_examples: A `DataFrame` containing the validation features.
      validation_targets: A `DataFrame` containing the validation labels.

    Returns:
      The trained `LinearClassifier` object.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a LinearClassifier object.
    # `Adagrad`: 각 매개변수의 경사를 재조정하여 사실상 각 매개변수에 독립적인 학습률을 부여하는 정교한 경사하강법 알고리즘.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(),
        # {NumericColumn(key='pixels', shape=(784,), default_value=None, dtype=tf.float32, normalizer_fn=None)}

        n_classes=10,
        # tensorflow.python.estimator.canned.linear.LinearClassifier: 선형 분류
        #
        # Args:
        # - n_classes: 레이블 클래스의 수, 기본값은 2
        #   클래스 레이블은 클래스 색인을 나타내는 정수(예: 0 에서 n_classes-1까지의 값)입니다.
        #   임의의 라벨 값 (예: 문자열 라벨)의 경우 먼저 클래스 색인으로 변환하십시오.

        optimizer=my_optimizer,

        config=tf.estimator.RunConfig(keep_checkpoint_max=1)
        # tensorflow.estimator.RunConfig: 'Estimator' 실행을 위한 구성을 지정합니다.
        #
        # Args:
        # - keep_checkpoint_max: 보관할 최신 검사 점 파일의 최대 개수입니다.
        #   새 파일이 만들어지면 이전 파일이 삭제됩니다.
        #   없음 또는 0 이면 모든 검사 점 파일이 보관됩니다.
        #   기본값은 5입니다 (즉, 가장 최근의 검사 점 파일 5 개가 유지됩니다).
    )

    # Create the input functions.
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        # print(training_predictions)
        # [{'logits': array([ 0.2673661 , -4.4253078 , -2.373763  ,  6.2576575 , -2.192929  ,
        #          0.03950174, -4.099607  , -3.4516742 , -0.9503593 , -1.958474  ],
        #        dtype=float32),
        #  'probabilities': array([2.4879707e-03, 2.2795179e-05, 1.7734450e-04, 9.9402171e-01,
        #         2.1249703e-04, 1.9810030e-03, 3.1571344e-05, 6.0351318e-05,
        #         7.3619676e-04, 2.6864294e-04], dtype=float32),
        #  'class_ids': array([3]),
        #  'classes': array([b'3'], dtype=object)}, ...]
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        # array([[2.48797075e-03, 2.27951787e-05, 1.77344496e-04, ...,
        #         6.03513181e-05, 7.36196758e-04, 2.68642936e-04],
        #        [8.80458276e-04, 9.47245164e-04, 1.01126824e-02, ...,
        #         2.39408138e-04, 2.11432464e-02, 4.38613258e-03],
        #        [2.48036231e-03, 1.19357568e-03, 3.77565116e-01, ...,
        #         2.34441031e-04, 2.83443257e-02, 3.92415514e-03],
        #        ...,
        #        [9.84722399e-04, 1.46163657e-04, 1.15041055e-01, ...,
        #         1.29549980e-01, 4.07314450e-02, 4.62269068e-01],
        #        [1.78463652e-05, 9.86286938e-01, 7.34999834e-04, ...,
        #         4.55965230e-04, 4.14442131e-03, 1.14288589e-03],
        #        [2.36431453e-02, 2.26535322e-03, 1.55863822e-01, ...,
        #         1.49819616e-03, 8.31196085e-02, 7.30846263e-03]], dtype=float32)
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        # array([3, 6, 3, ..., 9, 1, 6])
        # tf.keras.utils.to_categorical: 클래스 벡터(정수)를 이진 클래스 행렬로 변환합니다.
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)
        # array([[0., 0., 0., ..., 0., 0., 0.],
        #        [0., 0., 0., ..., 0., 0., 0.],
        #        [0., 0., 0., ..., 0., 0., 0.],
        #        ...,
        #        [0., 0., 0., ..., 0., 0., 1.],
        #        [0., 1., 0., ..., 0., 0., 0.],
        #        [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)
        # 예:
        # training_pred_class_id[0] = 3 경우
        # training_pred_one_hot[0] = array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], dtype=float32)
        # index=3 만 1, 나머지 0.
        # training_pred_class_id[1] = 6 경우
        # training_pred_one_hot[1] 의 index=6 만 1, 나머지 0.
        #   array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], dtype=float32)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier


# ### 해결 방법
#
# 다음은 약 0.9의 정확성을 달성하는 매개변수 세트입니다.
_ = train_linear_classification_model(
    learning_rate=0.02,  # 0.03
    steps=100,  # 1000
    batch_size=10,  # 30
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
# Training model...
# LogLoss error (on validation data):
#   period 00 : 18.32 -->  4.48
#   period 01 : 9.96  -->  3.81
#   period 02 : 9.06  -->  3.58
#   period 03 : 6.95  -->  3.69
#   period 04 : 8.41  -->  3.63
#   period 05 : 7.16  -->  3.41
#   period 06 : 6.33  -->  3.41
#   period 07 : 5.53  -->  3.36
#   period 08 : 5.62  -->  3.37
#   period 09 : 5.60  -->  3.38
# Model training finished.
# Final accuracy (on validation data): 0.84 --> 0.90
