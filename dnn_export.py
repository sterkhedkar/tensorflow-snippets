import tensorflow as tf
import numpy as np


# this code is self sufficient as I have included some dummy data and the classifier in the same file.
# You may want to replace the train X, Y with your own datasets.


TrainSetX = [
    [6.4, 2.8, 5.6, 2.2, 5.6],
    [5.0, 2.3, 3.3, 1.0, 1.3],
    [4.9, 2.5, 4.5, 1.7, 5.7],
    [4.9, 3.1, 1.5, 0.1, 3.2],
    [5.7, 3.8, 1.7, 0.3, 3.1],
    [4.4, 3.2, 1.3, 0.2, 3.2]
]

TrainSetY = [
    [2],
    [1],
    [2],
    [0],
    [0],
    [0]
]

number_of_features = 5                   # number of the columns in the data-set
number_of_classes = 3                    # number of the target classes from the data-set
hidden_layers = [10, 15, 10]             # Three layers with respective node counts
shape_of_dataset = [number_of_features]


# "hidden_layers" is the Iterable of number hidden units per layer.
# All layers are fully connected. Ex. [64, 32] means first layer
# has 64 nodes and second one has 32.

EXPORT_PATH = "/path/to/model"


def get_serving_input():
    feature_spec = {"x": tf.FixedLenFeature(dtype=tf.float32, shape=shape_of_dataset)}
    model_placeholder = tf.placeholder(dtype=tf.string,
                                       shape=[None],
                                       name='input'
                                       )
    receiver_tensors = {
        "model_inputs": model_placeholder
    }
    features = tf.parse_example(model_placeholder, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def main():
    feature_columns = [tf.feature_column.numeric_column("x", shape=shape_of_dataset)]

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_layers,
        n_classes=number_of_classes)

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(TrainSetX)},
        y=np.array(TrainSetY),
        num_epochs=None,
        shuffle=True
    )

    # Train model.
    classifier.train(
        input_fn=train_input_fn,
        steps=1
    )

    exported_model_path = classifier.export_savedmodel(
        export_dir_base=EXPORT_PATH,
        serving_input_receiver_fn=get_serving_input
    )

    print exported_model_path


if __name__ == '__main__':
    main()
