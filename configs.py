import ml_collections


def mnist_experiment():
    config = ml_collections.ConfigDict()
    config.learning_rate = 0.01
    config.batch_size = 120
    config.num_epochs = 25
    config.lambda_l2 = 0.001  # L2 regularization strength
    config.lambda_core = 0.2  # Conditional variance regularization strength
    config.train_size = 10000  # Size of training set
    config.aug_size = 200  # Number of augmented images
    return config
