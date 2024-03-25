import ml_collections


def mnist_experiment():
    config = ml_collections.ConfigDict()
    config.model = 'mnist'
    config.num_classes = 10
    config.learning_rate = 0.01
    config.decay_rate = 0.9999
    config.batch_size = 120
    config.num_epochs = 25
    config.lambda_l2 = 0.001  # L2 regularization strength
    config.lambda_core = 0.2  # Conditional variance regularization strength
    config.train_size = 10000  # Size of training set
    config.aug_size = 200  # Number of augmented images
    return config


def celebA_experiment():
    config = ml_collections.ConfigDict()
    config.model = 'celebA'
    config.num_classes = 2
    config.learning_rate = 0.01
    config.decay_rate = 0.9999
    config.batch_size = 120
    config.num_epochs = 50
    config.lambda_l2 = 0.001  # L2 regularization strength
    config.lambda_core = 0.2  # Conditional variance regularization strength
    config.train_size = 20000  # Size of training set
    config.test_size = (4000, 1200) # Size of test sets
    config.prop_glasses = 0.8 # Proportion of men with and women without glasses in training set
    return config