import ml_collections


def mnist_experiment():
    config = ml_collections.ConfigDict()
    config.model = 'mnist'
    config.num_classes = 10

    # Training
    config.batch_size = 120
    config.num_epochs = 25
    config.train_size = 10000 
    config.aug_size = 200  # Number of augmented images

    # Validation
    config.with_val = False
    config.val_size = 0.9

    # Optimizer
    config.schedule = "exp_decay"
    config.learning_rate = 0.01
    config.decay_rate = 0.9999
    config.delta = 0.1 
    config.patience = 5
    config.decay_steps = config.train_size // config.batch_size
    if config.with_val: config.decay_steps *= config.val_size
    
    # Regularization
    config.lambda_l2 = 0.001 
    config.lambda_core = 0.2 

    # Early stopping
    config.with_earlystop = False

    # Counterfactual annealing
    config.cfl_anneal = False
    config.no_cfl_frac = 0.1
    return config


def celebA_experiment():
    config = ml_collections.ConfigDict()
    config.model = 'celebA'
    config.num_classes = 2

    # Training
    config.batch_size = 128
    config.num_epochs = 100
    config.train_size = 20000
    config.test_size = (4000, 1200)
    config.prop_glasses = 0.8 # Proportion of men with and women without glasses in training set

    # Validation
    config.with_val = False
    config.val_size = 0.9

    # Optimizer
    config.schedule = "warmup_decay"
    config.learning_rate = 1e-3
    config.decay_steps = 1000
    config.warmup_steps = 100
    config.end_learning_rate = 1e-4

    # Regularization
    config.lambda_l2 = 0.001 
    config.lambda_core = 0.2
    
    # Early stopping
    config.with_earlystop = False
    config.delta = 0.1 
    config.patience = 25

    # Counterfactual annealing
    config.cfl_anneal = False
    config.no_cfl_frac = 0.1
    return config


def synthetic_experiment():
    config = ml_collections.ConfigDict()
    config.model = 'synthetic'
    config.num_classes = 2

    # Training
    config.batch_size = 120
    config.num_epochs = 30
    config.train_size = 20000

    # Validation
    config.with_val = False
    config.val_size = 0.9

    # Optimizer
    config.schedule = "exp_decay"
    config.learning_rate = 0.001
    config.decay_rate = 0.9999
    config.decay_steps = config.train_size // config.batch_size
    if config.with_val: config.decay_steps *= config.val_size

    # Regularization
    config.lambda_l2 = 0.001
    config.lambda_core = 1.0
    
    # Early stopping
    config.with_earlystop = False
    config.delta = 0.001 
    config.patience = 3

    # Counterfactual annealing
    config.cfl_anneal = False
    config.no_cfl_frac = 0.1
    return config