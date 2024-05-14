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
    config.decay_steps = config.train_size // config.batch_size
    if config.with_val: config.decay_steps *= config.val_size
    
    # Regularization
    config.lambda_l2 = 0.001 
    config.lambda_core = 0.2 

    # Early stopping
    config.with_earlystop = False
    config.delta = 0.1 
    config.patience = 5

    # Counterfactual annealing
    config.cfl_anneal = False
    config.no_cfl_frac = None
    return config


def celebA_experiment():
    config = ml_collections.ConfigDict()
    config.model = 'celebA'
    config.num_classes = 2

    # Training
    config.batch_size = 256
    config.num_epochs = 50
    config.train_size = 16982
    config.test_size = (4224,1120)
    config.prop_glasses = 0.95 # Proportion of men with and women without glasses in training set

    # Validation
    config.with_val = False
    config.val_size = None

    # Optimizer
    config.schedule = "warmup_exp_decay"
    config.learning_rate = 1e-2
    config.warmup_steps = 100
    config.decay_steps = 80
    config.decay_rate = 0.99

    # Regularization
    config.lambda_l2 = 1e-3
    config.lambda_core = 10.0
    
    # Early stopping
    config.with_earlystop = False
    config.delta = None
    config.patience = None

    # Counterfactual annealing
    config.cfl_anneal = False
    config.no_cfl_frac = None
    return config


def synthetic_experiment():
    config = ml_collections.ConfigDict()
    config.model = 'synthetic'
    config.num_classes = 2

    # Training
    config.batch_size = 256
    config.num_epochs = 30
    config.train_size = 20000

    # Validation
    config.with_val = False
    config.val_size = None

    # Optimizer
    config.schedule = "warmup_exp_decay"
    config.learning_rate = 1e-2
    config.warmup_steps = 100
    config.decay_steps = 80
    config.decay_rate = 0.99

    # Regularization
    config.lambda_l2 = 0.01
    config.lambda_core = 10.0
    
    # Early stopping
    config.with_earlystop = False
    config.delta = None
    config.patience = None

    # Counterfactual annealing
    config.cfl_anneal = False
    config.no_cfl_frac = None
    return config


def hybrid_AE():
    config = ml_collections.ConfigDict()

    config.lambda_aux = 0.1
    config.batch_size = 256
    config.input_shape = (64, 64, 1)
    config.learning_rate = 1e-3
    config.num_epochs = 50
    return config