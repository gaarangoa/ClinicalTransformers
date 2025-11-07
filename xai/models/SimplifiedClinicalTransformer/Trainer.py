import tensorflow as tf
import pickle 
import os
import pandas as pd
import numpy as np 
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from ...preprocessing.discretize import CategoricalConverter
from ...tokenizer.FeatureTokenizer import FeatureTokenizer

import logging
logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger = logging.getLogger('Trainer')

class Trainer():
    '''
        General Information
        -----------------
        This is a high level API for running the clninical transformer in multiple settings: 
        1. self-supervision
        2. classification
        3. survival

        4. Fine tunning: First running the self supervised mode and then running any other mode.

        Usage:
        -----------------
        This is a simple way to use the clinical transformer to perform self-supervised task:
        
        trainer = Trainer(
            out_dir = '/scratch/kmvr819/data/xai/training/simulated/self-supervision/',
            max_features=5,
            test_size=0.2,
            mode='self-supervision',
        )

        trainer.setup_data(
            data, 
            discrete_features = [],
            continuous_features =  features,
            target=['label']
        )

        # trainer.setup_prior(priors)

        trainer.setup_model(
            embedding_size=64, 
            num_heads=2, 
            num_layers=4,
            learning_rate=0.001,
            batch_size_max=True, # This will take a batch size of the training / testing shape,
            save_best_only=False
        )

        trainer.fit(repetitions=1, epochs=1000, verbose=0)

    ''' 
    def __init__(self, out_dir, max_features_percentile=75, test_size=0.2, mode='survival', from_pretrained=None, max_features=None, model=None, dataloader=None, loss=None, metrics=None):
        
        if max_features == None:
            max_features = 0

        self.max_features_percentile = max_features_percentile
        if max_features > 0:
            self.max_features_percentile = 0

        self.out_dir = out_dir
        self.sel_max_features = max_features
        self.test_size=test_size
        self.mode = mode
        self.priors = False
        self.description = ''

        logger.info('Setting up working directory: {}'.format(self.out_dir))
        os.makedirs(self.out_dir)

        # Transfer learning. If we transfer the weights from a previous model we need to use 
        # its vocabulary (Features), the input size (max_features) and the same topology (heads, layers).
        # All those parameters are retrieved from the pretrained model. And the new model is limited to those
        # parameters. For instance, it can only use the features available on the pretrained model.
        self.with_pretrained = False
        if from_pretrained:
            from_pretrained_path = "/".join(from_pretrained.split('/')[:-1])
            
            self.out_dir = "{}/{}/".format(out_dir, from_pretrained.split('/')[-1])
            os.makedirs(self.out_dir)
            logger.info('Setting up transfer learning directory: {}'.format(self.out_dir))

            self.with_pretrained = True

            # Load the pretrained model as base_model in the trainer class. 
            self.base_model = pickle.load( open('{}/traineer.pk'.format(from_pretrained_path), 'rb' ))
            
            # Load the testing set from the pretrained model to load the model and obtain the weights (this is how tensorflow works!)
            # _, self.base_model.testing_data_generator = pickle.load(open("{}/train_test_dataset_generator.pk".format(from_pretrained_path), 'rb'))
            self.base_model.load_model_weights(epoch=None, file='{}'.format(from_pretrained))

            # Extract weights from pretrained model
            self.base_model_encoder_weights = self.base_model.model.encoder.get_weights()
            
            # Remove base model to free memory space. 
            self.base_model.model = []
            
            # Copy priors from pretrained model (if used) to new model
            self.priors = self.base_model.priors
            if self.priors:
                logger.info('Using priors from pre-trained model: {}priors.pk'.format(self.base_model.out_dir))
                os.system('cp {} {}'.format('{}/priors.pk'.format(self.base_model.out_dir), '{}/priors.pk'.format(self.out_dir)))

        if self.mode == 'survival':

            # Set Model
            if model:
                self.Transformer = model

            # Set data loader
            if dataloader: 
                self.DataGenerator = dataloader

            # Set loss
            if loss:
                self.loss= loss

            # Set Metrics
            if metrics: 
                self.metrics = metrics

            # Adding <MASK> token to the model is set to False as this is used 
            # only for self supervision to predict the <MASK> tokens by looking
            # at the other features.
            self.add_mask = False

        if self.mode == 'survival+':

            # Set Model
            if model:
                self.Transformer = model

            # Set data loader
            if dataloader: 
                self.DataGenerator = dataloader

            # Set loss
            if loss:
                self.loss= loss

            # Set Metrics
            if metrics: 
                self.metrics = metrics

            # Adding <MASK> token to the model is set to True as this is used 
            # only for self supervision to predict the <MASK> tokens by looking
            # at the other features.
            self.add_mask = True
        
        if self.mode == 'self-supervision': 

            # Set model class
            # self.Transformer = SelfSupervisedTransformer
            if model:
                self.Transformer = model

            # Set data loader
            # self.DataGenerator = SelfSupervisedDataGenerator
            if dataloader:
                self.DataGenerator = dataloader
            
            # Set loss
            # self.loss =  tf.keras.losses.SparseCategoricalCrossentropy(
            #     reduction=tf.keras.losses.Reduction.NONE
            # )
            if loss:
                self.loss= loss
            
            # Set metrics
            self.metrics = []
            if metrics: 
                self.metrics = metrics

            self.add_mask = True

        if self.mode == 'value-self-supervision':

            if model:
                self.Transformer = model

            if dataloader:
                self.DataGenerator = dataloader

            if loss:
                self.loss = loss
            else:
                from ...losses.selfsupervision.value_mask import ValueMaskLoss

                self.loss = ValueMaskLoss()

            self.metrics = []
            if metrics:
                self.metrics = metrics

            self.add_mask = True

        if self.mode == 'classification':
            # Set model class
            # self.Transformer = SimplifiedClassifierTransformer 
            if model:
                self.Transformer = model

            # Set data loader class
            # self.DataGenerator = ClassifierGenerator
            if dataloader:
                self.DataGenerator = dataloader 
            
            # Set loss
            # self.loss = tf.keras.losses.SparseCategoricalCrossentropy() 
            if loss:
                self.loss= loss

            # Set metrics
            # self.metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
            if metrics: 
                self.metrics = metrics

            self.add_mask = False

    def setup_data(self, data, discrete_features, continuous_features, target=[], normalize=True): 
        '''
        This function takes the input dataframe and the discrete and continuous features. We use this way because
        we need to process discrete and continuous features separately. For discrete features we encode the values 
        into numbers. 

        '''

        # If we are doing survival, target is [time, event], if we are doing classification it corresponds
        # column with the classes 
        self.target = target if len(target) > 0 else target

        # Copy the input data to not modify its values
        data = data.copy()

        # set the discrete and continuous features and whether we are normalizing the data (default is True)
        self.discrete_features = discrete_features
        self.continuous_features = continuous_features
        self.normalize=normalize

        # For classification we set the total number of classes as the unique values in the target variable
        if self.mode == 'classification':
            self.num_classes = len(set(np.array(data[target])[:, 0]))
        
        logger.info('Number of continuous features: {}'.format(len(self.continuous_features)))
        logger.info('Number of discrete features: {}'.format(len(self.discrete_features)))
        logger.info('Number of samples: {}'.format(len(data)))
        
        # Features are the continuous and discrete features
        self.features = self.discrete_features + self.continuous_features

        # If we use a pretrained model, then we will use the tokenizer (feature names as integers) and the 
        # Data converter (from categorical to numbers) from the pretrained model. If not, we initialize 
        # both in here.
        if self.with_pretrained:
            # Loads tokenizer and data converter from pretrained model
            self.tokenizer = self.base_model.tokenizer
            self.data_converter = self.base_model.data_converter
        else:
            self.tokenizer = FeatureTokenizer(self.features)
            self.data_converter = CategoricalConverter()
            # Encode the categorical features
            self.data_converter.encode(data[self.features])
        
        # Transform the categorical features to numbers and replace it on the data. This looks at all the 
        # columns in the dataset, but, later we only use the features defined in self.features.
        data_ = self.data_converter.transform(data)

        # Our dataset only contains the target variables and the features.
        dataset = data_[self.target + self.features]

        # As we need to normalize our dataset using the min/max scaler, if we are using the pretrained model
        # we need to use the min / max per variable from the pretrained model. If not, we define our 
        # min max in here. Note that for getting the min and max we are setting the NaN values to 0. 
        if self.with_pretrained:
            # Load min and max feature values from pretrained model
            self.dataset_min = self.base_model.dataset_min
            self.dataset_max = self.base_model.dataset_max
        else:
            self.dataset_min = dataset[self.features].fillna(0).min()
            self.dataset_max = dataset[self.features].fillna(0).max()
        
        if self.normalize:
            # Normalize the data with either the pretrained values or the actual vaulues if the model is trained from stratch
            dataset_molecular_normalized = (dataset[self.features] - self.dataset_min) / (self.dataset_max - self.dataset_min)
            dataset = pd.concat([dataset[self.target], dataset_molecular_normalized ], axis=1).reset_index(drop=True)
        
        # We save the ready-to-go dataset as it can take a big portion of the memory. 
        pickle.dump(dataset, open('{}/dataset.pk'.format(self.out_dir) , 'wb'))

    def setup_prior(self, priors):
        '''Add prior knowledge to the model. In this function we are just setting the variable priors to True and saving the
           prior data into a dataframe that will be used later when the data is loaded. 
        '''
        self.priors = True
        pickle.dump(priors, open('{}/priors.pk'.format(self.out_dir) , 'wb'))
        
    def setup_model(self, embedding_size=128, num_heads=4, num_layers=4, batch_size=1024, learning_rate=0.00001, batch_size_max=False, save_best_only=True):
        
        # If we are using a pretrained model, then the architecture of the model, the embedding size, number of heads and 
        # layers are inherited form the pretrained model. If not, then we setup those parameters in here.
        if self.with_pretrained:
            self.embedding_size = self.base_model.embedding_size
            self.num_heads = self.base_model.num_heads
            self.num_layers = self.base_model.num_layers
        else:
            self.embedding_size = embedding_size
            self.num_heads = num_heads
            self.num_layers = num_layers
        

        self.save_best_only = save_best_only
        self.batch_size_max = batch_size_max
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # The number of classes is defined depending on the task. For classification this number is defined in the setup 
        # data function (We should move it here). In here, we are defining for the other tasks. For regression and 
        # survival, the classes is 1 as the model only returns one value.
        if self.mode == 'survival' or self.mode == 'regression' or self.mode == 'survival+':
            self.num_classes = 1
        elif self.mode == 'self-supervision':
            self.num_classes = self.tokenizer.vocabulary_size
        elif self.mode == 'value-self-supervision':
            self.num_classes = 1
        
        logger.info('Number of classes: {}'.format(self.num_classes)) 
            
    def data_generator(self, dataset, batch_size, max_features, normalize=False, shuffle=True, augment_copies=2, training=False):
        
        # The data generator uses the customized data loader that allows us to obtain the data in an organized way. 
        

        # If we are using priors, then, we load te priors in here. If the priors are from the pretrained model, those are 
        # already copied tothe pretrained model to this location. If not, we use the priors defined in the model call 
        if self.priors:
            priors = pickle.load(open('{}/priors.pk'.format(self.out_dir) , 'rb'))
        else:
            priors = []

        # Normalize the data when we are using an external dataset not used in the training. For training the default is 
        # False as the data is already normalized on the setup_data function. however, when we apply our model to other 
        # datasets, we need to normalize it. 
        if normalize == True:
            dataset_molecular_normalized = (dataset[self.features] - self.dataset_min) / (self.dataset_max - self.dataset_min)
            try:
                dataset = pd.concat([dataset[self.target], dataset_molecular_normalized ], axis=1).reset_index(drop=True)
            except:
                logger.info("{} target variables not available. They are not used during inference, therefore setting them to np.nan".format(self.target))
                for t in self.target:
                    dataset[t] = np.nan
                dataset = pd.concat([dataset[self.target], dataset_molecular_normalized ], axis=1).reset_index(drop=True)

        dataset = dataset.reset_index(drop=True)

        # Call the data generator. This function wil load the data by batch_size. It will define the number of features selected
        # based on the percentile of the distribution of the featuers by patient that are greater than zero. And for each 
        # epoch the order of the features is selected randomly (see the dataloader class for details).
        return self.DataGenerator(
            data=dataset.reset_index(drop=True), 
            discrete_features=self.discrete_features,
            continuous_features=self.continuous_features,
            time=self.target[0] if 'survival' in self.mode else '',
            event=self.target[1] if 'survival' in self.mode else '',
            target=self.target,
            priors = priors,
            batch_size=batch_size,
            tokenizer=self.tokenizer,
            max_features_percentile=self.max_features_percentile,
            max_features = max_features,
            shuffle=shuffle,
            add_mask=self.add_mask,
            augment_copies=augment_copies,
            training=training
        )

    def split_dataset(self, test_size, fold=0):
        
        # This function takes the dataset and split it into training and testing. 

        self.test_size = test_size
        
        # Load the dataset
        dataset = pickle.load(open('{}/dataset.pk'.format(self.out_dir) , 'rb'))

        # Define the stratify variable to get equal number of samples per dataset. 
        if self.mode == 'survival' or self.mode == 'survival+':
            if len(self.target)>2: #for predictive loss function that has event_vector and treatment variable.
                
                stratify=dataset[[ self.target[1], self.target[2] ]  ]
            else:
                stratify = dataset[self.target[1]]
        elif self.mode == 'classification' or self.mode == 'regression':
            stratify = dataset[self.target]
        elif self.mode == 'self-supervision' or self.mode == 'value-self-supervision':
            stratify = [1]*dataset.shape[0]
        else:
            raise('Wrong mode parameter: {}'.format(self.mode))

        # Split the data in train and test. 
        # when test_size == 0, then train with the full dataset: 
        if test_size == 0 or test_size == None:
            X_train = dataset
            X_test = dataset[:10].copy()

        else:
            X_train, X_test = train_test_split(
                dataset, 
                test_size=test_size, 
                random_state=fold,
                stratify=stratify
            )
        
        # Define the batch size
        train_batch_size = self.batch_size
        test_batch_size = self.batch_size

        # Sometimes we have very small datsets, so, we can just do the batches to the total number of samples. 
        if self.batch_size_max:
            train_batch_size = X_train.shape[0]
            test_batch_size = X_test.shape[0]

        # If we are using a pretrained model, the max features we are taking by patient are already defined. 
        if self.with_pretrained:
            max_features = self.base_model.max_features
            self.max_features_percentile = self.base_model.max_features_percentile
        else: 
            # If we use the percentile, then the max_features variable is ignored if it is equal to 0
            max_features = 0 # if the max_featuers is 0 it will use the max_features percentile

        
        if self.sel_max_features > 0:
            max_features = self.sel_max_features

        # The augment copies creates a copy of each input sample, and the model takes the average of the predicted risk
        # score. This is a type of augmentation that helps when the input features are very sparce as the same patient can
        # have a very different set of features. Using this at training time can help to decrease the variability on the 
        # results. 
        augment_copies = 1
        if self.mode == 'survival':
            augment_copies = 2
        
        # Build the data generator for training set
        training_data_generator = self.data_generator(X_train, train_batch_size, max_features=max_features, augment_copies=augment_copies, training=False)
        
        # And use some of the parameters (max_features) from the training set to the test set for consistency. Shuffle is set to false, because we don't want to shuffle in the test
        # set the order of the samples. We always get the same patients in the same order. 
        testing_data_generator = self.data_generator(X_test, test_batch_size, max_features=training_data_generator.max_features, augment_copies=augment_copies, training=False, shuffle=False)
        
        self.max_features = training_data_generator.max_features
        
        # Save the dataset
        pickle.dump([X_train, X_test], open("{}/train_test_dataset.pk".format(self.out_dir_fold), 'wb'))
        pickle.dump([training_data_generator, testing_data_generator], open("{}/train_test_dataset_generator.pk".format(self.out_dir_fold), 'wb'))

        logger.info('Training samples: {}'.format(X_train.shape[0]))
        logger.info('Testing samples: {}'.format(X_test.shape[0]))

        # We also created a dummy data variable that can be used when initializing the model or loading the model.
        # This is the tensorflow way of getting the model working when loading the weights. 
        X, y = testing_data_generator.__getitem__(0)
        
        logger.debug('Process data shape: {}'.format( [np.array(i)[:1].shape for i in X] ))
        self.process_data_shape = [np.array(i)[:1].shape for i in X]

        self.dummy_data = [ np.zeros(shape=i) for i in self.process_data_shape]

        logger.info('Number of features at {}th percentile: {} that are non nans'.format(self.max_features_percentile, self.max_features))

        return training_data_generator, testing_data_generator
    
    def training_fold(self, fold, epochs=10, run_id=None):
        
        # Define the run id by taking the timestamp
        run_id = fold #time.time()
        logger.info('RUN ID: fold-{}_id-{}'.format(fold, run_id))

        self.out_dir_fold = "{}/fold-{}_id-{}/".format(self.out_dir, fold, run_id)
        
        os.makedirs(self.out_dir_fold)
        logger.info('RUN ID out directory: {}'.format(self.out_dir_fold))

        # setup tensorboard callbacks for metrics and for saving the epoch weights during training. 
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="{}/fold-{}_id-{}/".format(self.out_dir, fold, run_id))
        checkpoint_filepath = '{}/fold-{}_id-{}/'.format(self.out_dir, fold, run_id) + '/model.E{epoch:06d}.h5'
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=self.save_best_only
        )
        
        # Generate the training / testing data
        training_data_generator, testing_data_generator = self.split_dataset(
            test_size=self.test_size,
            fold=fold,
        )
        
        _metrics = self.metrics
        self.metrics = []
        pickle.dump(self, open('{}/fold-{}_id-{}/traineer.pk'.format(self.out_dir, fold, run_id) , 'wb'))
        self.metrics = _metrics

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=0.1
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # self.training_data_generator = training_data_generator
        model = self.Transformer(
            num_layers=self.num_layers, #change number of layers (more layers)
            d_model=self.embedding_size, 
            num_heads=self.num_heads, # reduce number of heads (one head)
            dff=self.embedding_size,
            num_classes=self.num_classes, 
            masking=True, # This masking here is referred to the <pad> Padding Masking process. Not the <MASK> token. 
            num_features=self.tokenizer.vocabulary_size, 
            embedding_size=self.embedding_size, 
            max_features=self.max_features,
            with_prior=self.priors
        )
        
        model.compile(
            optimizer=optimizer,
            loss=self.loss, 
            metrics=self.metrics
        )
        
        if self.with_pretrained:
            # If we used a pretrained model we need first to get the model ready and then set the weights from the 
            # pretrained model and set it to trainable as we will be modifying the weights. 

            _ = model([i[:2] for i in self.dummy_data])
            model.encoder.set_weights(self.base_model_encoder_weights) 
            model.encoder.trainable = True
        
        # Fit the model
        _ = model.fit(
            training_data_generator,
            verbose=self.verbose,
            epochs=epochs,
            validation_data = testing_data_generator,
            callbacks=[tensorboard_callback, model_checkpoint_callback]
        )
    
    def fit(self, epochs, repetitions=10, verbose=0, one_repetition=0, seed=0):

        tf.random.set_seed(seed)
        
        self.verbose = verbose
        
        if one_repetition>0:
            self.training_fold(one_repetition, epochs)
        else:
            for fold in range(repetitions):
                self.training_fold(fold, epochs)
    
    def predict(self, dataset, batch_size=128, normalize=True, transform=True):
        # this function is deprecated
        if transform:
            dataset = self.data_converter.transform(dataset)
        
        dataloader = self.data_generator(dataset, batch_size= batch_size, normalize=normalize, max_features=self.max_features, shuffle=False)
        out = []
        for X, y in dataloader:
            out.append(self.model(X))

        return out

    def load_model_weights(self, epoch, file=''):
        self.model = self.Transformer(
            num_layers=self.num_layers, #change number of layers (more layers)
            d_model=self.embedding_size, 
            num_heads=self.num_heads, # reduce number of heads (one head)
            dff=self.embedding_size,
            num_classes=self.num_classes, 
            masking=True,
            num_features=self.tokenizer.vocabulary_size, 
            embedding_size=self.embedding_size, 
            max_features=self.max_features, 
            with_prior=self.priors
        )
        
        _=self.model(self.dummy_data)
        
        if file:
            self.model.load_weights(file)
        else:
            self.model.load_weights("{dir}/model.E{epoch:06d}.h5".format(dir=self.out_dir_fold, epoch=epoch))
    
    def load_data_snapshot(self, path=''):
        # This function is deprecated
        dataset = pickle.load(open('{}/dataset.pk'.format(path) , 'rb'))
        X_train, X_test = pickle.load(open("{}/train_test_dataset.pk".format(path), 'rb'))
        training_data_generator, testing_data_generator = pickle.dump(open("{}/train_test_dataset_generator.pk".format(path), 'rb'))

        return dataset, X_train, X_test, training_data_generator, testing_data_generator
