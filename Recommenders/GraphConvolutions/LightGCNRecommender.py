import tensorflow as tf
import numpy as np
import gc

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping


class LightGCNRecommender(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    
    RECOMMENDER_NAME = "LightGCNRecommender"
    AVAILABLE_OPTIMIZERS = ["adam", "sgd", "rmsprop", "adagrad"]
    AVAILABLE_LOSSES = ["mse", "xentropy"] # todo: add bpr
    
    def __init__(self, URM_train):
        super(LightGCNRecommender, self).__init__(URM_train)
        
        # building the tensorflow sparse tensor, .T is done because an array of (row, col) is needed
        URM_train_coo = self.URM_train.tocoo()
        self.URM_train_tensor = tf.sparse.SparseTensor(
            np.array([URM_train_coo.row, URM_train_coo.col]).T, URM_train_coo.data, URM_train_coo.shape
        )
        
    def fit(
        self, 
        epochs=100, 
        num_layers=2,
        embedding_size=10,
        learning_rate=1e-3,
        init_mean=0.0,
        init_std=0.01,
        optimizer='adam',
        loss='mse',
        l2_user=1e-2,
        l2_item=1e-2,
        **earlystopping_kwargs
    ):
        
        if optimizer not in self.AVAILABLE_OPTIMIZERS:
            raise ValueError("Value for 'optimizer' not recognized. Acceptable values are {}, provided was '{}'".format(self.AVAILABLE_OPTIMIZERS, optimizer))
            
        if loss not in self.AVAILABLE_LOSSES:
            raise ValueError("Value for 'loss' not recognized. Acceptable values are {}, provided was '{}'".format(self.AVAILABLE_LOSSES, loss))

        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        
        self._build_optimizer(optimizer)
        self._build_loss(loss)
        
        # the only variables are the embeddings at layer 0
        self.user_embeddings_initial_step = tf.Variable(tf.random.normal([self.n_users, self.embedding_size], mean=init_mean, stddev=init_std))
        self.item_embeddings_initial_step = tf.Variable(tf.random.normal([self.n_items, self.embedding_size], mean=init_mean, stddev=init_std))
        
        self.user_embeddings_reg = tf.keras.regularizers.l2(l2_user)(self.user_embeddings_initial_step)
        self.item_embeddings_reg = tf.keras.regularizers.l2(l2_item)(self.item_embeddings_initial_step)
        
        # neighborhood sizes of the items
        self.N_i = tf.sparse.reduce_sum(self.URM_train_tensor, axis=0)
        self.N_i = tf.sqrt(self.N_i + 1e-6)
        self.N_i_inv = 1.0 / self.N_i
        self.N_i_inv = tf.reshape(self.N_i_inv, shape=[-1, 1])
        
        # neighborhood sizes of the users
        self.N_u = tf.sparse.reduce_sum(self.URM_train_tensor, axis=1)
        self.N_u = tf.sqrt(self.N_u  + 1e-6)
        self.N_u_inv = 1.0 / self.N_u
        self.N_u_inv = tf.reshape(self.N_u_inv, shape=[-1, 1])
        
        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # the final user and items factors, obtained by applying convolutions for num_layers times
        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best
        
    def _build_optimizer(self, optimizer):
        # todo: consider adding the other parameters of the optimizers
        if optimizer == 'adam':
            self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = tf.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            self.optimizer = tf.optimizers.Adagrad(learning_rate=self.learning_rate)
            
    def _build_loss(self, loss):
        if loss == 'mse':
            self.loss_function = tf.losses.mean_squared_error
        else:
            self.loss_function = tf.losses.binary_crossentropy
    
    def _run_epoch(self, num_epoch):
        
        with tf.GradientTape(persistent=True) as tape:
            user_embeddings = self.user_embeddings_initial_step
            item_embeddings = self.item_embeddings_initial_step

            # k steps (layers) of convolutions
            for _ in range(self.num_layers):
                # computing the embeddings at step k+1 using the embedding at step k
                # this means that user embeddings have to be updated only at the end
                user_embeddings_update = self._graph_convolution_user(item_embeddings)
                item_embeddings_update = self._graph_convolution_item(user_embeddings)
                
                user_embeddings = user_embeddings_update
                item_embeddings = item_embeddings_update
                
            user_item_scores = tf.matmul(user_embeddings, item_embeddings, transpose_b=True)
            
            regularization = self.user_embeddings_reg + self.item_embeddings_reg
            loss = self.loss_function(tf.sparse.to_dense(self.URM_train_tensor), user_item_scores) + regularization

        user_gradients = tape.gradient(loss, self.user_embeddings_initial_step)
        item_gradients = tape.gradient(loss, self.item_embeddings_initial_step)

        self.optimizer.apply_gradients(zip([user_gradients, item_gradients], [self.user_embeddings_initial_step, self.item_embeddings_initial_step]))
        
        self._update_factors()
        gc.collect()

        del tape
        
    def _graph_convolution_user(self, item_embeddings):
        normalized_embeddings = item_embeddings * self.N_i_inv

        conv_output = tf.sparse.sparse_dense_matmul(self.URM_train_tensor, normalized_embeddings)
        return conv_output * self.N_u_inv
    
    def _graph_convolution_item(self, user_embeddings):
        normalized_embeddings = user_embeddings * self.N_u_inv

        conv_output = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(self.URM_train_tensor), normalized_embeddings)
        return conv_output * self.N_i_inv
        
    def _update_factors(self):
        self.USER_factors = self.user_embeddings_initial_step.numpy().copy()
        self.ITEM_factors = self.item_embeddings_initial_step.numpy().copy()
        
        for _ in range(self.num_layers):
            USER_factors = self._graph_convolution_user(self.ITEM_factors).numpy()
            ITEM_factors = self._graph_convolution_item(self.USER_factors).numpy()
            
            self.USER_factors = USER_factors
            self.ITEM_factors = ITEM_factors
        
    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        
    def _prepare_model_for_validation(self):
        pass