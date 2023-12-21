import torch
import pandas as pd
import random

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.GraphConvolutions.PyTorch.LightGCN_Model import LightGCN_Model


class LightGCNRecommender_PyTorch(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    
    RECOMMENDER_NAME = "LightGCNRecommender_PyTorch"
    AVAILABLE_OPTIMIZERS = ["adam", "sgd", "rmsprop", "adagrad", "adadelta"]
    
    def __init__(self, URM_train):
        super(LightGCNRecommender_PyTorch, self).__init__(URM_train)
        
        # building the tensorflow sparse tensor, .T is done because an array of (row, col) is needed
        URM_train_coo = self.URM_train.tocoo()
        self.URM_train_df = pd.DataFrame({'UserID': URM_train_coo.row, 'ItemID': URM_train_coo.col, 'Rating': URM_train_coo.data})
        
        self.interacted_items_df = self.URM_train_df.groupby('UserID')['ItemID'].apply(list).reset_index()
        self.warm_user_ids = self.interacted_items_df.UserID.values
        
    def fit(
        self, 
        epochs=100, 
        num_layers=2,
        embedding_size=10,
        learning_rate=1e-3,
        batch_size=64,
        optimizer='adam',
        l2_reg=1e-2,
        **earlystopping_kwargs
    ):
        
        if optimizer not in self.AVAILABLE_OPTIMIZERS:
            raise ValueError("Value for 'optimizer' not recognized. Acceptable values are {}, provided was '{}'".format(self.AVAILABLE_OPTIMIZERS, optimizer))

        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        
        self.lightGCN_model = LightGCN_Model(self.URM_train_df, self.n_users, self.n_items, self.num_layers, self.embedding_size)
        
        self._print('Built LightGCN Model, starting to train')
        
        self._build_optimizer(optimizer)
        
        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best
        
    def _build_optimizer(self, optimizer):
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.lightGCN_model.parameters(), lr=self.learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.lightGCN_model.parameters(), lr=self.learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.lightGCN_model.parameters(), lr=self.learning_rate)
        elif optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.lightGCN_model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.Adadelta(self.lightGCN_model.parameters(), lr=self.learning_rate)
            

    def _run_epoch(self, num_epoch):
        
        n_batch = int(len(self.warm_user_ids) / self.batch_size)
  
        self.lightGCN_model.train()
        for batch_idx in range(n_batch):

            self.optimizer.zero_grad()

            users, pos_items, neg_items = self._data_loader()

            users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = self.lightGCN_model.forward(users, pos_items, neg_items)

            mf_loss, reg_loss = self._bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0)
            reg_loss = self.l2_reg * reg_loss
            final_loss = mf_loss + reg_loss

            final_loss.backward()
            self.optimizer.step()
            
        embeddings = self.lightGCN_model.E0.weight.detach().numpy()
        self.USER_factors = embeddings[:self.n_users, :]
        self.ITEM_factors = embeddings[self.n_users:, :]
        
    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        
    def _prepare_model_for_validation(self):
        pass
    
    def _bpr_loss(self, users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0):
  
        reg_loss = (1/2)*(userEmb0.norm().pow(2) + 
                        posEmb0.norm().pow(2)  +
                        negEmb0.norm().pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss
    
    def _data_loader(self):

        def sample_neg(x):
            while True:
                neg_id = random.randint(0, self.n_items - 1)
                if neg_id not in x:
                    return neg_id

        if self.n_users < self.batch_size:
            indices = [x for x in range(len(self.warm_user_ids))]
            users = [self.warm_user_ids[random.choice(indices)] for _ in range(self.batch_size)]
        else:
            users = random.sample(self.warm_user_ids.tolist(), self.batch_size)

        users.sort()
        users_df = pd.DataFrame(users, columns = ['users'])

        interacted_items_df = pd.merge(self.interacted_items_df, users_df, how = 'right', left_on = 'UserID', right_on = 'users')

        pos_items = interacted_items_df['ItemID'].apply(lambda x: random.choice(x)).values

        neg_items = interacted_items_df['ItemID'].apply(lambda x: sample_neg(x)).values

        return list(users), list(pos_items), list(neg_items)