import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Similarity.Compute_Similarity_Python import Incremental_Similarity_Builder
import time, sys


class HHPRecommender(BaseItemSimilarityMatrixRecommender):
    '''
    HHP (Hybrid HeatS - ProbS) is an hybrid of ProbS (also called P3) and HeatS that uses lambda
    to weight the two contributions. While ProbS implements a random walk with 3 steps, HeatS uses the
    out degree of the neighboring objects.
    
    Original Paper: https://arxiv.org/pdf/0808.2670.pdf
    '''
    
    RECOMMENDER_NAME = "HHPRecommender"
    
    def __init__(self, URM_train):
        super(HHPRecommender, self).__init__(URM_train)
        
    def fit(self, hybrid_lambda=0.5, topK=100, min_rating=0, implicit=False, normalize_similarity=True):
        
        self.hybrid_lambda = hybrid_lambda
        self.topK = topK
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)
                
        k_u = np.ediff1d(sps.csr_matrix(self.URM_train).indptr) # user out degree
        k_i = np.ediff1d(sps.csc_matrix(self.URM_train).indptr) # item out degree
        heatS_contribution = (1 / (np.float_power(k_i, 1 - self.hybrid_lambda) + 1e-6)).reshape(1, -1)
        
        Pui = normalize(self.URM_train, norm='l1', axis=1)
        
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        Piu_lambda = X_bool.multiply((1 / (np.float_power(k_i, self.hybrid_lambda) + 1e-6)).reshape(-1, 1))
        Piu_lambda = Piu_lambda.tocsr()

        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu_lambda

        similarity_builder = Incremental_Similarity_Builder(Pui.shape[1], initial_data_block=Pui.shape[1]*self.topK, dtype = np.float32)

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.multiply(heatS_contribution).toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                relevant_items_partition = np.argpartition(-row_data, self.topK-1, axis=0)[:self.topK]
                row_data = row_data[relevant_items_partition]

                # Incrementally build sparse matrix, do not add zeros
                if np.any(row_data == 0.0):
                    non_zero_mask = row_data != 0.0
                    relevant_items_partition = relevant_items_partition[non_zero_mask]
                    row_data = row_data[non_zero_mask]

                similarity_builder.add_data_lists(row_list_to_add=np.ones(len(row_data), dtype = np.int) * (current_block_start_row + row_in_block),
                                                  col_list_to_add=relevant_items_partition,
                                                  data_list_to_add=row_data)


            if time.time() - start_time_printBatch > 300 or current_block_start_row + block_dim == Pui.shape[1]:
                new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time)

                self._print("Similarity column {} ({:4.1f}%), {:.2f} column/sec. Elapsed time {:.2f} {}".format(
                     current_block_start_row + block_dim,
                    100.0 * float( current_block_start_row + block_dim) / Pui.shape[1],
                    float( current_block_start_row + block_dim) / (time.time() - start_time),
                    new_time_value, new_time_unit))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = similarity_builder.get_SparseMatrix()

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')