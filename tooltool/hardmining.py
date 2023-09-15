"""
Tools to get 'hard examples'.
"""
import numpy as np
import torch


class BinaryIndex:
    def __init__(self, embeddings, embedding_identifiers=None, index_size=16):
        """
        Fast dynamic search for approximately closest neighbors.

        :param embeddings: [n_emb, c] ndarray. It is expected that each component is approximately standardized.
        :param embedding_identifiers: any associated ids. [n_emb] or [n_emb, D]
        """
        assert isinstance(embeddings, torch.Tensor)
        self.index_size = index_size

        n_keys, n_channels = embeddings.shape
        assert n_channels >= index_size
        assert index_size < 25, "there is little point in huge indices"

        if embedding_identifiers is None:
            self.entities = torch.arange(n_keys, device=embeddings.device, dtype=torch.int32)
        else:
            assert embedding_identifiers.shape[0] == n_keys
            self.entities = embedding_identifiers

        self.components = torch.randperm(n_channels)[:index_size]
        keys = BinaryIndex._binarize(embeddings, components=self.components)
        order = torch.argsort(keys, dim=0)
        n_elements = torch.bincount(keys, minlength=2**index_size).cpu().numpy()
        self.entities = self.entities[order]

        self.np_n_elements = n_elements
        self.np_starts = np.concatenate([[0], np.cumsum(n_elements)], axis=0)

    @staticmethod
    def _binarize(keys, components) -> torch.Tensor:
        result = torch.zeros(len(keys), dtype=torch.int32, device=keys.device)
        for c in components.tolist():
            result <<= 1
            result |= keys[:, c] > 0
        return result

    def query(self, query_embeddings, max_embeddings_per_query=100, allow_one_error=False) -> torch.Tensor:
        binary_queries = BinaryIndex._binarize(query_embeddings, self.components)
        # deduplication
        binary_queries = torch.unique(binary_queries).cpu().numpy()
        binary_query2nembeddings_left = {bq: max_embeddings_per_query for bq in binary_queries}

        result = []
        processed_bqs = set()

        def add_all_for(binary_query, max: int) -> int:
            if binary_query in processed_bqs:
                return 0
            if max <= 0:
                return 0
            start, end = self.np_starts[binary_query], self.np_starts[binary_query + 1]
            result.append(self.entities[start:end][:max])
            processed_bqs.add(binary_query)
            return len(result[-1])

        for bq in binary_queries:
            binary_query2nembeddings_left[bq] -= add_all_for(bq, max=binary_query2nembeddings_left[bq])

        if allow_one_error:
            for i in range(self.index_size):
                for bq in binary_queries:
                    bq_corrupted = bq ^ (1 << i)
                    binary_query2nembeddings_left[bq] -= add_all_for(
                        bq_corrupted, max=binary_query2nembeddings_left[bq]
                    )

        return torch.concatenate(result, dim=0)
