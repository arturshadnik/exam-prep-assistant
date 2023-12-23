import os
import json
import logging
import numpy as np
import faiss
from time import time
from openai import Embedding

from app.core.database import storage
from app.core.logging import logger

class Indexer:
    def __init__(self):
        self.chunk_index = None
        self.chunks = []
        self.chunk_doc_ids = []
        self.list_of_index = []
        self.ACCURACY_THRESHOLD = float(os.getenv('ACCURACY_THRESHOLD'))
        self.EMBEDDINGS_BUCKET_NAME = os.getenv('EMBEDDINGS_BUCKET_NAME')
        self.VIEWABLE_BUCKET_NAME = os.getenv('VIEWABLE_BUCKET_NAME')

    def get_doc_id(self, index: int) -> str:
        return self.chunk_doc_ids[self.list_of_index[index]]

    def load_chunk_data(self):
        tic = time.time()
        bucket = storage.bucket(self.EMBEDDINGS_BUCKET_NAME)
        blobs = bucket.list_blobs()
        self.chunk_doc_ids = [blob.name for blob in blobs if not blob.name.endswith('.pdf') and blob.name != '80f0866dfea20cce09611243e17c22e5aba4dfc1ca716a87d8893abefcd63ffd']

        embeddings = []
        for idx, blob_id in enumerate(self.chunk_doc_ids):
            blob = bucket.get_blob(blob_id)
            data = json.loads(blob.download_as_string())
            num_embeddings = len(data['embeddings'])
            embeddings.extend(data['embeddings'])
            self.chunks.extend(data['chunks'])
            self.list_of_index.extend([idx] * num_embeddings)

        embeddings = np.array(embeddings).astype('float32')
        self.chunk_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.chunk_index.add(embeddings)
        logger.info(f'Loaded chunk data in {time.time() - tic} seconds')

    def load_chunks_on_startup(self):
        self.load_chunk_data()

    def add_chunk_to_index(self, folder_id: str, doc_id: str):
        if self.chunk_index is None:
            return 'Index not loaded'
        if doc_id in self.chunk_doc_ids:
            return f'Document ID {doc_id} already added to index'
        bucket = storage.bucket(self.EMBEDDINGS_BUCKET_NAME)
        blob = bucket.get_blob(f'{folder_id}/{doc_id}')
        if not blob:
            return f'Document ID {doc_id} not found in storage'
        try:
            data = json.loads(blob.download_as_string())
        except Exception as e:
            return f'Error processing blob data for doc ID {doc_id}: {str(e)}'
        num_embeddings = len(data['embeddings'])
        self.chunk_index.add(np.array(data['embeddings']).astype('float32'))
        self.chunks.extend(data['chunks'])
        self.chunk_doc_ids.append(doc_id)
        self.list_of_index.extend([len(self.chunk_doc_ids) - 1] * num_embeddings)
        return f'Document ID {doc_id} added to index'

    def find_chunks(self, question: str, count: int):
        if self.chunk_index is None:
            logger.info("Chunk data not found, loading...")
            self.load_chunk_data()
        data = Embedding.create(input=question, engine='text-embedding-ada-002').data
        embedded_question = data[0]['embedding']
        D, I = self.chunk_index.search(np.array([embedded_question]), count)
        scores = D[0].tolist()
        logger.info(scores)
        return_doc_ids = [self.get_doc_id(I[0][idx]) for idx, score in enumerate(scores) if score >= self.ACCURACY_THRESHOLD]
        return_chunks = [self.chunks[I[0][idx]] for idx, score in enumerate(scores) if score >= self.ACCURACY_THRESHOLD]
        scores = [score for score in scores if score >= self.ACCURACY_THRESHOLD]
        return {
            'doc_ids': return_doc_ids,
            'chunks': return_chunks,
            'scores': scores
        }