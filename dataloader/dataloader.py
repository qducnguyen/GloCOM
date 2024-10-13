import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.sparse
import scipy.io
from sentence_transformers import SentenceTransformer
from utils import file_utils
from typing import List, Union

class DocEmbedModel:
    def __init__(
            self,
            model: Union[str, callable]="all-MiniLM-L6-v2",
            device: str='cpu',
            verbose: bool=False
        ):
        self.verbose = verbose

        if isinstance(model, str): 
            self.model = SentenceTransformer(model, device=device)
        else:
            self.model = model

    def encode(self,
               docs:List[str],
               convert_to_tensor: bool=False
            ):

        embeddings = self.model.encode(
                        docs,
                        convert_to_tensor=convert_to_tensor,
                        show_progress_bar=self.verbose
                    )
        return embeddings



class BasicDatasetWithGlobal:
    def __init__(self,
                 dataset_dir,
                 batch_size=200,
                 read_labels=False,
                 as_tensor=True,
                 contextual_embed=False,
                 doc_embed_model="all-MiniLM-L6-v2",
                 global_dir="global",
                 device='cpu'
                ):
        # bow: NxV
        # word_emeddings: VxD
        # vocab: V, ordered by word id.

        self.load_data(dataset_dir, read_labels, global_dir)
        self.vocab_size = len(self.vocab)

        print("data_size: ", self.train_bow.shape[0])
        print("vocab_size: ", self.vocab_size)
        print("average length: {:.3f}".format(self.train_bow.sum(1).sum() / self.train_bow.shape[0]))

        if contextual_embed:
            self.doc_embedder = DocEmbedModel(doc_embed_model, device)
            self.train_contextual_embed = self.doc_embedder.encode(self.train_texts)
            self.test_contextual_embed = self.doc_embedder.encode(self.test_texts)

            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if global_dir:
            self.global_bow_maps = np.stack([self.global_bow[idx_doc] for idx_doc in self.global_maps])

        if as_tensor:
            if contextual_embed:
                self.train_data = np.concatenate((self.train_bow, self.train_contextual_embed), axis=1)
                self.test_data = np.concatenate((self.test_bow, self.test_contextual_embed), axis=1)
            if global_dir:
                self.train_data = np.concatenate((self.train_bow, self.global_bow_maps), axis=1)
                self.test_data = np.concatenate((self.test_bow, self.global_bow_maps), axis=1) # test and train is the same for now

            if not contextual_embed and not global_dir:
                self.train_data = self.train_bow
                self.test_data = self.test_bow

            self.train_data = torch.from_numpy(self.train_data).to(device)
            self.test_data = torch.from_numpy(self.test_data).to(device)

            self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
            self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

    def load_data(self, path, read_labels, global_dir=None):

        self.train_bow = scipy.sparse.load_npz(f'{path}/bow.npz').toarray().astype('float32')

        # Basically in short text TM problem, we evaluate on the training set. So in here, test_bow is basically test_bow
        self.test_bow = scipy.sparse.load_npz(f'{path}/bow.npz').toarray().astype('float32')
        self.pretrained_WE = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')

        # Similar reasons like above
        self.train_texts = file_utils.read_text(f'{path}/texts.txt')
        self.test_texts = file_utils.read_text(f'{path}/texts.txt')

        if read_labels:
            # Similar reasons like above
            self.train_labels = np.loadtxt(f'{path}/labels.txt', dtype=int)
            self.test_labels = np.loadtxt(f'{path}/labels.txt', dtype=int)
        
        if global_dir:
            self.global_bow = scipy.sparse.load_npz(os.path.join(path, global_dir, "global_bow.npz")).toarray().astype('float32')
            self.global_maps = []
            with open(os.path.join(path, global_dir, "global_maps.txt")) as fIn:
                for data in fIn:
                    self.global_maps.append(int(data.strip()))

        self.vocab = file_utils.read_text(f'{path}/vocab.txt')


class DatasetWithIndex(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = self.data[idx]
        return idx, batch  

class BasicDatasetWithIndex:
    def __init__(self,
                 dataset_dir,
                 batch_size=200,
                 read_labels=False,
                 as_tensor=True,
                 contextual_embed=False,
                 doc_embed_model="all-MiniLM-L6-v2",
                 device='cpu'
                ):
        # bow: NxV
        # word_emeddings: VxD
        # vocab: V, ordered by word id.

        self.load_data(dataset_dir, read_labels)
        self.vocab_size = len(self.vocab)

        print("data_size: ", self.train_bow.shape[0])
        print("vocab_size: ", self.vocab_size)
        print("average length: {:.3f}".format(self.train_bow.sum(1).sum() / self.train_bow.shape[0]))

        if contextual_embed:
            self.doc_embedder = DocEmbedModel(doc_embed_model, device)
            self.train_contextual_embed = self.doc_embedder.encode(self.train_texts)
            self.test_contextual_embed = self.doc_embedder.encode(self.test_texts)

            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if as_tensor:
            if not contextual_embed:
                self.train_data = self.train_bow
                self.test_data = self.test_bow
            else:
                self.train_data = np.concatenate((self.train_bow, self.train_contextual_embed), axis=1)
                self.test_data = np.concatenate((self.test_bow, self.test_contextual_embed), axis=1)

            self.train_data = torch.from_numpy(self.train_data).to(device)
            self.test_data = torch.from_numpy(self.test_data).to(device)

            self.train_dataloader = DataLoader(DatasetWithIndex(self.train_data), batch_size=batch_size, shuffle=True)
            self.test_dataloader = DataLoader(DatasetWithIndex(self.test_data), batch_size=batch_size, shuffle=False)

    def load_data(self, path, read_labels):

        # Basically in short text TM problem, we evaluate on the training set. So in here, test_bow is basically test_bow
        self.train_bow = scipy.sparse.load_npz(f'{path}/bow.npz').toarray().astype('float32')
        self.test_bow = scipy.sparse.load_npz(f'{path}/bow.npz').toarray().astype('float32')
        self.pretrained_WE = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')

        # Similar reasons like above
        self.train_texts = file_utils.read_text(f'{path}/texts.txt')
        self.test_texts = file_utils.read_text(f'{path}/texts.txt')
        
        if read_labels:
            # Similar reasons like above
            self.train_labels = np.loadtxt(f'{path}/labels.txt', dtype=int)
            self.test_labels = np.loadtxt(f'{path}/labels.txt', dtype=int)

        self.vocab = file_utils.read_text(f'{path}/vocab.txt')

