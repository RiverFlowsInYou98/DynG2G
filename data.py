import pandas as pd
import torch
import numpy as np
import scipy.sparse
import tarfile
from utils import *


def is_compatible(filename):
    return any(filename.endswith(extension) for extension in [".txt"])


def cluster_negs_and_positives(ratings):
    pos_indices = ratings > 0
    neg_indices = ratings <= 0
    ratings[pos_indices] = 1
    ratings[neg_indices] = -1
    return ratings


class Namespace(object):
    """
    helps referencing object in a dictionary as dict.key instead of dict['key']
    """

    def __init__(self, adict):
        self.__dict__.update(adict)


def aggregate_by_time(time_vector, time_win_aggr):
    time_vector = time_vector - time_vector.min()
    time_vector = torch.div(time_vector, time_win_aggr, rounding_mode="floor")
    return time_vector


def load_data_from_tar(
    file,
    tar_archive,
    replace_unknow=False,
    starting_line=1,
    sep=",",
    type_fn=float,
    tensor_const=torch.DoubleTensor,
):
    f = tar_archive.extractfile(file)
    lines = f.read()
    lines = lines.decode("utf-8")
    if replace_unknow:
        lines = lines.replace("unknow", "-1")
        lines = lines.replace("-1n", "-1")
    lines = lines.splitlines()
    data = [[type_fn(r) for r in row.split(sep)] for row in lines[starting_line:]]
    data = tensor_const(data)
    return data


class Dataset_SBM(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.A_list = []
        self.X_list = []

        dataframe = pd.read_csv(file_path)
        max_size = 0
        for t in range(max(dataframe.loc[:, "time"]) + 1):
            print("loading graph at time stamp %d" % (t))
            edges_t = dataframe.loc[
                dataframe["time"] == t, ["source", "target"]
            ].to_numpy()
            A, X, size = self.get_graph(edges_t, max_size)
            if size > max_size:
                max_size = size
            self.A_list.append(A)
            self.X_list.append(X)
        print(
            "loading finished! The dynamic graph has %d time stamps." % (self.__len__())
        )

    def __len__(self):
        return len(self.A_list)

    def __getitem__(self, idx):
        return self.A_list[idx], self.X_list[idx]

    def get_graph(self, edges, max_size):
        largest_index = np.max(edges)
        smallest_index = 0
        num_nodes = largest_index - smallest_index + 1
        size = np.maximum(max_size, num_nodes)
        A = np.zeros((size, size))
        for edge in edges:
            A[int(edge[0] - smallest_index), int(edge[1] - smallest_index)] = 1
        A[range(len(A)), range(len(A))] = 0
        A = scipy.sparse.csr_matrix(A)
        X = A + scipy.sparse.eye(A.shape[0])
        X = ScipySparse2TorchSparse(X)
        return A, X, size


class Dataset_UCI(torch.utils.data.Dataset):
    def __init__(self, file_tuple):
        self.A_list = []
        self.X_list = []
        tar_archive = tarfile.open(file_tuple[0], "r:bz2")
        data = load_data_from_tar(file_tuple[1], tar_archive, starting_line=2, sep=" ")

        cols = Namespace({"source": 0, "target": 1, "weight": 2, "time": 3})
        data = data.long()
        data[:, [cols.source, cols.target]] -= 1
        data = torch.cat(
            [data, data[:, [cols.target, cols.source, cols.weight, cols.time]]], dim=0
        )
        data[:, cols.time] = aggregate_by_time(data[:, cols.time], 190080)
        idx = data[:, [cols.source, cols.target, cols.time]]
        df = pd.DataFrame(idx.numpy())
        df.columns = ["source", "target", "time"]

        max_size = 0
        for t in range(df["time"].max() + 1):
            print("loading graph at time stamp %d" % (t))
            df1 = df[df["time"] == t]
            edges_t = df1[["source", "target"]].to_numpy()
            A, X, size = self.get_graph(edges_t, max_size)
            if size > max_size:
                max_size = size
            if A.nnz < 10 or A.shape[0] < 10:
                print("time stamp %d discarded" % (t))
                continue
            self.A_list.append(A)
            self.X_list.append(X)
        print(
            "loading finished! The dynamic graph has %d time stamps." % (self.__len__())
        )

    def __len__(self):
        return len(self.A_list)

    def __getitem__(self, idx):
        return self.A_list[idx], self.X_list[idx]

    def get_graph(self, edges, max_size):
        largest_index = np.max(edges)
        smallest_index = 0
        num_nodes = largest_index - smallest_index + 1
        size = np.maximum(max_size, num_nodes)
        A = np.zeros((size, size))
        for edge in edges:
            A[int(edge[0] - smallest_index), int(edge[1] - smallest_index)] = 1
        A[range(len(A)), range(len(A))] = 0
        A = scipy.sparse.csr_matrix(A)
        X = A + scipy.sparse.eye(A.shape[0])
        X = ScipySparse2TorchSparse(X)
        return A, X, size


class Dataset_AS(torch.utils.data.Dataset):
    def __init__(self, dir_path):
        self.filename_list = [
            os.path.join(dir_path, x) for x in os.listdir(dir_path) if is_compatible(x)
        ]
        self.filename_list.sort()
        self.filename_list = self.filename_list[0:100]
        self.A_list = []
        self.X_list = []
        max_size = 0
        for t in range(len(self.filename_list)):
            print("loading graph at time stamp %d" % (t))
            filename = self.filename_list[t]
            graph_table = pd.read_table(filename, skiprows=3)
            edges_t = graph_table[["# FromNodeId", "ToNodeId"]].to_numpy()
            A, X, size = self.get_graph(edges_t, max_size)
            if size > max_size:
                max_size = size

            self.A_list.append(A)
            self.X_list.append(X)
        print(
            "loading finished! The dynamic graph has %d time stamps." % (self.__len__())
        )

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        return self.A_list[idx], self.X_list[idx]

    def get_graph(self, edges, max_size):
        largest_index = np.max(edges)
        smallest_index = 1
        num_nodes = largest_index - smallest_index + 1
        size = np.maximum(max_size, num_nodes)
        A = np.zeros((size, size))
        for edge in edges:
            A[int(edge[0] - smallest_index), int(edge[1] - smallest_index)] = 1
        A[range(len(A)), range(len(A))] = 0
        A = scipy.sparse.csr_matrix(A)
        X = A + scipy.sparse.eye(A.shape[0])
        X = ScipySparse2TorchSparse(X)
        return A, X, size


class Dataset_BitCoin(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.A_list = []
        self.X_list = []

        graph_table = pd.read_csv(
            "datasets/soc-sign-bitcoinotc.csv",
            names=["FromNodeId", "ToNodeId", "Weight", "TimeStep"],
        )
        graph_tensor = torch.tensor(graph_table.values, dtype=torch.long)
        ecols = Namespace({"FromNodeId": 0, "ToNodeId": 1, "Weight": 2, "TimeStep": 3})

        edges = graph_tensor[:, [ecols.FromNodeId, ecols.ToNodeId]]
        _, edges = edges.unique(return_inverse=True)
        graph_tensor[:, [ecols.FromNodeId, ecols.ToNodeId]] = edges
        timesteps = aggregate_by_time(graph_tensor[:, ecols.TimeStep], 1200000)
        graph_tensor[:, ecols.TimeStep] = timesteps

        graph_tensor[:, ecols.Weight] = cluster_negs_and_positives(
            graph_tensor[:, ecols.Weight]
        )

        # add the reversed link to make the graph undirected
        # graph_tensor = torch.cat(
        #     [
        #         graph_tensor,
        #         graph_tensor[
        #             :, [ecols.ToNodeId, ecols.FromNodeId, ecols.Weight, ecols.TimeStep]
        #         ],
        #     ]
        # )

        df = pd.DataFrame(graph_tensor.numpy())
        df.columns = ["source", "target", "weight", "time"]

        max_size = 0
        for t in range(df["time"].max() + 1):
            print("loading graph at time stamp %d" % (t))
            df1 = df[df["time"] == t]
            edges_t = df1[["source", "target"]].to_numpy()
            A, X, size = self.get_graph(edges_t, max_size)
            if size > max_size:
                max_size = size
            if A.nnz < 15 or A.shape[0] < 10:
                print("time stamp %d discarded" % (t))
                continue
            self.A_list.append(A)
            self.X_list.append(X)
        print(
            "loading finished! The dynamic graph has %d time stamps." % (self.__len__())
        )

    def __len__(self):
        return len(self.A_list)

    def __getitem__(self, idx):
        return self.A_list[idx], self.X_list[idx]

    def get_graph(self, edges, max_size):
        largest_index = np.max(edges)
        smallest_index = 0
        num_nodes = largest_index - smallest_index + 1
        size = np.maximum(max_size, num_nodes)
        A = np.zeros((size, size))
        for edge in edges:
            A[int(edge[0] - smallest_index), int(edge[1] - smallest_index)] = 1
        A[range(len(A)), range(len(A))] = 0
        A = scipy.sparse.csr_matrix(A)
        X = A + scipy.sparse.eye(A.shape[0])
        X = ScipySparse2TorchSparse(X)
        return A, X, size


class Dataset_RM(torch.utils.data.Dataset):
    def __init__(self, file_tuple):
        self.A_list = []
        self.X_list = []
        tar_archive = tarfile.open(file_tuple[0], "r:bz2")
        data = load_data_from_tar(file_tuple[1], tar_archive, starting_line=2, sep=None)

        cols = Namespace({"source": 0, "target": 1, "weight": 2, "time": 3})
        data = data.long()
        # first id should be 0 (they are already contiguous)
        data[:, [cols.source, cols.target]] -= 1

        # add edges in the other direction (symmetric)
        # data = torch.cat(
        #     [data, data[:, [cols.target, cols.source, cols.weight, cols.time]]], dim=0
        # )

        data[:, cols.time] = aggregate_by_time(data[:, cols.time], 222400)
        idx = data[:, [cols.source, cols.target, cols.time]]
        df = pd.DataFrame(idx.numpy())
        df.columns = ["source", "target", "time"]

        max_size = 0
        for t in range(df["time"].max() + 1):
            print("loading graph at time stamp %d" % (t))
            df1 = df[df["time"] == t]
            edges_t = df1[["source", "target"]].to_numpy()
            A, X, size = self.get_graph(edges_t, max_size)
            if size > max_size:
                max_size = size
            if A.nnz < 15 or A.shape[0] < 15:
                print("time stamp %d discarded" % (t))
                continue
            self.A_list.append(A)
            self.X_list.append(X)
        print(
            "loading finished! The dynamic graph has %d time stamps." % (self.__len__())
        )

    def __len__(self):
        return len(self.A_list)

    def __getitem__(self, idx):

        return self.A_list[idx], self.X_list[idx]

    def get_graph(self, edges, max_size):
        largest_index = np.max(edges)
        smallest_index = 0
        num_nodes = largest_index - smallest_index + 1
        size = np.maximum(max_size, num_nodes)
        A = np.zeros((size, size))
        for edge in edges:
            A[int(edge[0] - smallest_index), int(edge[1] - smallest_index)] = 1
        A[range(len(A)), range(len(A))] = 0
        A = scipy.sparse.csr_matrix(A)
        X = A + scipy.sparse.eye(A.shape[0])
        X = ScipySparse2TorchSparse(X)
        return A, X, size
