import itertools
from typing import Callable

import numpy as np


class CollectionUtils:

    @staticmethod
    def ensure_arr_size(arr: np.ndarray, y: int):
        if len(arr.shape) == 1:
            return np.hstack((arr, np.zeros(y - len(arr) + 1))) if len(arr) - 1 < y else arr
        else:
            return np.hstack((arr, np.zeros((arr.shape[0], y - arr.shape[1] + 1, arr.shape[2])))) if arr.shape[1] - 1 < y else arr

    @staticmethod
    def ensure_list2d_size(lst: list, y: int, element_creator: Callable[[], any]):
        if len(lst[0]) > y:
            return lst

        for l in lst:
            l_len = len(l) - 1
            for _ in range(y - l_len):
                l.append(element_creator())

        return lst

    @staticmethod
    def split_list(lst, chunk_size: int):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    @staticmethod
    def flatten_list(lst):
        return list(itertools.chain.from_iterable(lst))

    @staticmethod
    def gen_static_seq(num_classes, num_superclasses):
        seq, s = [], 0

        for i in range(num_classes):
            seq.append((s, [i], {i: i % num_superclasses}))
            s += 1

        return seq

    @staticmethod
    def gen_drift_seq(seq_len, num_classes, num_superclasses, stat_n, drift_n):
        assert stat_n >= drift_n
        seq, q = [], []
        s, sn, dn, c = 0, 1, 1, 0

        while s < seq_len:
            if sn % (stat_n + 1):
                sc = c % num_classes
                seq.append((s, [sc], {sc: sc % num_superclasses}))
                q.append((sc, sc % num_superclasses))
                c += 1
                sn += 1
            elif dn % (drift_n + 1):
                dc, oc = q.pop(0)
                seq.append((s, [dc], {dc: (oc + 1) % num_superclasses}))
                dn += 1

            if not (sn % (stat_n + 1) or dn % (drift_n + 1)):
                sn, dn = 1, 1

            s += 1

        return seq
