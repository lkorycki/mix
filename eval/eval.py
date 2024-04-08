import collections
import copy
import os
import traceback

import torch
from typing import Callable, Union
from abc import ABC, abstractmethod

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheTensorDataset, AvalancheConcatDataset
from skmultiflow.drift_detection import ADWIN
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import torch.utils.tensorboard as tb
import tensorflow as tf
import numpy as np

from core.clearn import ContinualLearner, ContinualExtractor, LossTracker, TimeTracker, AvalancheContinualLearner, \
    GenericContinualLearner
from data.stream import Stream, InstanceStream, ClassStream
from eval.tf_writers import TBScalars, TBImages


class Evaluator(ABC):

    @abstractmethod
    def evaluate(self, model_creator: (str, Callable[[], ContinualLearner]), data_creator: (str, Callable[[], Stream])):
        pass


class InstanceStreamEvaluator(Evaluator):

    def __init__(self, batch_size: int, shuffle=False, init_skip_frac=0.05, numpy=False, logdir_root: str='runs'):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.init_skip_frac = init_skip_frac
        self.numpy = numpy
        self.logdir_root = logdir_root

    def evaluate(self, model_creator: (str, Callable[[], ContinualLearner]), data_creator: (str, Callable[[], Stream])):
        model_label, model_creator = model_creator
        stream_label, stream_creator = data_creator

        print('[1/3] Preparing data')
        instance_stream: InstanceStream = stream_creator()
        instance_stream_loader = DataLoader(instance_stream.get_data(), batch_size=self.batch_size, shuffle=self.shuffle)

        print('[2/3] Preparing model')
        model = model_creator()
        # print(model.__dict__)

        init_data = instance_stream.get_init_data()
        n = len(init_data)
        if n > 0:
            print(f'Initializing model with {n} instances')
            init_data_loader = DataLoader(init_data, batch_size=n, shuffle=self.shuffle)
            inputs_batch, labels_batch = next(iter(init_data_loader))
            if self.numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()
            model.initialize(inputs_batch, labels_batch)

        print('[3/3] Preparing metrics')
        per_class_acc = {}
        acc = ADWIN()
        correct = 0.0
        all = 0.0
        init_skip_num = self.init_skip_frac * len(instance_stream)

        logdir = f'{self.logdir_root}/{model_label}'
        tb_writer = tb.SummaryWriter(logdir)
        TBImages.init()

        print('Evaluating...')
        i = 0
        for inputs_batch, labels_batch in tqdm(instance_stream_loader):
            if self.numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()

            i += len(inputs_batch)
            preds = model.predict(inputs_batch)
            model.update(inputs_batch, labels_batch)

            results = [int(int(p) == int(y)) for p, y in zip(preds, labels_batch)]
            correct += sum(results)
            all += len(inputs_batch)

            for r, l in zip(results, labels_batch):
                acc.add_element(float(r))
                l = int(l)

                if l not in per_class_acc:
                    per_class_acc[l] = ADWIN()
                per_class_acc[l].add_element(float(r))

            if i > init_skip_num:
                tb_writer.add_scalar(f'ALL/{stream_label}', acc.estimation, i)

                for c, c_acc in per_class_acc.items():
                    tb_writer.add_scalar(f'{stream_label}/{stream_label}-C{c}', c_acc.estimation, i)


class ClassStreamEvaluator(Evaluator):

    def __init__(self, batch_size: int=256, shuffle: bool=True, num_epochs: int=10, n_eval=1, epoch_eval=True,
                 num_workers: int=0, max_classes: int=-1, full_load: bool=False, numpy=False, confusion_matrix=True,
                 vis=True, emb_vis=False, emb_epoch_vis=False, results_dir: str='results', logdir_root: str='runs'):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.n_eval = n_eval
        self.epoch_eval = epoch_eval
        self.max_classes = max_classes
        self.num_workers = num_workers
        self.full_load = full_load
        self.numpy = numpy
        self.confusion_matrix = confusion_matrix
        self.vis = vis
        self.emb_vis = emb_vis
        self.emb_epoch_vis = emb_epoch_vis
        self.results_dir = results_dir
        self.logdir_root = logdir_root

    def evaluate(self, model_creator: (str, Callable[[], ContinualLearner]), data_creator: (str, Callable[[], Stream])):
        model_label, model_creator = model_creator
        stream_label, stream_creator = data_creator

        print('[1/3] Preparing data')
        class_stream: ClassStream = stream_creator()
        train_class_stream = class_stream.get_train_data()
        test_class_stream = iter(class_stream.get_test_data())

        print('[2/3] Preparing model')
        model: ContinualLearner = model_creator()
        for k, v in model.__dict__.items():
            if k != 'model':
                print(f'{k}={v}')

        init_class_concept_mapping, init_data = class_stream.get_init_data()
        n = len(init_data)
        if n > 0:
            print(f'Initializing model with {n} instances')
            init_data_loader = DataLoader(init_data, batch_size=n, num_workers=self.num_workers, shuffle=self.shuffle)
            inputs_batch, labels_batch = next(iter(init_data_loader))
            labels_batch = Tensor([init_class_concept_mapping[int(cls.item())] for cls in labels_batch])
            if self.numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()
            model.initialize(inputs_batch, labels_batch)

        print('[3/3] Preparing metrics')
        logdir = f'{self.logdir_root}/{model_label}'
        tb_file_writer = tf.summary.create_file_writer(logdir)
        tb_file_writer.set_as_default()
        classes_test_data = {}
        class_test_concept_mapping = {}
        results = collections.defaultdict(list)
        cms = []
        TBImages.init()

        print('Evaluating...')
        try:
            for i, class_batch_data in enumerate(tqdm(train_class_stream)):
                if 0 < self.max_classes == i:
                    break

                (class_idx, class_batch_train_data, class_concept_mapping) = class_batch_data
                (test_class_idx, class_batch_test_data, test_class_concept_mapping) = next(test_class_stream)

                assert class_idx == test_class_idx and class_concept_mapping == test_class_concept_mapping
                class_test_concept_mapping.update(class_concept_mapping)

                classes_test_data[class_idx] = DataLoader(class_batch_test_data, batch_size=self.batch_size,
                                                          num_workers=self.num_workers, shuffle=False)
                if self.vis: TBImages.write_test_data(class_batch_test_data, i, stream_label, class_stream.cls_names)

                time_metrics = {'update': [], 'prediction': []}

                for j in range(self.num_epochs):
                    bs = self.batch_size if not self.full_load else len(class_batch_train_data)
                    train_data_loader = DataLoader(class_batch_train_data, batch_size=bs, num_workers=self.num_workers,
                                                   shuffle=self.shuffle)

                    for inputs_batch, labels_batch in train_data_loader:
                        labels_batch = Tensor([class_concept_mapping[int(cls.item())] for cls in labels_batch])
                        if self.numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()
                        model.update(inputs_batch, labels_batch, epoch=j)

                    if self.epoch_eval:
                        tasks_acc, _ = evaluate_tasks(model, classes_test_data, class_test_concept_mapping, self.numpy)
                        TBScalars.write_epoch_result(tasks_acc, j, stream_label, i)

                        if self.emb_vis and self.emb_epoch_vis and isinstance(model, ContinualExtractor):
                            TBImages.write_embeddings_vis(classes_test_data, model, model_label, stream_label, i, epoch=True, j=j)
                        if isinstance(model, LossTracker):
                            TBImages.write_epoch_loss(model.get_loss(), j, stream_label, i)

                        if isinstance(model, TimeTracker):
                            tm = model.get_time_metrics()
                            if tm is not None:
                                time_metrics['update'].append(tm['update'])
                                time_metrics['prediction'].append(tm['prediction'])
                            model.reset_time_metrics()

                if i == 0 or ((i + 1) % self.n_eval) == 0:
                    tasks_acc, (task_targets, task_preds) = evaluate_tasks(model, classes_test_data, class_test_concept_mapping,
                                                                           self.numpy)

                    for k, task_acc in enumerate(tasks_acc): results[k + 1].append(task_acc)
                    results[0].append(sum(tasks_acc) / len(tasks_acc))

                    TBScalars.write_tasks_results(stream_label, tasks_acc, i)

                    if self.confusion_matrix:
                        cm = TBImages.write_confusion_matrices(task_targets, task_preds, i, stream_label)
                        if (i + 1) % 10 == 0:
                            cms.append(cm)

                    if self.emb_vis and isinstance(model, ContinualExtractor):
                        TBImages.write_embeddings_vis(classes_test_data, model, model_label, stream_label, i)

                    if self.epoch_eval and isinstance(model, TimeTracker):
                        TBScalars.write_time_metrics(time_metrics, stream_label, i)
        except Exception as e:
            print(e)
            traceback.print_exc()
            write_result_to_file(self.results_dir, model_label, stream_label, results, cms, e)
        else:
            write_result_to_file(self.results_dir, model_label, stream_label, results, cms)


class AvalancheStreamEvaluator(Evaluator):

    def __init__(self, test_batch_size: int=256, n_eval=1, num_workers=0, max_classes=-1, numpy=False,
                 confusion_matrix=True, vis=True, emb_vis=False, results_dir='results', logdir_root='runs'):
        self.test_batch_size = test_batch_size
        self.n_eval = n_eval
        self.max_classes = max_classes
        self.num_workers = num_workers
        self.numpy = numpy
        self.confusion_matrix = confusion_matrix
        self.vis = vis
        self.emb_vis = emb_vis
        self.results_dir = results_dir
        self.logdir_root = logdir_root

    def evaluate(self, model_creator: (str, Callable[[], AvalancheContinualLearner]), data_creator: (str, Callable[[], Stream])):
        model_label, model_creator = model_creator
        stream_label, stream_creator = data_creator

        print('[1/3] Preparing data')
        class_stream: ClassStream = stream_creator()
        train_class_stream = class_stream.get_train_data()
        test_class_stream = class_stream.get_test_data()

        origin_dt = train_class_stream[0][1].dataset

        if hasattr(origin_dt, 'cls_map'):
            mp = train_class_stream[0][1].dataset.cls_map
            targets = origin_dt.dataset.targets
            for i in range(len(targets)):
                if targets[i] in mp:
                    targets[i] = mp[targets[i]]

        class_order = [td[1][0][-1].item() if torch.is_tensor(td[1][0][-1]) else td[1][0][-1]
                       for td in train_class_stream]
        train_set = AvalancheConcatDataset([cb[1] for cb in train_class_stream])

        scenario = nc_benchmark(
            train_dataset=train_set,
            test_dataset=AvalancheTensorDataset(torch.Tensor(), torch.Tensor()),
            n_experiences=len(train_class_stream),  # only class-incremental for now
            task_labels=False,
            seed=123,
            shuffle=False,
            fixed_class_order=class_order,
            class_ids_from_zero_from_first_exp=True
        )

        print('[2/3] Preparing model')
        model: AvalancheContinualLearner = model_creator()
        for k, v in model.__dict__.items():
            if k != 'model':
                print(f'{k}={v}')

        print('[3/3] Preparing metrics')
        logdir = f'{self.logdir_root}/{model_label}'
        tb_file_writer = tf.summary.create_file_writer(logdir)
        tb_file_writer.set_as_default()
        classes_test_data = {}
        class_test_concept_mapping = {}
        results = collections.defaultdict(list)
        cms = []
        TBImages.init()

        print('Evaluating...')
        try:
            for i, exp in enumerate(tqdm(scenario.train_stream)):
                if 0 < self.max_classes == i:
                    break

                (train_class_idx, _, train_class_concept_mapping) = train_class_stream[i]
                (test_class_idx, class_batch_test_data, test_class_concept_mapping) = test_class_stream[i]

                assert train_class_idx == test_class_idx and train_class_concept_mapping == test_class_concept_mapping
                class_test_concept_mapping.update(train_class_concept_mapping)

                classes_test_data[train_class_idx] = DataLoader(class_batch_test_data, batch_size=self.test_batch_size,
                                                                num_workers=self.num_workers, shuffle=False)
                if self.vis: TBImages.write_test_data(class_batch_test_data, i, stream_label, class_stream.cls_names)

                time_metrics = {'update': [], 'prediction': []}

                model.update(exp)

                if isinstance(model, TimeTracker):
                    tm = model.get_time_metrics()
                    if tm is not None:
                        time_metrics['update'].append(tm['update'])
                        time_metrics['prediction'].append(tm['prediction'])
                    model.reset_time_metrics()

                if i == 0 or ((i + 1) % self.n_eval) == 0:
                    tasks_acc, (task_targets, task_preds) = evaluate_tasks(model, classes_test_data,
                                                                           class_test_concept_mapping,
                                                                           self.numpy)

                    for k, task_acc in enumerate(tasks_acc): results[k + 1].append(task_acc)
                    results[0].append(sum(tasks_acc) / len(tasks_acc))

                    TBScalars.write_tasks_results(stream_label, tasks_acc, i)

                    if self.confusion_matrix:
                        cm = TBImages.write_confusion_matrices(task_targets, task_preds, i, stream_label)
                        if (i + 1) % 10 == 0:
                            cms.append(cm)

                    if self.emb_vis and isinstance(model, ContinualExtractor):
                        TBImages.write_embeddings_vis(classes_test_data, model, model_label, stream_label, i)

                    if isinstance(model, TimeTracker):
                        TBScalars.write_time_metrics(time_metrics, stream_label, i)

        except Exception as e:
            print(e)
            traceback.print_exc()
            write_result_to_file(self.results_dir, model_label, stream_label, results, cms, e)
        else:
            write_result_to_file(self.results_dir, model_label, stream_label, results, cms)


class OfflineClassStreamEvaluator(Evaluator):

    def __init__(self, batch_size, num_epochs, n_eval=1, num_workers=0, epoch_eval=True, numpy=False,
                 confusion_matrix=True, vis=False, results_dir='results', logdir_root='runs',
                 model_path: str=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.n_eval = n_eval
        self.num_workers = num_workers
        self.epoch_eval = epoch_eval
        self.numpy = numpy
        self.confusion_matrix = confusion_matrix
        self.vis = vis
        self.logdir_root = logdir_root
        self.results_dir = results_dir
        self.model_path = model_path

    def evaluate(self, model_creator: (str, Callable[[], ContinualLearner]), data_creator: (str, Callable[[], Stream])):
        model_label, model_creator = model_creator
        stream_label, stream_creator = data_creator
        model = None

        print('[1/2] Preparing data')
        class_stream: ClassStream = stream_creator()
        train_class_stream = class_stream.get_train_data()
        test_class_stream = iter(class_stream.get_test_data())

        print('[2/2] Preparing metrics')
        logdir = f'{self.logdir_root}/{model_label}'
        tb_file_writer = tf.summary.create_file_writer(logdir)
        tb_file_writer.set_as_default()
        all_train_data = None
        classes_test_data = {}
        class_test_concept_mapping = {}
        results = collections.defaultdict(list)
        cms = []
        TBImages.init()

        print('Evaluating...')
        for i, class_batch_data in enumerate(tqdm(train_class_stream)):
            (class_idx, class_batch_train_data, class_concept_mapping) = class_batch_data
            (test_class_idx, class_batch_test_data, test_class_concept_mapping) = next(test_class_stream)

            assert class_idx == test_class_idx and class_concept_mapping == test_class_concept_mapping
            class_test_concept_mapping.update(class_concept_mapping)

            all_train_data = all_train_data + class_batch_train_data if all_train_data is not None else class_batch_train_data
            all_train_data_loader = DataLoader(all_train_data, batch_size=self.batch_size, num_workers=self.num_workers,
                                               shuffle=True)

            classes_test_data[class_idx] = DataLoader(class_batch_test_data, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=True)
            if self.vis: TBImages.write_test_data(class_batch_test_data, i, stream_label, class_stream.cls_names)

            model = model_creator()

            for j in tqdm(range(self.num_epochs)):
                for inputs_batch, labels_batch in all_train_data_loader:
                    labels_batch = Tensor([class_test_concept_mapping[int(cls.item())] for cls in labels_batch])
                    if self.numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()
                    model.update(inputs_batch, labels_batch)

                if hasattr(model, 'scheduler') and model.scheduler is not None:
                    model.scheduler.step()

                if self.epoch_eval:
                    tasks_acc, _ = evaluate_tasks(model, classes_test_data, class_test_concept_mapping, self.numpy)
                    TBScalars.write_epoch_result(tasks_acc, j, stream_label, i)

            if (i == 0 or (i + 1) % self.n_eval) == 0:
                tasks_acc, (task_targets, task_preds) = evaluate_tasks(model, classes_test_data, class_test_concept_mapping,
                                                                       self.numpy)

                for k, task_acc in enumerate(tasks_acc): results[k + 1].append(task_acc)
                results[0].append(sum(tasks_acc) / len(tasks_acc))

                TBScalars.write_tasks_results(stream_label, tasks_acc, i)

                if self.confusion_matrix:
                    cm = TBImages.write_confusion_matrices(task_targets, task_preds, i, stream_label)
                    if (i + 1) % 10 == 0:
                        cms.append(cm)

        write_result_to_file(self.results_dir, model_label, stream_label, results, cms)

        if self.model_path:
            print(f'Saving model: {self.model_path}')
            torch.save(model.get_net().state_dict(), self.model_path)


def evaluate_tasks(model: GenericContinualLearner, classes_test_data,
                   class_test_concept_mapping, numpy):
    classes_acc, class_targets, class_preds = [], [], []

    for j, class_test_data in classes_test_data.items():
        correct, all = 0.0, 0.0

        for inputs_batch, labels_batch in class_test_data:
            labels_batch = Tensor([class_test_concept_mapping[int(cls.item())] for cls in labels_batch.long()])
            if numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()

            preds_batch = model.predict(inputs_batch)
            results = [p == y for p, y in zip(preds_batch, labels_batch)]
            correct += sum(results)
            all += len(inputs_batch)

            class_targets += list(labels_batch)
            class_preds += list(preds_batch)

        acc = correct / all
        classes_acc.append(acc)  # todo: add per subclass

    return classes_acc, (class_targets, class_preds)


def write_result_to_file(results_dir, model_label, stream_label, results, cms, error=None):
    os.makedirs(results_dir, exist_ok=True)
    stamp = '_error' if error is not None else ''
    path = f'{results_dir}/{model_label}#{stream_label}{stamp}.csv'
    f = open(path, 'w')

    print('Writing results to file ', path)
    if error is not None:
        f.write(f'Evaluation failed: {error}')
    else:
        num_tasks = len(results[0])
        for task_id, values in results.items():
            ext = [0.0] * (num_tasks - len(values))
            values = ext + values
            values = [str(f.item() if torch.is_tensor(f) else str(f)) for f in values]
            vals = ','.join(values)
            f.write(f'{task_id},{vals}\n')

        if len(cms) > 0:
            os.makedirs(f'{results_dir}/cms', exist_ok=True)
            path = f'{results_dir}/cms/{model_label}#{stream_label}_cms.npy'
            print(f'Writing {len(cms)} confusion matrices to file ', path)
            np.save(path, np.array(cms, dtype=object))

    f.close()

