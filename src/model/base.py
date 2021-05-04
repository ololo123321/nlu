import random
import os
import json
import shutil
from typing import Dict, List, Callable, Tuple, Iterable
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from src.data.base import Example
from src.utils import train_test_split


class ModeKeys:
    TRAIN = "train"  # need labels, dropout on
    VALID = "valid"  # need labels, dropout off
    TEST = "test"  # don't need labels, dropout off


class BaseModel(ABC):
    """
    Interface for all models

    config = {
        "model": {
            "embedder": {
                ...
            },
            "ner": {
                "loss_coef": 0.0,
                ...
            },
            "re": {
                "loss_coef": 0.0,
                ...
            }
        },
        "training": {
            "num_epochs": 100,
            "batch_size": 16,
            "max_epochs_wo_improvement": 10
        },
        "inference": {
            "window": 1,
            "max_tokens_per_batch": 10000
        },
        "optimizer": {
            "init_lr": 2e-5,
            "num_train_steps": 100000,
            "num_warmup_steps": 10000
        }
    }
    """

    model_scope = "model"

    def __init__(self, sess: tf.Session = None, config: Dict = None):
        self.sess = sess
        self.config = config

        self.loss = None
        self.train_op = None
        self.training_ph = None

    # специфичные для каждой модели методы

    @abstractmethod
    def _build_graph(self):
        """построение вычислительного графа (без loss и train_op)"""

    @abstractmethod
    def _build_embedder(self):
        """вход - токены, выход - векторизованные токены"""

    @abstractmethod
    def _get_feed_dict(self, examples: List[Example], mode: str) -> Dict:
        """mode: {train, valid, test} (см. ModeKeys)"""

    @abstractmethod
    def _set_placeholders(self):
        pass

    @abstractmethod
    def _set_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def _set_train_op(self):
        pass

    @abstractmethod
    def predict(self, examples: List[Example], **kwargs) -> None:
        """
        Вся логика инференса должна быть реализована здесь.
        Предполагается, что модель училась не на целых документах, а на кусках (chunks).
        Следовательно, предикт модель должна делать тоже на уровне chunks.
        Но в конечном итоге нас интересуют предсказанные лейблы на исходных документах (examples).
        Поэтому схема такая:
        1. получить модельные предикты на уровне chunks
        2. аггрегировать результат из п.1 и записать на уровне examples

        :param examples: исходные документы. атрибут chunks должен быть заполнен!
        :param kwargs:
        :return:
        """

    @abstractmethod
    def evaluate(self, examples: List[Example], **kwargs) -> Dict:
        """
        Возвращаемый словарь должен обязательно содержать ключи "score" и "loss"
        :param examples: исходные документы. атрибут chunks должен быть заполнен!
        :return:
        """

    @abstractmethod
    def load_encoders(self, model_dir: str):
        """
        Подгрузка маппингов "label -> code"
        :param model_dir:
        :return:
        """

    @abstractmethod
    def save_encoders(self, model_dir: str):
        """
        Сохранение маппингов "label -> code"
        :param model_dir:
        :return:
        """

    # общие методы для всех моделей

    def build(self):
        self._set_placeholders()
        with tf.variable_scope(self.model_scope):
            self._build_graph()
            self._set_loss()
            self._set_train_op()
        self.reset_weights()

    # альтернативная версия данной функции вынесена в src._old.wip
    def train(
            self,
            examples_train: List[Example],
            examples_valid: List[Example],
            train_op_name: str = "train_op",
            model_dir: str = None,
            scope_to_save: str = None,
            verbose: bool = True,
            verbose_fn: Callable = None,
    ):
        """

        :param examples_train:
        :param examples_valid:
        :param train_op_name:
        :param model_dir:
        :param scope_to_save:
        :param verbose:
        :param verbose_fn: вход - словарь с метриками (выход self.evaluate); выход - None. функция должна вывести
                           релевантные метрики в stdout
        :return:

        нет возможности переопределять batch_size, потому что есть следующая зависимость:
        batch_size -> num_train_steps -> lr schedule for adamw
        поэтому если хочется изменить batch_size, то нужно переопределить train_op. иными словами, проще сделать так:
        tf.reset_default_graph()
        sess = tf.Session()
        model = ...
        model.build()
        model.initialize()
        """
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
            checkpoint_path = os.path.join(model_dir, "model.ckpt")
            print(f"model dir {model_dir} created")
        else:
            checkpoint_path = None
            print("model dir is None, so checkpoints will not be saved")

        if checkpoint_path is not None:
            if scope_to_save is not None:
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_to_save)
            else:
                var_list = tf.trainable_variables()
            saver = tf.train.Saver(var_list)
        else:
            saver = None

        # релзиовано так для возможности выбора train_op: обычная или с аккумуляцией градиентов.
        # TODO: реализовать вторую (уже реализовывал, когда решал dependency parsing, нужно скопипастить сюда)
        train_op = getattr(self, train_op_name)

        chunks_train = []
        for x in examples_train:
            assert len(x.chunks) > 0, f"[{x.id}] example didn't split by chunks!"
            chunks_train += x.chunks

        chunks_valid = []
        for x in examples_valid:
            assert len(x.chunks) > 0, f"[{x.id}] example didn't split by chunks!"
            chunks_valid += x.chunks

        batch_size = self.config["training"]["batch_size"]
        num_epoch_steps = int(len(chunks_train) / batch_size)
        best_score = -1
        num_steps_wo_improvement = 0
        verbose_fn = verbose_fn if verbose_fn is not None else print
        train_loss = []

        for epoch in range(self.config["training"]["num_epochs"]):
            for _ in range(num_epoch_steps):
                if len(chunks_train) > batch_size:
                    chunks_batch = random.sample(chunks_train, batch_size)
                else:
                    chunks_batch = chunks_train
                feed_dict = self._get_feed_dict(chunks_batch, mode=ModeKeys.TRAIN)
                try:
                    _, loss = self.sess.run([train_op, self.loss], feed_dict=feed_dict)
                    train_loss.append(loss)
                except Exception as e:
                    print("current batch:", [x.id for x in chunks_batch])
                    raise e

            # pycharm bug:
            # Cannot find reference {mean, std} in __init__.pyi | __init__.pxd
            # so, np.mean(train_loss) highlights yellow
            print(f"epoch {epoch} finished. mean train loss: {np.array(train_loss).mean()}. evaluation starts.")
            performance_info = self.evaluate(examples=examples_valid, batch_size=batch_size)
            if verbose:
                verbose_fn(performance_info)
            score = performance_info["score"]

            print("current score:", score)

            if score > best_score:
                print("!!! new best score:", score)
                best_score = score
                num_steps_wo_improvement = 0

                if saver is not None:
                    saver.save(self.sess, checkpoint_path)
                    print(f"saved new head to {checkpoint_path}")
            else:
                num_steps_wo_improvement += 1
                print("best score:", best_score)
                print("steps wo improvement:", num_steps_wo_improvement)

                if num_steps_wo_improvement == self.config["training"]["max_epochs_wo_improvement"]:
                    print("training finished due to max number of steps wo improvement encountered.")
                    break

            print("=" * 50)

        if saver is not None:
            print(f"restoring model from {checkpoint_path}")
            saver.restore(self.sess, checkpoint_path)

    def cross_validate(
            self,
            examples: List[Example],
            folds: Iterable,
            valid_frac: float = 0.15,
            verbose_fn: Callable = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param examples:
        :param folds:
        :param valid_frac:
        :param verbose_fn:
        :return:
        """
        for x in examples:
            assert len(x.chunks) > 0, f"[{x.id}] example didn't split by chunks!"

        scores_valid = []
        scores_test = []

        verbose_fn = verbose_fn if verbose_fn is not None else print

        for i, (train_files, test_files) in enumerate(folds):
            print(f"FOLDS {i}")

            train_files_set = set(train_files)
            test_files_set = set(test_files)

            examples_train_valid = [x for x in examples if x.filename in train_files_set]
            examples_test = [x for x in examples if x.filename in test_files_set]

            train_frac = 1.0 - valid_frac
            examples_train, examples_valid = train_test_split(
                examples_train_valid, train_frac=train_frac, seed=228
            )

            # TODO: lr schedule depends on num train steps, which depends on num train sample and batch size.

            self.reset_weights()

            self.train(
                examples_train=examples_train,
                examples_valid=examples_valid,
                train_op_name="train_op",
                model_dir=None,
                scope_to_save=None,
                verbose=False,
                verbose_fn=verbose_fn
            )

            # TODO: batch_size для инференса вынести в config
            d_valid = self.evaluate(examples=examples_valid, batch_size=16)
            verbose_fn(d_valid)
            d_test = self.evaluate(examples=examples_test, batch_size=16)
            verbose_fn(d_test)

            scores_valid.append(d_valid["re"]["metrics"]["micro"]["f1"])
            scores_test.append(d_test["re"]["metrics"]["micro"]["f1"])

            print("=" * 80)

        # pycharm bug:
        # Cannot find reference {mean, std} in __init__.pyi | __init__.pxd
        # so, np.mean(scores) highlights yellow
        scores_valid = np.array(scores_valid)
        scores_test = np.array(scores_test)

        print(f"scores valid: {scores_valid} (mean {scores_valid.mean()}, std {scores_valid.std()})")
        print(f"scores test: {scores_test} (mean {scores_test.mean()}, std {scores_test.std()})")

        return scores_valid, scores_test

    def save(self, model_dir: str, force: bool = True, scope_to_save: str = None):
        assert self.config is not None
        assert self.sess is not None

        if force:
            if os.path.isdir(model_dir):
                shutil.rmtree(model_dir)
        else:
            assert not os.path.isdir(model_dir), \
                f'dir {model_dir} already exists. exception raised due to flag "force" was set to "False"'

        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

        self.save_weights(model_dir=model_dir, scope=scope_to_save)
        self.save_encoders(model_dir=model_dir)

    @classmethod
    def load(cls, sess: tf.Session, model_dir: str, scope_to_load: str = None):

        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)

        model = cls(sess=sess, config=config)
        model.build()
        model.restore_weights(model_dir=model_dir, scope=scope_to_load)
        model.load_encoders(model_dir=model_dir)
        return model

    def save_weights(self, model_dir: str,  scope: str = None):
        self._save_or_restore(model_dir=model_dir, save=True, scope=scope)

    def restore_weights(self, model_dir: str,  scope: str = None):
        self._save_or_restore(model_dir=model_dir, save=False, scope=scope)

    def reset_weights(self, scope: str = None):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        init_op = tf.variables_initializer(variables)
        self.sess.run(init_op)

    def _save_or_restore(self, model_dir: str, save: bool, scope: str = None):
        scope = scope if scope is not None else self.model_scope
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list)
        checkpoint_path = os.path.join(model_dir, "model.ckpt")
        if save:
            saver.save(self.sess, checkpoint_path)
        else:
            saver.restore(self.sess, checkpoint_path)
