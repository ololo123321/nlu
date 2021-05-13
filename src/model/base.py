import random
import os
import json
import math
from typing import Dict, List, Callable, Tuple, Iterable
from abc import ABC, abstractmethod
from collections import namedtuple

import tensorflow as tf
import numpy as np
import tqdm
from bert.modeling import BertModel, BertConfig
from bert.optimization import create_optimizer

from src.data.base import Example
from src.utils import train_test_split, get_filtered_by_length_chunks, log
from src.model.layers import StackedBiRNN


class ModeKeys:
    TRAIN = "train"  # need labels, dropout on
    VALID = "valid"  # need labels, dropout off
    TEST = "test"  # don't need labels, dropout off


BertInputs = namedtuple(
    "BertInputs",
    ["input_ids", "input_mask", "segment_ids", "first_pieces_coords", "num_tokens", "num_pieces"]
)


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
    def _set_layers(self):
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

    @property
    @abstractmethod
    def _is_bpe_level(self) -> bool:
        """
        используется ли bpe-токенизация. нужно для корректной фильтрации примеров по длине при обучении/инференсе.
        реализовано так, потому что надо прокидывать значение в функции train, которая общая для всех моделей.
        TODO: подумать, мб можно сделать лучше
        """

    # общие методы для всех моделей

    def build(self, mode: str = ModeKeys.TRAIN):
        self._set_placeholders()
        with tf.variable_scope(self.model_scope):
            self._set_layers()
            self._build_graph()
            if mode != ModeKeys.TEST:
                self._set_loss()
            if mode == ModeKeys.TRAIN:
                self._set_train_op()

    # альтернативная версия данной функции вынесена в src._old.wip
    # TODO: мб объекты группировать в батчи по числу элементарных объектов, на которых считается loss?
    @log
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
            if os.path.isdir(model_dir):
                print(f"model dir {model_dir} exists")
            else:
                os.makedirs(model_dir, exist_ok=True)
                print(f"model dir {model_dir} created")
            checkpoint_path = os.path.join(model_dir, "model.ckpt")
            print(f"checkpoint path: {checkpoint_path}")

            if scope_to_save is not None:
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_to_save)
            else:
                var_list = tf.trainable_variables()
            saver = tf.train.Saver(var_list)
        else:
            checkpoint_path = None
            saver = None
            print("model dir is None, so checkpoints will not be saved")

        # релзиовано так для возможности выбора train_op:
        # 1) обнолвять все веса
        # 2) обновлять только веса головы
        # *3) аккумуляция градиентов
        # TODO: реализовать последнюю (уже реализовывал, когда решал dependency parsing, нужно скопипастить сюда)
        train_op = getattr(self, train_op_name)

        maxlen = self.config["training"]["maxlen"]
        chunks_train = get_filtered_by_length_chunks(
            examples=examples_train, maxlen=maxlen, pieces_level=self._is_bpe_level
        )

        batch_size = self.config["training"]["batch_size"]
        num_epoch_steps = math.ceil(len(chunks_train) / batch_size)
        best_score = -1
        num_steps_wo_improvement = 0
        verbose_fn = verbose_fn if verbose_fn is not None else print
        train_loss = []

        for epoch in range(self.config["training"]["num_epochs"]):
            for _ in tqdm.trange(num_epoch_steps):
                if len(chunks_train) > batch_size:
                    chunks_batch = random.sample(chunks_train, batch_size)
                else:
                    chunks_batch = chunks_train
                feed_dict = self._get_feed_dict(chunks_batch, mode=ModeKeys.TRAIN)
                try:
                    _, loss = self.sess.run([train_op, self.loss], feed_dict=feed_dict)
                    assert not np.isnan(loss), "loss becomes nan"
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

    @log
    def cross_validate(
            self,
            examples: List[Example],
            folds: Iterable,
            valid_frac: float = 0.15,
            model_dir: str = None,
            verbose: bool = False,
            verbose_fn: Callable = None,
            **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param examples:
        :param folds:
        :param valid_frac:
        :param model_dir:
        :param verbose:
        :param verbose_fn:
        :return:
        """
        assert self.sess is None

        # for x in examples:
        #     assert len(x.chunks) > 0, f"[{x.id}] example didn't split by chunks!"

        scores_valid = []
        scores_test = []

        verbose_fn = verbose_fn if verbose_fn is not None else print

        if model_dir is None:
            print("[WARNING] model dir is not set => evaluation on test data will be done on last weights, "
                  "which might differ from best ones")

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

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

            print("num train examples:", len(examples_train))
            print("num valid examples:", len(examples_valid))
            print("num test examples:", len(examples_test))

            # TODO: lr schedule depends on num train steps, which depends on num train samples and batch size.

            with tf.Session(config=sess_config) as sess:
                self.sess = sess
                self.reset_weights(**kwargs)
                self.train(
                    examples_train=examples_train,
                    examples_valid=examples_valid,
                    train_op_name="train_op",
                    model_dir=model_dir,
                    scope_to_save=None,
                    verbose=verbose,
                    verbose_fn=verbose_fn
                )
                print("best valid scores:")
                d_valid = self.evaluate(examples=examples_valid)
                verbose_fn(d_valid)
                print("\ntest scores:")
                d_test = self.evaluate(examples=examples_test)
                verbose_fn(d_test)

            scores_valid.append(d_valid["score"])
            scores_test.append(d_test["score"])

            print("=" * 80)

        # pycharm bug:
        # Cannot find reference {mean, std} in __init__.pyi | __init__.pxd
        # so, np.mean(scores) highlights yellow
        scores_valid = np.array(scores_valid)
        scores_test = np.array(scores_test)

        print(f"scores valid: {scores_valid} (mean {scores_valid.mean()}, std {scores_valid.std()})")
        print(f"scores test: {scores_test} (mean {scores_test.mean()}, std {scores_test.std()})")

        return scores_valid, scores_test

    def save_config(self, model_dir: str):
        assert self.config is not None
        assert os.path.isdir(model_dir)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

    @classmethod
    def load(cls, sess: tf.Session, model_dir: str, scope_to_load: str = None, mode: str = ModeKeys.TEST):

        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)

        model = cls(sess=sess, config=config)
        model.build(mode=mode)
        model.restore_weights(model_dir=model_dir, scope=scope_to_load)
        return model

    def save_weights(self, model_dir: str,  scope: str = None):
        self._save_or_restore(model_dir=model_dir, save=True, scope=scope)

    def restore_weights(self, model_dir: str,  scope: str = None):
        self._save_or_restore(model_dir=model_dir, save=False, scope=scope)

    def reset_weights(self, scope: str = None, **kwargs):
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


class BaseModelBert(BaseModel):
    def __init__(self, sess: tf.Session = None, config: dict = None):
        super().__init__(sess=sess, config=config)

        # PLACEHOLDERS
        self.input_ids_ph = None
        self.input_mask_ph = None
        self.segment_ids_ph = None
        self.first_pieces_coords_ph = None
        self.num_pieces_ph = None  # для обучаемых с нуля рекуррентных слоёв
        self.num_tokens_ph = None  # для crf

        # LAYERS
        self.bert_dropout = None
        self.birnn_bert = None

    def _build_embedder(self):
        self.bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
        self.bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]

    def _build_bert(self, training):
        if self.config["model"]["bert"]["test_mode"]:
            input_shape = tf.shape(self.input_ids_ph)
            bert_dim = self.config["model"]["bert"]["params"]["hidden_size"]
            x = tf.random.uniform((input_shape[0], input_shape[1], bert_dim))
            return x
        else:
            bert_scope = self.config["model"]["bert"]["scope"]
            reuse = not training
            with tf.variable_scope(bert_scope, reuse=reuse):
                bert_config = BertConfig.from_dict(self.config["model"]["bert"]["params"])
                model = BertModel(
                    config=bert_config,
                    is_training=training,
                    input_ids=self.input_ids_ph,
                    input_mask=self.input_mask_ph,
                    token_type_ids=self.segment_ids_ph
                )
                x = model.get_sequence_output()
            return x

    def _set_layers(self):
        self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])
        if self.config["model"]["birnn"]["use"]:
            self.birnn_bert = StackedBiRNN(**self.config["model"]["birnn"]["params"])

    def _actual_name_to_checkpoint_name(self, name: str) -> str:
        bert_scope = self.config["model"]["bert"]["scope"]
        prefix = f"{self.model_scope}/{bert_scope}/"
        name = name[len(prefix):]
        name = name.replace(":0", "")
        return name

    def _set_placeholders(self):
        # bert inputs
        self.input_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")

        # ner inputs
        # [id_example, id_piece]
        self.first_pieces_coords_ph = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name="first_pieces_coords")
        self.num_pieces_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_pieces")
        self.num_tokens_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_tokens")

        # common inputs
        self.training_ph = tf.placeholder(dtype=tf.bool, shape=None, name="training_ph")

    def reset_weights(self, scope: str = None, **kwargs):
        """в kwargs должен быть bert_dir!"""
        super().reset_weights(scope=scope)

        if not self.config["model"]["bert"]["test_mode"]:
            bert_scope = self.config["model"]["bert"]["scope"]
            var_list = {
                self._actual_name_to_checkpoint_name(x.name): x for x in tf.trainable_variables()
                if x.name.startswith(f"{self.model_scope}/{bert_scope}")
            }
            saver = tf.train.Saver(var_list)
            bert_dir = kwargs["bert_dir"]
            checkpoint_path = os.path.join(bert_dir, "bert_model.ckpt")
            saver.restore(self.sess, checkpoint_path)

    def _set_train_op(self):
        num_samples = self.config["training"]["num_train_samples"]
        batch_size = self.config["training"]["batch_size"]
        num_epochs = self.config["training"]["num_epochs"]
        num_train_steps = int(num_samples / batch_size) * num_epochs
        warmup_proportion = self.config["optimizer"]["warmup_proportion"]
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        init_lr = self.config["optimizer"]["init_lr"]
        self.train_op = create_optimizer(
            loss=self.loss,
            init_lr=init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=False
        )

    @property
    def _is_bpe_level(self) -> bool:
        return True

    def _get_token_level_embeddings(self, bert_out: tf.Tensor) -> tf.Tensor:
        # dropout
        bert_out = self.bert_dropout(bert_out, training=self.training_ph)

        # pieces -> tokens
        x = tf.gather_nd(bert_out, self.first_pieces_coords_ph)  # [batch_size, num_tokens, bert_dim]

        # birnn  TODO: мб это преобразования стоит делать перед преобразованием "pieces -> tokens"?
        if self.birnn_bert is not None:
            sequence_mask = tf.sequence_mask(self.num_tokens_ph)
            x = self.birnn_bert(x, training=self.training_ph, mask=sequence_mask)  # [N, num_tokens, cell_dim * 2]
        return x

    def _get_bert_input_for_feed_dict(self, examples: List[Example]) -> BertInputs:
        input_ids = []
        input_mask = []
        segment_ids = []
        first_pieces_coords = []
        num_tokens = []
        num_pieces = []

        for i, x in enumerate(examples):
            input_ids_i = []
            input_mask_i = []
            segment_ids_i = []
            first_pieces_coords_i = []
            num_tokens_i = 0

            # [CLS]
            input_ids_i.append(self.config["model"]["bert"]["cls_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            for t in x.tokens:
                n = len(t.token_ids)
                if n == 0:
                    print(f"[{x.id}] [WARNING] token <bos>{t.text}<eos> "
                          f"(span_abs: {t.span_abs}, span_rel: {t.span_rel}, "
                          f"index_abs: {t.index_abs}, index_rel: {t.index_rel}) could not be split by pieces")
                    print(f"num chars: {len(t.text)}")
                    if len(t.text) == 1:
                        print(f"unicode code point: {ord(t.text)}")
                    # без continue в first_pieces_coords_i запишется дубликат
                    continue
                first_pieces_coords_i.append((i, len(input_ids_i)))
                input_ids_i += t.token_ids
                input_mask_i += [1] * n
                segment_ids_i += [0] * n
                num_tokens_i += 1

            # [SEP]
            input_ids_i.append(self.config["model"]["bert"]["sep_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            input_ids.append(input_ids_i)
            input_mask.append(input_mask_i)
            segment_ids.append(segment_ids_i)
            first_pieces_coords.append(first_pieces_coords_i)
            num_tokens.append(num_tokens_i)
            num_pieces.append(len(input_ids_i))

        # padding
        pad_token_id = self.config["model"]["bert"]["pad_token_id"]
        num_tokens_max = max(num_tokens)
        num_pieces_max = max(num_pieces)
        for i in range(len(examples)):
            input_ids[i] += [pad_token_id] * (num_pieces_max - num_pieces[i])
            input_mask[i] += [0] * (num_pieces_max - num_pieces[i])
            segment_ids[i] += [0] * (num_pieces_max - num_pieces[i])
            first_pieces_coords[i] += [(i, 0)] * (num_tokens_max - num_tokens[i])

        return BertInputs(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            first_pieces_coords=first_pieces_coords,
            num_tokens=num_tokens,
            num_pieces=num_pieces
        )


class BaseModelNER(BaseModel):
    ner_scope = "ner"

    def __init__(self, sess, config: Dict = None, ner_enc: Dict = None):
        super().__init__(sess=sess, config=config)
        self._ner_enc = None
        self._inv_ner_enc = None

        self.ner_enc = ner_enc

        self.ner_labels_ph = None

    def _build_graph(self):
        self._build_embedder()
        with tf.variable_scope(self.ner_scope):
            self._build_ner_head()

    def save_config(self, model_dir: str):
        assert self.ner_enc is not None
        super().save_config(model_dir=model_dir)
        with open(os.path.join(model_dir, "ner_enc.json"), "w") as f:
            json.dump(self.ner_enc, f, indent=4)

    @classmethod
    def load(cls, sess: tf.Session, model_dir: str, scope_to_load: str = None, mode: str = ModeKeys.TEST):
        model = super().load(sess=sess, model_dir=model_dir, scope_to_load=scope_to_load, mode=mode)
        with open(os.path.join(model_dir, "ner_enc.json")) as f:
            model.ner_enc = json.load(f)
        return model

    @abstractmethod
    def _build_ner_head(self):
        pass

    @property
    def ner_enc(self):
        return self._ner_enc

    @property
    def inv_ner_enc(self):
        return self._inv_ner_enc

    @ner_enc.setter
    def ner_enc(self, ner_enc: Dict):
        self._ner_enc = ner_enc
        if ner_enc is not None:
            self._inv_ner_enc = {v: k for k, v in ner_enc.items()}


class BaseModelRelationExtraction(BaseModelNER):
    """
    сущности уже известны. требуется найти отношения между ними.
    наследуется от BaseModelNER, потому что ner_enc тоже нужен для возмонжости использовать в модели лейблы сущностей.
    """
    re_scope = "re"

    def __init__(self, sess, config: Dict = None, ner_enc: Dict = None, re_enc: Dict = None):
        super().__init__(sess=sess, config=config)
        self._ner_enc = None
        self._inv_ner_enc = None
        self.ner_enc = ner_enc

        self._re_enc = None
        self._inv_re_enc = None
        self.re_enc = re_enc

        self.ner_labels_ph = None

    def _build_graph(self):
        self._build_embedder()
        with tf.variable_scope(self.re_scope):
            self._build_re_head()

    # TODO: костыль
    def _build_ner_head(self):
        pass

    def save_config(self, model_dir: str):
        assert self.ner_enc is not None
        assert self.re_enc is not None
        super().save_config(model_dir=model_dir)
        with open(os.path.join(model_dir, "ner_enc.json"), "w") as f:
            json.dump(self.ner_enc, f, indent=4)
        with open(os.path.join(model_dir, "re_enc.json"), "w") as f:
            json.dump(self.re_enc, f, indent=4)

    @classmethod
    def load(cls, sess: tf.Session, model_dir: str, scope_to_load: str = None, mode: str = ModeKeys.TEST):
        model = super().load(sess=sess, model_dir=model_dir, scope_to_load=scope_to_load, mode=mode)
        with open(os.path.join(model_dir, "ner_enc.json")) as f:
            model.ner_enc = json.load(f)
        with open(os.path.join(model_dir, "re_enc.json")) as f:
            model.re_enc = json.load(f)
        return model

    @abstractmethod
    def _build_re_head(self):
        pass

    @property
    def re_enc(self):
        return self._re_enc

    @property
    def inv_re_enc(self):
        return self._inv_re_enc

    @re_enc.setter
    def re_enc(self, re_enc: Dict):
        self._re_enc = re_enc
        if re_enc is not None:
            self._inv_re_enc = {v: k for k, v in re_enc.items()}


# лучше делать отдельный класс под joint модели (ner + re, mentions + coreference):
# * для ner + re в конфиге нужна секция model.ner, а в re - нет
# * для ner + re нужно создавать слои под ner, а для re - нет
# * для инференса модели re нужны истинные ner-лейблы, а для инференса модели ner + re - нет.
class BaseModelNerAndRelationExtracion(BaseModelRelationExtraction):
    """
    требуется найти сущности и отношения между ними
    """
    def __init__(self, sess, config: Dict = None, ner_enc: Dict = None, re_enc: Dict = None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

    def _build_graph(self):
        self._build_embedder()
        with tf.variable_scope(self.ner_scope):
            self._build_ner_head()
        with tf.variable_scope(self.re_scope):
            self._build_re_head()


class BaseModeDependencyParsing(BaseModel):
    dep_scope = "dependency_parser"

    def __init__(self, sess: tf.Session = None, config: Dict = None, rel_enc: Dict = None):
        super().__init__(sess=sess, config=config)

        # PLACEHOLDERS
        self.labels_ph = None  # [id_example, id_head, id_dep, id_rel]

        # LAYERS
        self.birnn = None
        self.arc_enc = None
        self.type_enc = None

        # TENSORS
        self.logits_arc_train = None
        self.logits_type_train = None
        self.s_arc = None
        self.type_labels_pred = None
        # for debug:
        self.total_loss_arc = None
        self.total_loss_type = None

        self._rel_enc = None
        self._inv_rel_enc = None

        self.rel_enc = rel_enc

    def _build_graph(self):
        self._build_embedder()
        with tf.variable_scope(self.dep_scope):
            self._build_dependency_parser()

    @abstractmethod
    def _build_dependency_parser(self):
        pass

    @property
    def rel_enc(self):
        return self._rel_enc

    @property
    def inv_rel_enc(self):
        return self._inv_rel_enc

    @rel_enc.setter
    def rel_enc(self, rel_enc: Dict):
        self._rel_enc = rel_enc
        if rel_enc is not None:
            self._inv_rel_enc = {v: k for k, v in rel_enc.items()}

    def save_config(self, model_dir: str):
        assert self.rel_enc is not None
        super().save_config(model_dir=model_dir)
        with open(os.path.join(model_dir, "rel_enc.json"), "w") as f:
            json.dump(self.rel_enc, f, indent=4)

    @classmethod
    def load(cls, sess: tf.Session, model_dir: str, scope_to_load: str = None, mode: str = ModeKeys.TEST):
        model = super().load(sess=sess, model_dir=model_dir, scope_to_load=scope_to_load, mode=mode)
        with open(os.path.join(model_dir, "rel_enc.json")) as f:
            model.rel_enc = json.load(f)
        return model


class BaseModeCoreferenceResolution(BaseModel):
    coref_scope = "coref"
    coref_rel = "COREFERENCE"

    def __init__(self, sess: tf.Session = None, config: Dict = None):
        super().__init__(sess=sess, config=config)

        # PLACEHOLDERS
        self.mention_spans_ph = None  # [id_example, start, end]
        self.labels_ph = None  # [id_example, id_anaphora, id_antecedent]

        # LAYERS
        self.birnn = None

        # TENSORS
        self.logits_train = None
        self.logits_pred = None
        self.total_loss = None
        self.loss_denominator = None

    def _build_graph(self):
        self._build_embedder()
        with tf.variable_scope(self.coref_scope):
            self._build_coref_head()

    @abstractmethod
    def _build_coref_head(self):
        pass
