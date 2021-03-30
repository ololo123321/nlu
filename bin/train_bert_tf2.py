import tqdm
import os
import json
from typing import List, Dict
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_hub as hub

from official.nlp.bert.tokenization import FullTokenizer
from official.nlp.optimization import create_optimizer


def create_bert_examples(data, label2id, symbol2piece, vocab_file, do_lower_case):

    tokenizer = FullTokenizer(vocab_file, do_lower_case=do_lower_case)
    token2pieces = {}

    def create_bert_example(x):
        tokens_new = ['[CLS]']

        def get_pieces(token):
            if token not in token2pieces:
                token2pieces[token] = tokenizer.tokenize(token)
            return token2pieces[token]

        def get_label_id(label):
            if label not in label2id:
                label2id[label] = len(label2id)
            return label2id[label]

        def get_special_symbol(t):
            if t not in symbol2piece:
                symbol2piece[t] = f'[unused{len(symbol2piece) + 1}]'
            return symbol2piece[t]

        subj = f'subj={x["subj_type"]}'
        obj = f'obj={x["obj_type"]}'

        for i, pieces in enumerate(map(get_pieces, x["token"])):

            if i == x['subj_start']:
                tokens_new.append(get_special_symbol(subj))
            if i == x['obj_start']:
                tokens_new.append(get_special_symbol(obj))

            if not ((x['subj_start'] <= i <= x['subj_start']) or (x['obj_start'] <= i <= x['obj_start'])):
                tokens_new += pieces

        tokens_new.append('[SEP]')

        token_ids = tokenizer.convert_tokens_to_ids(tokens_new)

        x_new = dict(id=x["id"], tokens=token_ids, label=get_label_id(x["relation"]))

        return x_new

    examples = list(map(create_bert_example, tqdm.tqdm(data)))
    return examples


def build_model(model_dir, dropout, num_relations):
    tf.keras.backend.clear_session()
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    input_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="segment_ids")
    bert_layer = hub.KerasLayer(model_dir, trainable=True, name='bert')
    inputs = [input_ids, input_mask, segment_ids]
    pooled_output, _ = bert_layer(inputs)
    x = tf.keras.layers.Dropout(dropout)(pooled_output)
    output = tf.keras.layers.Dense(num_relations, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


class ExamplesIterator:
    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __iter__(self):
        for x in self.examples:
            token_ids = x["tokens"]
            input_mask = [1] * len(token_ids)
            segment_ids = [0] * len(token_ids)
            yield (token_ids, input_mask, segment_ids), x["label"]

    @property
    def output_types(self):
        return (tf.int32, tf.int32, tf.int32), tf.int32

    @property
    def output_shapes(self):
        return (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])), tf.TensorShape([])


def build_ds(it, training: bool, batch_size=16, buffer=10000):
    ds = tf.data.Dataset.from_generator(lambda: it, output_types=it.output_types, output_shapes=it.output_shapes)
    if training:
        ds = ds.shuffle(buffer)
        ds = ds.repeat()
    ds = ds.padded_batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def main(args):
    assert tf.test.is_gpu_available()

    data_train = json.load(open(os.path.join(args.data_dir, "train.json")))[:1000]
    data_valid = json.load(open(os.path.join(args.data_dir, "dev.json")))[:1000]

    labels_path = os.path.join(args.model_dir, "label2id.json")
    if os.path.exists(labels_path):
        label2id = json.load(open(labels_path))
        print("loaded file with relations encodings:")
        print(label2id)
    else:
        label2id = {}
        print("created empty relations encodings")

    symbols_path = os.path.join(args.model_dir, "symbol2piece.json")
    if os.path.exists(symbols_path):
        symbol2piece = json.load(open(symbols_path))
        print("loaded file with special symbols encodings:")
        pritn(symbol2piece)
    else:
        symbol2piece = {}
        print("created empty special symbols encodings")

    vocab_file = os.path.join(args.pretrained_dir, "assets", "vocab.txt")

    examples_train = create_bert_examples(
        data=data_train,
        label2id=label2id,
        symbol2piece=symbol2piece,
        vocab_file=vocab_file,
        do_lower_case=args.do_lower_case
    )

    examples_valid = create_bert_examples(
        data=data_valid,
        label2id=label2id,
        symbol2piece=symbol2piece,
        vocab_file=vocab_file,
        do_lower_case=args.do_lower_case
    )

    it_train = ExamplesIterator(examples=examples_train)
    it_valid = ExamplesIterator(examples=examples_valid)

    ds_train = build_ds(it_train, training=True, batch_size=args.batch_size_train, buffer=args.buffer)
    ds_valid = build_ds(it_valid, training=False, batch_size=args.batch_size_valid)

    model = build_model(model_dir=args.pretrained_dir, dropout=args.dropout, num_relations=len(label2id))

    steps_per_epoch = len(examples_train) // args.batch_size_train + 1
    num_train_steps = args.epochs * steps_per_epoch
    warmup_steps = int(num_train_steps * args.warmup_proportion)
    optimizer = create_optimizer(args.lr, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def f1_micro(y_true, y_pred):
        # TODO: проверить размерности y_true, y_pred
        mask_true = tf.not_equal(y_true, 0)
        mask_pred = tf.not_equal(y_pred, 0)
        mask = tf.concat([mask_true[:, None], mask_pred[:, None], tf.equal(y_true, y_pred)[:, None]], axis=1)
        mask_correct = tf.reduce_all(mask, axis=1)

        num_true = tf.reduce_sum(tf.cast(mask_true, tf.float32))
        num_pred = tf.reduce_sum(tf.cast(mask_pred, tf.float32))
        num_correct = tf.reduce_sum(tf.cast(mask_correct, tf.float32))

        precision = num_correct / num_pred
        recall = num_correct / num_true
        f1 = 2.0 * precision * recall / (precision + recall)

        return f1

    metrics = ["accuracy", f1_micro]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, "model.hdf5"),
            monitor='val_f1_micro',
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]
    model.fit(
        ds_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_valid,
        callbacks=callbacks,
        verbose=1
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--pretrained_dir")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--model_dir")
    parser.add_argument("--dropout", type=float, default=0.1, required=False)
    parser.add_argument("--batch_size_train", type=int, default=32, required=False)
    parser.add_argument("--batch_size_valid", type=int, default=32, required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--lr", type=float, default=2e-5, required=False)
    parser.add_argument("--warmup_proportion", type=float, default=0.1, required=False)
    parser.add_argument("--buffer", type=int, default=10000, required=False)
    args_ = parser.parse_args()
    print(args_)
    main(args_)
