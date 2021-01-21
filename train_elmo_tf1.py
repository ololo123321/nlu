import random
import sys
import json
import os
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import tensorflow as tf

from src.model import RelationExtractor
from src.preprocessing import ParserRuREBus, ExampleEncoder, check_example


NER_ENCODING = 'bilou'
NER_SUFFIX_JOINER = '-'


def load_examples(data_dir, split=True, window=1):
    examples = []
    num_bad = 0
    num_examples = 0
    p = ParserRuREBus(ner_encoding=NER_ENCODING, ner_suffix_joiner=NER_SUFFIX_JOINER)
    for x_raw in p.parse(data_dir=data_dir, n=None):
        # проверяем целый пример
        try:
            check_example(x_raw, ner_encoding=NER_ENCODING)
        except AssertionError as e:
            print("[doc]", e)
            num_bad += 1
            continue
        if split:
            for x_raw_chunk in x_raw.chunks(window=window):
                num_examples += 1
                try:
                    check_example(x_raw_chunk, ner_encoding=NER_ENCODING)
                    examples.append(x_raw_chunk)
                except AssertionError as e:
                    print("[sent]", e)
                    num_bad += 1
        else:
            num_examples += 1
            examples.append(x_raw)
    print(f"{num_bad} / {len(examples)} examples are bad")
    return examples


def main(args):
    examples_train = load_examples(data_dir=args.train_data_dir, split=bool(args.split), window=args.window)
    examples_valid = load_examples(data_dir=args.valid_data_dir, split=bool(args.split), window=args.window)

    print("num train examples:", len(examples_train))
    print("num valid examples:", len(examples_valid))

    examples_train = [x for x in examples_train if len(x.entities) > 0]
    examples_valid = [x for x in examples_valid if len(x.entities) > 0]

    print("num train examples filtered:", len(examples_train))
    print("num valid examples filtered:", len(examples_valid))

    add_seq_bounds = args.span_emb_type == 1
    example_encoder = ExampleEncoder(
        ner_encoding=NER_ENCODING,
        ner_suffix_joiner=NER_SUFFIX_JOINER,
        add_seq_bounds=add_seq_bounds
    )

    examples_train_encoded = example_encoder.fit_transform(examples_train)
    examples_valid_encoded = example_encoder.transform(examples_valid)

    print("saving encodings...")
    example_encoder.save(encoder_dir=args.model_dir)

    config = {
        "model": {
            # конфигурация веткоризации токенов
            "embedder": {
                # векторные представления токенов
                "type": "elmo",
                "dir": args.elmo_dir,
                "dropout": args.elmo_dropout,
                "dim": 1024,
                "attention": {
                    "enabled": False,
                    "num_layers": 4,
                    "num_heads": 4,
                    "head_dim": 32,
                    "dff": 512,
                    "dropout_rc": 0.2,
                    "dropout_ff": 0.2
                },
                "rnn": {
                    "enabled": True,
                    "num_layers": args.num_recurrent_layers,
                    "skip_connections": False,
                    "cell_name": args.cell_name,
                    "cell_dim": args.cell_dim,
                    "dropout": args.rnn_dropout,
                    "recurrent_dropout": 0.0
                }
            },
            # конфигурация головы, решающей relation extraction
            "re": {
                "ner_embeddings": {
                    "use": bool(args.use_ner_emb),
                    "num_labels": example_encoder.vocab_ner.size,
                    "dim": 1024,
                    "dropout": args.ner_emb_dropout,
                },
                "merged_embeddings": {
                    "merge_mode": "sum",  # {'sum', 'mul', 'concat', 'ave'}
                    "dropout": args.merged_emb_dropout,
                    "layernorm": True
                },
                "span_embeddings": {
                    "type": args.span_emb_type,
                },
                "mlp": {
                    "num_layers": 1,
                    "dropout": 0.33
                },
                "bilinear": {
                    "num_labels": example_encoder.vocab_re.size,
                    "hidden_dim": 128
                },
                "no_rel_id": example_encoder.vocab_re.get_id("O"),
                "ner_other_label_id": example_encoder.vocab_ner.get_id("O"),
                "version": 2
            },
        },
        "optimizer": {
            # noam
            "noam_scheme": False,
            # reduce on plateau (если True, то предыдущая опция игнорится)
            "reduce_lr_on_plateau": True,
            "max_steps_wo_improvement": 50,
            "lr_reduce_patience": 5,
            "lr_reduction_factor": 0.7,
            # custome schedule (если True, то предыдущая опция игнорится)
            "custom_schedule": True,
            "min_lr": 1e-5,
            # opt name
            "opt_name": "adam",  # {adam, adamw}, имеет значение только при аккумуляции градиентов
            # gradients accumulation
            "accumulate_gradients": False,
            "num_accumulation_steps": 1,  # имеет значение только при аккумуляции градиентов
            "init_lr": 1e-3,
            "warmup_steps": 180,
            # gradients clipping
            "clip_grads": True,
            "clip_norm": 1.0
        }
    }
    print("model and training config:")
    print(config)

    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    tf.reset_default_graph()
    sess = tf.Session()
    model = RelationExtractor(sess, config)
    model.build()
    model.initialize()

    def print_tvars_info():
        print("=" * 50)
        print("TRAINABLE VARIABLES:")
        n = 0
        for v in tf.trainable_variables():
            ni = 1
            for dim in v.shape:
                ni *= dim
            print(f"name: {v.name}; shape {v.shape}; num weights: {ni}")
            n += ni

        print("num trainable params:", n)
        print("=" * 50)

    print_tvars_info()

    print("train size:", len(examples_train_encoded))
    print("valid size:", len(examples_valid_encoded))

    def check_entities_spans(examples_):
        for x in examples_:
            for entity in x.entities:
                actual = ' '.join(x.tokens[entity.start_token_id:entity.end_token_id + 1])
                expected = ' '.join(entity.tokens)
                assert actual == expected
                assert entity.start_token_id > 0

    print("checking examples...")
    check_entities_spans(examples_train_encoded + examples_valid_encoded)
    print("OK")

    checkpoint_path = os.path.join(args.model_dir, "model.ckpt")

    model.train(
        train_examples=examples_train_encoded,
        eval_examples=examples_valid_encoded,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        no_rel_id=example_encoder.vocab_re.get_id("O"),
        id2label=example_encoder.vocab_re.inv_encodings,
        checkpoint_path=checkpoint_path
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data_dir")
    parser.add_argument("--valid_data_dir")
    parser.add_argument("--elmo_dir")
    parser.add_argument("--model_dir")
    parser.add_argument("--elmo_dropout", type=float, default=0.1, required=False)
    parser.add_argument("--use_ner_emb", type=int, default=1, help="приниимает int 0 (False) или 1 (True)")
    parser.add_argument("--span_emb_type", type=int, default=1, help="приниимает int 0 (False) или 1 (True)")
    parser.add_argument("--ner_emb_dropout", type=float, default=0.2, required=False)
    parser.add_argument("--merged_emb_dropout", type=float, default=0.0, required=False)
    parser.add_argument("--num_recurrent_layers", type=int, default=2, required=False)
    parser.add_argument("--cell_name", choices=["gru", "lstm"], default="lstm", required=False)
    parser.add_argument("--cell_dim", type=int, default=128, required=False)
    parser.add_argument("--rnn_dropout", type=float, default=0.5, required=False)
    parser.add_argument("--epochs", type=int, default=50, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--split", type=int, default=1, help="приниимает int 0 (False) или 1 (True)")
    parser.add_argument("--window", type=int, default=1, required=False)

    _args = parser.parse_args()
    print(_args)

    main(_args)
