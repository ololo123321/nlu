import json
import os
from argparse import ArgumentParser

import tensorflow as tf

from src.model import JointModelV1
from src.preprocessing import ExampleEncoder, ExamplesLoader, NerEncodings
from src.utils import check_entities_spans

# TODO: добавить в конфиг и использовать при инференсе
NER_SUFFIX_JOINER = '-'


def main(args):
    loader = ExamplesLoader(
        ner_encoding=NerEncodings.BILOU,  # TODO: добавить в конфиг и использовать при инференсе
        ner_suffix_joiner=NER_SUFFIX_JOINER,  # TODO: добавить в конфиг и использовать при инференсе
        event_tags={"Bankruptcy"}
    )

    examples_train = loader.load_examples(
        data_dir=args.train_data_dir,
        n=None,
        split=bool(args.split),  # TODO: добавить в конфиг и использовать при инференсе
        window=args.window,  # TODO: добавить в конфиг и использовать при инференсе
    )

    examples_valid = loader.load_examples(
        data_dir=args.valid_data_dir,
        n=None,
        split=bool(args.split),
        window=args.window,
    )

    print("num train examples:", len(examples_train))
    print("num valid examples:", len(examples_valid))

    examples_train = [x for x in examples_train if len(x.entities) > 0]
    examples_valid = [x for x in examples_valid if len(x.entities) > 0]

    print("num train examples filtered:", len(examples_train))
    print("num valid examples filtered:", len(examples_valid))

    add_seq_bounds = args.span_emb_type == 1
    example_encoder = ExampleEncoder(
        ner_encoding=NerEncodings.BILOU,
        ner_suffix_joiner=NER_SUFFIX_JOINER,
        add_seq_bounds=add_seq_bounds
    )

    examples_train_encoded = example_encoder.fit_transform(examples_train)
    examples_valid_encoded = example_encoder.transform(examples_valid)

    print("saving encodings...")
    example_encoder.save(encoder_dir=args.model_dir)

    # вывод айдишников начала событий и остальных именых сущностей
    start_ids_entity = []
    start_ids_event = []
    event_tag = "BANKRUPTCY"
    for label, i in example_encoder.vocab_ner.encodings.items():
        if label.startswith("B"):
            if event_tag.endswith(event_tag):
                start_ids_event.append(i)
            else:
                start_ids_entity.append(i)

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
            # поиск
            "ner": {
                "start_ids": start_ids_entity,
                "other_label_id": 0
            },
            # событие
            "event": {
                "num_labels": example_encoder.vocab_re.size,  # TODO: должен быть отдельный vocab под событие
                "start_ids": start_ids_event,
                "tag": "Bankruptcy",
                "other_label_id": 0
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
                    "dropout": args.mlp_dropout,
                    "activation": "relu"
                },
                "bilinear": {
                    "num_labels": example_encoder.vocab_re.size,
                    "hidden_dim": 128  # TODO: вынести в аргументы
                },
                "no_rel_id": example_encoder.vocab_re.get_id("O")
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
            # custom schedule (если True, то предыдущая опция игнорится)
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
    model = JointModelV1(sess, config)
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

    print("checking examples...")
    check_entities_spans(examples=examples_train_encoded + examples_valid_encoded, span_emb_type=args.span_emb_type)
    print("OK")

    checkpoint_path = os.path.join(args.model_dir, "model.ckpt")

    model.train(
        train_examples=examples_train_encoded,
        eval_examples=examples_valid_encoded,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        no_rel_id=example_encoder.vocab_re.get_id("O"),
        id2label_ner=example_encoder.vocabs_events[event_tag].inv_encodings,
        id2label_roles=example_encoder.vocab_re.inv_encodings,
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
    parser.add_argument("--mlp_dropout", type=float, default=0.33, required=False)
    parser.add_argument("--epochs", type=int, default=50, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--split", type=int, default=1, help="приниимает int 0 (False) или 1 (True)")
    parser.add_argument("--window", type=int, default=1, required=False)

    _args = parser.parse_args()
    print(_args)

    main(_args)
