import copy
import tensorflow as tf

from src.model.ner import BertForFlatNER, BertForNestedNER
from src.model.coreference_resolution import BertForCoreferenceResolutionMentionPair, BertForCoreferenceResolutionMentionRanking
from src.data.base import Example, Entity, Token, Span


def build_examples():
    tokens = [
        Token(text="мама", labels=["B_FOO"], token_ids=[3], index_abs=0, index_rel=0, id_sent=0,
              span_abs=Span(start=0, end=4), span_rel=Span(start=0, end=4)),
        Token(text="мыла", labels=["I_FOO"], token_ids=[4, 5], index_abs=1, index_rel=1, id_sent=0,
              span_abs=Span(start=5, end=9), span_rel=Span(start=5, end=9)),
        Token(text="раму", labels=["O"], token_ids=[6], index_abs=2, index_rel=2, id_sent=0,
              span_abs=Span(start=10, end=14), span_rel=Span(start=10, end=14))
    ]
    entities = [
        Entity(text="мама мыла", tokens=tokens[:2], label="FOO", id_chain=0, index=0)
    ]
    text = "мама мыла раму"
    _examples = [
        Example(id="0", filename="0", text=text, tokens=tokens, entities=entities, chunks=[
            Example(id="chunk_0", tokens=tokens, entities=entities, parent="0")
        ]),
        Example(id="1", filename="1", text=text, tokens=tokens, entities=entities, chunks=[
            Example(id="chunk_1", tokens=tokens, entities=entities, parent="1")
        ]),
        Example(id="2", filename="2", text=text, tokens=tokens, entities=entities, chunks=[
            Example(id="chunk_2", tokens=tokens, entities=entities, parent="2")
        ])
    ]
    return _examples


examples = build_examples()

common_config = {
    "model": {
        "bert": {
            "test_mode": True,
            "dir": None,
            "dim": 16,
            "attention_probs_dropout_prob": 0.5,  # default 0.1
            "hidden_dropout_prob": 0.1,
            "dropout": 0.2,
            "scope": "bert",
            "pad_token_id": 0,
            "cls_token_id": 1,
            "sep_token_id": 2,
        },
        "ner": None  # setup
    },
    "training": {
        "num_epochs": 1,
        "batch_size": 16,
        "max_epochs_wo_improvement": 1,
        "num_train_samples": 100,
    },
    "optimizer": {
        "init_lr": 2e-5,
        "warmup_proportion": 0.1,
    },
    "inference": {
        "max_tokens_per_batch": 100,
        "window": 1
    },
    "valid": {}  # чтоб пайчарм не подчёркивал ниже
}

folds = [
    (["0", "1"], ["2"]),
    (["0", "2"], ["1"]),
    (["1", "2"], ["0"])
]


def _test_model(model_cls, config, **kwargs):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = model_cls(sess=sess, config=config, **kwargs)

        model.build()

        model.train(
            examples_train=examples,
            examples_valid=examples,
            train_op_name="train_op",
            model_dir=None,
            scope_to_save=None,
            verbose=True,
            verbose_fn=None
        )

        model.cross_validate(
            examples=examples,
            folds=folds,
            valid_frac=0.5,
            verbose_fn=None
        )

        examples_test = copy.deepcopy(examples)
        for x in examples_test:
            x.entities = []
            for t in x.tokens:
                t.labels = []
        model.predict(examples=examples_test)


def test_bert_for_flat_ner():
    ner_enc = {
        "O": 0,
        "B_FOO": 1,
        "I_FOO": 2,
        "B_BAR": 3,
        "I_BAR": 4
    }
    config = common_config.copy()
    config["model"]["ner"] = {
        "use_crf": True,
        "num_labels": len(ner_enc),
        "no_entity_id": 0,
        "start_ids": [v for k, v in ner_enc.items() if k[0] == "B"],
        "prefix_joiner": "-",
        "use_birnn": False,
        "rnn": {
            "num_layers": 1,
            "cell_dim": 256,
            "dropout": 0.5,
            "recurrent_dropout": 0.0
        }
    }
    _test_model(BertForFlatNER, config=config, ner_enc=ner_enc)


def test_bert_for_nested_ner():
    ner_enc = {
        "O": 0,
        "FOO": 1,
        "BAR": 2
    }
    config = common_config.copy()
    config["model"]["ner"] = {
        "no_entity_id": 0,
        "use_birnn": False,
        "rnn": {
            "num_layers": 1,
            "cell_dim": 256,
            "dropout": 0.5,
            "recurrent_dropout": 0.0
        },
        "biaffine": {
            "num_mlp_layers": 1,
            "activation": "relu",
            "head_dim": 128,
            "dep_dim": 128,
            "dropout": 0.33,
            "num_labels": len(ner_enc),
        }
    }
    _test_model(BertForNestedNER, config=config, ner_enc=ner_enc)


def test_bert_for_cr_mention_pair():
    config = common_config.copy()
    config["model"]["coref"] = {
        "use_birnn": False,
        "rnn": {
            "num_layers": 1,
            "cell_dim": 256,
            "dropout": 0.5,
            "recurrent_dropout": 0.0
        },
        "use_attn": True,
        "attn": {
            "hidden_dim": 128,
            "dropout": 0.3,
            "activation": "relu"
        },
        "hoi": {
            "order": 2,
            "w_dropout": 0.5,
            "w_dropout_policy": 0  # 0 - one mask; 1 - different mask
        },
        "biaffine": {
            "num_mlp_layers": 1,
            "activation": "relu",
            "head_dim": 128,
            "dep_dim": 128,
            "dropout": 0.33,
            "num_labels": 1,
            "use_dep_prior": False
        }
    }
    config["valid"] = {
        "path_true": "/tmp/gold.conll",
        "path_pred": "/tmp/pred.conll",
        "scorer_path": "/home/vitaly/reference-coreference-scorers/scorer.pl"
    }
    _test_model(BertForCoreferenceResolutionMentionPair, config=config)
