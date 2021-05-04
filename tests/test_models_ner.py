import tensorflow as tf

from src.model.ner import BertForFlatNER, BertForNestedNER
from src.data.base import Example, Entity, Token


def build_examples():
    tokens = [
        Token(text="мама", labels=["B_FOO"], token_ids=[3], index_abs=0, index_rel=0),
        Token(text="мыла", labels=["I_FOO"], token_ids=[4, 5], index_abs=1, index_rel=1),
        Token(text="раму", labels=["O"], token_ids=[6], index_abs=2, index_rel=2)
    ]
    entities = [
        Entity(text="мама мыла", tokens=tokens[:2], label="FOO")
    ]
    chunk = Example(id="chunk", tokens=tokens, entities=entities)
    example = Example(id="0", tokens=tokens, entities=entities, chunks=[chunk])
    return [example]


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
    }
}


def _test_model(model_cls, config, ner_enc):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = model_cls(sess=sess, config=config, ner_enc=ner_enc)
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
