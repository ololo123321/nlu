import json
import os
from argparse import ArgumentParser

import tensorflow as tf

from src.model import JointModelV1
from src.preprocessing import ExampleEncoder, ExamplesLoader
from src.utils import check_entities_spans


def main(args):
    event_tag = "Bankruptcy"  # TODO: вынести в конфиг

    # подгрузка конфига
    config = json.load(open(os.path.join(args.model_dir, "config.json")))

    # подгрузка примеров
    loader = ExamplesLoader(
        ner_encoding=config["preprocessing"]["ner_encoding"],
        ner_prefix_joiner=config["preprocessing"]["ner_prefix_joiner"],
        event_tags={event_tag}
    )
    examples = loader.load_examples(
        data_dir=args.data_dir,
        n=None,
        split=config["preprocessing"]["split"],
        window=config["preprocessing"]["window"],
    )

    # удаление рёбер, если они есть. иначе будет феил при сохранении предиктов
    # удаление событий, т.к. их мы сами предсказываем
    for x in examples:
        x.arcs.clear()
        x.entities = x.entities_wo_events

    # кодирование примеров
    example_encoder = ExampleEncoder.load(encoder_dir=args.model_dir)

    examples_encoded = example_encoder.transform(examples)

    assert all(x.filename is not None for x in examples_encoded)

    # print("saving predictions")
    # id2relation = {v: k for k, v in example_encoder.vocab_re.encodings.items()}
    # save_predictions(examples=examples_encoded, output_dir=args.output_dir, id2relation=id2relation)

    # создание модели + подгрузка весов
    tf.reset_default_graph()
    sess = tf.Session()
    model = JointModelV1(sess, config)
    model.build()
    model.restore(model_dir=args.model_dir)
    model.initialize()

    print("checking examples...")
    check_entities_spans(examples=examples_encoded, span_emb_type=config["model"]["re"]["span_embeddings"]["type"])
    print("OK")

    # рёбра пишутся в сразу в инстансы классов Example
    model.predict(examples_encoded, batch_size=args.batch_size)

    def check_arcs():
        """
        каждое ребро должно иметь уникальный айдишник
        """
        from collections import defaultdict

        d_set = defaultdict(set)
        d_int = defaultdict(int)

        for x in examples_encoded:
            assert x.filename is not None
            for arc in x.arcs:
                d_set[x.filename].add(arc.id)
                d_int[x.filename] += 1
        for x in examples_encoded:
            assert len(d_set[x.filename]) == d_int[x.filename], f"id: {x.id}, filename: {x.filename}: {len(d_set[x.filename])} != {d_int[x.filename]}"

    check_arcs()

    print("saving predictions")
    id2relation = {v: k for k, v in example_encoder.vocab_re.encodings.items()}
    loader.save_predictions(
        examples=examples_encoded,
        output_dir=args.output_dir,
        id2relation=id2relation,
        copy_texts=args.copy_texts,
        collection_dir=args.data_dir
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--copy_texts", action="store_true",
                        help="нужно ли копировать тексты в папку с ответами, "
                             "чтоб эту папку можно было просто перетащить в сервис разметки для дальнейшей работы"
                        )

    _args = parser.parse_args()
    print(_args)

    main(_args)
