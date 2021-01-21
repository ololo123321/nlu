import random
import sys
import json
import os
from typing import Set
from argparse import ArgumentParser
from copy import deepcopy
from collections import defaultdict, namedtuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from src.model import RelationExtractor
from src.preprocessing import ParserRuREBus, ExampleEncoder, check_example, Vocab


NER_ENCODING = 'bilou'
NER_SUFFIX_JOINER = '-'


def load_examples(data_dir):
    examples = []
    num_bad = 0
    num_examples = 0
    parser = ParserRuREBus(ner_encoding=NER_ENCODING, ner_suffix_joiner=NER_SUFFIX_JOINER)
    for x_raw in parser.parse(data_dir=data_dir, n=None):
        # проверяем целый пример
        try:
            check_example(x_raw, ner_encoding=NER_ENCODING)
        except AssertionError as e:
            print("[doc]", e)
            continue
        for x_raw_chunk in x_raw.chunks:
            num_examples += 1
            try:
                check_example(x_raw_chunk, ner_encoding=NER_ENCODING)
                examples.append(x_raw_chunk)
            except AssertionError as e:
                print("[sent]", e)
                num_bad += 1
    print(f"{num_bad} / {len(examples)} examples are bad")
    return examples


EventArgument = namedtuple("EventArgument", ["id", "role"])


class Event:
    def __init__(self, id=None, id_trigger=None, label=None, arguments: Set[EventArgument] = None):
        self.id_trigger = id_trigger
        self.label = label
        self.arguments = arguments if arguments is not None else set()
        self.id = id


# def save_predictions(examples, output_dir, id2relation):
#     """
#     из пар (head=event_trigger, dep=entity) брать entity и добавлять в список аргументов соответствующего события
#     """
#     # R105\tTSK Arg1:T370 Arg2:T371
#     # E0\tBankruptcy:T0 Bunkrupt:T1 Bunkrupt2:T2
#     for x in examples:
#         assert x.filename is not None, f"example with id {x.id} has no filename!"
#         with open(os.path.join(output_dir, f"{x.filename}.ann"), "a") as f:
#             for arc in x.arcs:
#                 rel = id2relation[arc.rel]
#                 f.write(f"R{arc.id}\t{rel} Arg1:{arc.head} Arg2:{arc.dep}\n")


def save_predictions(examples, output_dir, id2relation):
    event_counter = defaultdict(int)
    for x in examples:
        with open(os.path.join(output_dir, f"{x.filename}.ann"), "a") as f:
            events = {}
            # исходные сущности
            for entity in x.entities:
                # TODO: добавить атрибут сущности label
                # TODO: start_index, end_index должны быть как в исходном примере!
                line = f"{entity.id}\t{entity.label} {entity.start_index} {entity.end_index}\t{entity.text}\n"
                f.write(line)
                # TODO: добавить атрибут сущности is_event_trigger
                if entity.is_event_trigger:
                    if entity.id not in events:
                        id_event = event_counter[x.filename]
                        events[entity.id] = Event(
                            id=id_event,
                            id_trigger=entity.id,
                            label=entity.label,
                            arguments=None,
                        )
                        event_counter[x.filename] += 1

            # отношения
            for arc in x.arcs:
                rel = id2relation[arc.rel]
                if arc.head in events:
                    arg = EventArgument(id=arc.dep, role=rel)
                    events[arc.head].arguments.add(arg)
                else:
                    line = f"R{arc.id}\t{rel} Arg1:{arc.head} Arg2:{arc.dep}\n"
                    f.write(line)

            # события
            for event in events.values():
                line = f"E{event.id}\t{event.label}:{event.id_trigger}"
                role2count = defaultdict(int)
                args_str = ""
                for arg in event.arguments:
                    i = role2count[arg.role]
                    role = arg.role
                    if i > 0:
                        role += str(i + 1)
                    args_str += f"{role}:{arg.id}" + ' '
                    role2count[arg.role] += 1
                args_str = args_str.strip()
                if args_str:
                    line += ' ' + args_str
                line += '\n'
                f.write(line)


def main(args):
    # подгрузка примеров
    examples = load_examples(data_dir=args.data_dir)

    # удаление рёбер, если они есть. иначе будет феил при сохранении предиктов
    for x in examples:
        x.arcs.clear()

    # кодирование примеров
    example_encoder = ExampleEncoder.load(encoder_dir=args.model_dir)

    examples_encoded = example_encoder.transform(examples)

    assert all(x.filename is not None for x in examples_encoded)

    # print("saving predictions")
    # id2relation = {v: k for k, v in example_encoder.vocab_re.encodings.items()}
    # save_predictions(examples=examples_encoded, output_dir=args.output_dir, id2relation=id2relation)

    # подгрузка конфига
    config = json.load(open(os.path.join(args.model_dir, "config.json")))

    # создание модели + подгрузка весов
    tf.reset_default_graph()
    sess = tf.Session()
    model = RelationExtractor(sess, config)
    model.build()
    model.restore(model_dir=args.model_dir)
    model.initialize()

    # нет смысла искать рёбра у графа без вершин
    examples_filtered = [x for x in examples_encoded if len(x.entities) > 0]

    def check_entities_spans():
        for x in examples_filtered:
            for entity in x.entities:
                actual = ' '.join(x.tokens[entity.start_token_id:entity.end_token_id + 1])
                expected = ' '.join(entity.tokens)
                assert actual == expected
                assert entity.start_token_id > 0

    print("checking examples...")
    check_entities_spans()
    print("OK")

    # рёбра пишутся в сразу в инстансы классов Example
    model.predict(examples_filtered, batch_size=args.batch_size)

    def check_arcs():
        """
        каждое ребро должно иметь уникальный айдишник
        """
        from collections import defaultdict

        d_set = defaultdict(set)
        d_int = defaultdict(int)

        for x in examples_filtered:
            assert x.filename is not None
            for arc in x.arcs:
                d_set[x.filename].add(arc.id)
                d_int[x.filename] += 1
        for x in examples_filtered:
            assert len(d_set[x.filename]) == d_int[x.filename], f"id: {x.id}, filename: {x.filename}: {len(d_set[x.filename])} != {d_int[x.filename]}"

    check_arcs()

    print("saving predictions")
    id2relation = {v: k for k, v in example_encoder.vocab_re.encodings.items()}
    save_predictions(examples=examples_filtered, output_dir=args.output_dir, id2relation=id2relation)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--batch_size", type=int, default=32, required=False)

    args = parser.parse_args()
    print(args)

    main(args)
