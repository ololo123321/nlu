import os
from collections import defaultdict
from argparse import ArgumentParser
from src.preprocessing import Arc, Event, EventArgument, ExamplesLoader, NerEncodings


def predict_and_save_baseline(examples, output_dir):
    """
    Бейзлайн:
    * если в предложении с событием есть компании, то привязать их все к событию
    * иначе, привязат к событию ближайшую компанию слева.

    ВАЖНО: примеры должны быть на уровне предложений!
    """
    event_counter = defaultdict(int)
    arc_counter = defaultdict(int)

    def get_event_id(filename):
        idx = event_counter[filename]
        event_counter[filename] += 1
        return idx

    def get_arc_id(filename):
        idx = arc_counter[filename]
        arc_counter[filename] += 1
        return idx

    id2example = {x.id: x for x in examples}
    ORG = "ORG"
    BANKRUPT = "Bankrupt"
    for x in examples:
        with open(os.path.join(output_dir, f"{x.filename}.ann"), "a") as f:
            events = {}
            org_ids = set()
            # исходные сущности
            for entity in x.entities:
                line = f"{entity.id}\t{entity.label} {entity.start_index} {entity.end_index}\t{entity.text}\n"
                f.write(line)
                if entity.is_event_trigger:
                    if entity.id not in events:
                        id_event = get_event_id(x.filename)
                        events[entity.id] = Event(
                            id=id_event,
                            id_trigger=entity.id,
                            label=entity.label,
                            arguments=None,
                        )
                elif entity.label == ORG:
                    org_ids.add(entity.id)

            # добавим рёбра
            if len(org_ids) > 0:
                for trigger in events.keys():
                    for id_entity in org_ids:
                        id_arc = get_arc_id(x.filename)
                        arc = Arc(
                            id=id_arc,
                            head=trigger,
                            dep=id_entity,
                            rel=BANKRUPT
                        )
                        x.arcs.append(arc)
            else:
                id_sent = int(x.id.split("_")[-1])
                if id_sent >= 1:
                    id_org_nearest = None
                    for j in range(id_sent - 1, -1, -1):
                        x_prev = id2example[f"{x.filename}_{j}"]
                        for entity in sorted(x_prev.entities, key=lambda e: e.end_index, reverse=True):
                            if entity.label == ORG:
                                id_org_nearest = entity.id
                                break
                        if id_org_nearest is not None:
                            for trigger in events.keys():
                                id_arc = get_arc_id(x.filename)
                                arc = Arc(
                                    id=id_arc,
                                    head=trigger,
                                    dep=id_org_nearest,
                                    rel=BANKRUPT
                                )
                                x.arcs.append(arc)
                            break

            # отношения
            for arc in x.arcs:
                if arc.head in events.keys():
                    arg = EventArgument(id=arc.dep, role=arc.rel)
                    events[arc.head].arguments.add(arg)
                else:
                    line = f"R{arc.id}\t{arc.rel} Arg1:{arc.head} Arg2:{arc.dep}\n"
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
    loader = ExamplesLoader(
        ner_encoding=NerEncodings.BILOU,
        ner_prefix_joiner="-",
        fix_new_line_symbol=True
    )
    examples = loader.load_examples(data_dir=args.data_dir, n=None, split=True, window=1)

    predict_and_save_baseline(examples=examples, output_dir=args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")

    _args = parser.parse_args()
    main(_args)
