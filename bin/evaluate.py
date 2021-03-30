import os
import re
from argparse import ArgumentParser


# TODO: NER
# TODO: event and entities attrs


def compute_metrics(y_true: set, y_pred: set):
    tp = len(y_true & y_pred)

    if len(y_pred) > 0:
        precision = tp / len(y_pred)
    else:
        precision = 0.0

    if len(y_true) > 0:
        recall = tp / len(y_true)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)


def main(args):
    def get_edges(data_dir):
        """
        получечие троек (имя файла, триггер, аргумент)
        """
        events = set()
        relations = set()
        for file in os.listdir(data_dir):
            if file.endswith(".ann"):
                with open(os.path.join(data_dir, file)) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("E"):
                            # E0\tBankruptcy:T6 Bankrupt:T16 Bankrupt2:T17
                            id_event, content = line.split("\t")
                            event_args = content.split()
                            trigger = event_args.pop(0)
                            for arg in event_args:
                                role, id_arg = arg.split(":")
                                role = re.sub(r'\d+', '', role)  # Bankrupt2 -> Bankrupt
                                events.add((file, trigger, id_arg, role))
                        elif line.startswith("R"):
                            # R1\tCoreference Arg1:T16 Arg2:T7
                            id_rel, content = line.split("\t")
                            rel, arg1, arg2 = content.split()
                            head = arg1.split(":")[1]
                            dep = arg2.split(":")[1]
                            relations.add((file, rel, head, dep))
        return events, relations

    events_true, relations_true = get_edges(data_dir=args.answers_dir)
    events_pred, relations_pred = get_edges(data_dir=args.predictions_dir)

    print("EVENTS METRICS:")
    compute_metrics(y_true=events_true, y_pred=events_pred)

    print("\nRELATIONS METRICS:")
    compute_metrics(y_true=relations_true, y_pred=relations_pred)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--answers_dir")
    parser.add_argument("--predictions_dir")
    _args = parser.parse_args()

    main(_args)
