import os
from argparse import ArgumentParser


# TODO: оценка с учётом предсказанных ролей
# TODO: выводить classification report
# TODO: оценка качества поиска отношений (R)
def main(args):
    def get_triples(data_dir):
        """
        получечие троек (имя файла, триггер, аргумент)
        """
        s = set()
        for file in os.listdir(data_dir):
            if file.endswith(".ann"):
                with open(os.path.join(data_dir, file)) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("E"):
                            id_event, content = line.split("\t")
                            event_args = content.split()
                            trigger = event_args.pop(0)
                            for arg in event_args:
                                role, id_arg = arg.split(":")
                                s.add((file, trigger, id_arg))
        return s

    y_true = get_triples(data_dir=args.answers_dir)
    y_pred = get_triples(data_dir=args.predictions_dir)

    # print("y_true - y_pred:", y_true - y_pred)
    # print("y_pred - y_true:", y_pred - y_true)
    
    tp = len(y_true & y_pred)
    precision = tp / len(y_pred)
    recall = tp / len(y_true)
    f1 = 2 * precision * recall / (precision + recall)

    print("EVENTS METRICS:")
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--answers_dir")
    parser.add_argument("--predictions_dir")
    args = parser.parse_args()

    main(args)
