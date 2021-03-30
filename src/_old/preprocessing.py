# check

# # TODO: адаптировать
# def check_example(example: Example, ner_encoding: str = NerEncodings.BIO):
#     """
#     NER:
#     * число токенов равно числу лейблов
#     * entity.start >= entity.end
#     * начало сущности >= 0, конец сущности < len(tokens)
#     RE:
#     * оба аргумента отношений есть в entities
#     """
#     # обязателен айдишник
#     assert example.id is not None, f"example {example} has no id!"
#     prefix = f"[{example.id}]: "
#
#     # ner-кодировка
#     assert ner_encoding in {NerEncodings.BIO, NerEncodings.BILOU}, \
#         f"expected ner_encoding {NerEncodings.BIO} or {NerEncodings.BILOU}, got {ner_encoding}"
#
#     num_tokens = len(example.tokens)
#
#     # # биекция между токенами и лейблами
#     # assert num_tokens == len(example.labels), \
#     #     prefix + f"tokens and labels mismatch, {num_tokens} != {len(example.labels)}"
#
#     entity_ids_all = set()
#     entity_ids_wo_events = set()
#     entity_spans = set()
#     event2entities = defaultdict(set)
#     for entity in example.entities:
#         # обязателен айдишник
#         assert entity.id is not None, \
#             prefix + f"[{entity}] entity has no id!"
#
#         # проверка валидности спана
#         assert 0 <= entity.start_token_id <= entity.end_token_id < num_tokens, \
#             prefix + f"[{entity}] strange entity span: " \
#                 f"start token id: {entity.start_token_id}, end token id: {entity.end_token_id}. num tokens: {num_tokens}"
#
#         # проверка корректности соответстия токенов сущности токенам примера
#         expected_tokens = example.tokens[entity.start_token_id:entity.end_token_id + 1]
#         assert expected_tokens == entity.tokens, \
#             prefix + f"[{entity}] tokens and example tokens mismatch: {entity.tokens} != {expected_tokens}"
#
#         # # проверка корректности соответстия лейблов сущности лейблам примера
#         # if entity.is_event_trigger:
#         #     ner_labels = example.labels_events[entity.label]
#         # else:
#         #     ner_labels = example.labels
#         # expected_labels = ner_labels[entity.start_token_id:entity.end_token_id + 1]
#         # assert expected_labels == entity.labels, \
#         #     prefix + f"[{entity}]: labels and example labels mismatch: {entity.labels} != {expected_labels}"
#
#         # кэш
#         entity_ids_all.add(entity.id)
#         entity_spans.add((entity.start_token_id, entity.end_token_id))
#         if entity.is_event_trigger:
#             event2entities[entity.label].add(entity.id)
#         else:
#             entity_ids_wo_events.add(entity.id)
#
#     # проверка уникальности сущностей
#     assert len(example.entities) == len(entity_ids_all), \
#         prefix + f"entity ids are not unique: {len(example.entities)} != {len(entity_ids_all)}"
#
#     # проверка биекции между множеством спанов и множеством сущностей.
#     # пока предполагается её наличие.
#     # @ каждому событию соответствет своя сущность;
#     # @ каждой сущности соответствует свой спан;
#     # @ если одному паттерну соответствуют несколько событий, то ему могут соответствовать несколько спанов
#     # @ пока такое возможно
#     # assert len(example.entities) == len(entity_spans), \
#     #     prefix + f"there are span duplicates: " \
#     #     f"number of entities is {len(example.entities)}, but number of unique text spans is {len(entity_spans)}"
#
#     def check_ner_labels(ent_ids, labels, ner_label_other):
#         """проверка непротиворечивости множества сущностей лейблам"""
#         if len(ent_ids) == 0:
#             assert set(labels) == {ner_label_other}, \
#                 prefix + f"ner labels and named entities mismatch: ner labels are {set(labels)}, " \
#                 f"but there are no entities in example."
#         else:
#             assert set(labels) != {ner_label_other}, \
#                 prefix + f"ner labels and named entities mismatch: ner labels are {set(labels)}, " \
#                 f"but there are following entities in example: {ent_ids}"
#
#     # check_ner_labels(ent_ids=entity_ids_wo_events, labels=example.labels, ner_label_other=self.ner_label_other)
#     #
#     # for k, v in event2entities.items():
#     #     check_ner_labels(ent_ids=v, labels=example.labels_events[k], ner_label_other=self.ner_label_other)
#
#     arc_args = []
#     for arc in example.arcs:
#         # проверка того, что в примере есть исходящая вершина
#         assert arc.head in entity_ids_all, \
#             prefix + f"something is wrong with arc {arc.id}: head {arc.head} is unknown"
#         # проверка того, что в примере есть входящая вершина
#         assert arc.dep in entity_ids_all, \
#             prefix + f"something is wrong with arc {arc.id}: dep {arc.dep} is unknown"
#         arc_args.append((arc.head, arc.dep))
#     # проверка того, что одному ребру соответствует одно отношение
#     if len(arc_args) != len(set(arc_args)):
#         arc_counts = {k: v for k, v in Counter(arc_args).items() if v > 1}
#         raise AssertionError(prefix + f'there duplicates in arc args: {arc_counts}')
#
#     num_triggers = sum(entity.is_event_trigger for entity in example.entities)
#     num_events = len(example.events)
#     assert num_events == num_triggers, \
#         prefix + f"number of event triggers is {num_triggers}, but number of events is {num_events}"
#
#     if ner_encoding == NerEncodings.BILOU:
#         # проверка того, что число начал сущности равно числу концов
#         num_start_ids = sum(x.startswith("B") for x in example.labels)
#         num_end_ids = sum(x.startswith("L") for x in example.labels)
#         assert num_start_ids == num_end_ids, \
#             prefix + f"num start ids: {num_start_ids}, num end ids: {num_end_ids}"


# def _check_ner(example: Example):
#     """
#     1. число токенов = число лейблов
#     2. индекс начала сущности <= индекс конца сущности
#     3. символ начала сущности <= символ конца сущности
#     4. entity.text == example.text[entity.span_rel[0]:entity.stan_rel[1]]
#     """
#     for entity in example.entities:
#         pass


# encoding


class Vocab:
    def __init__(self, values):
        if isinstance(values, dict):  # str -> int
            self._value2id = values
            self._id2value = {v: k for k, v in values.items()}
        elif isinstance(values, set):
            special_value = "O"
            values -= {special_value}
            self._id2value = dict(enumerate(sorted(values), 1))
            self._id2value[0] = special_value
            self._value2id = {v: k for k, v in self._id2value.items()}
        else:
            raise

    @property
    def size(self):
        return len(self._id2value)

    @property
    def encodings(self):
        return self._value2id

    @property
    def inv_encodings(self):
        return self._id2value

    def get_value(self, id):
        return self._id2value[id]

    def get_id(self, value):
        return self._value2id[value]


# class ExampleEncoder:
#     def __init__(
#             self,
#             ner_encoding: str = NerEncodings.BIO,
#             ner_label_other: str = "O",
#             re_label_other: str = "O",
#             ner_prefix_joiner: str = NerPrefixJoiners.HYPHEN,
#             add_seq_bounds: bool = True
#     ):
#         assert ner_encoding in {NerEncodings.BIO, NerEncodings.BILOU}
#         self.ner_encoding = ner_encoding
#         self.ner_label_other = ner_label_other
#         self.re_label_other = re_label_other
#         self.add_seq_bounds = add_seq_bounds
#         self.ner_prefix_joiner = ner_prefix_joiner
#
#         self.vocab_ner = None
#         self.vocab_re = None
#         self.vocabs_events = {}
#
#     def fit_transform(self, examples):
#         self.fit(examples)
#         return self.transform(examples)
#
#     def fit(self, examples):
#         # инициализация значений словаря
#         vocab_ner = set()
#         vocabs_events = defaultdict(set)
#
#         prefixes = {"B", "I"}
#         if self.ner_encoding == NerEncodings.BILOU:
#             prefixes |= {"L", "U"}
#
#         def extend_vocab(label_, ner_prefix_joiner, vocab_values):
#             if ner_prefix_joiner in label_:
#                 # предполагаем, что каждая сущность может состоять из нескольких токенов
#                 label_ = label_.split(ner_prefix_joiner)[-1]
#                 for p in prefixes:
#                     vocab_values.add(p + ner_prefix_joiner + label_)
#             else:
#                 vocab_values.add(label_)
#
#         for x in examples:
#             for label in x.labels:
#                 extend_vocab(label, self.ner_prefix_joiner, vocab_ner)
#             for event_tag, labels in x.labels_events.items():
#                 for label in labels:
#                     extend_vocab(label, self.ner_prefix_joiner, vocabs_events[event_tag])
#
#         vocab_ner.add(self.ner_label_other)
#         self.vocab_ner = Vocab(vocab_ner)
#
#         self.vocabs_events = {}
#         for k, v in vocabs_events.items():
#             v.add(self.ner_label_other)
#             self.vocabs_events[k] = Vocab(v)
#
#         # arcs vocab
#         vocab_re = set()
#         for x in examples:
#             for arc in x.arcs:
#                 vocab_re.add(arc.rel)
#         vocab_re.add(self.re_label_other)
#         self.vocab_re = Vocab(vocab_re)
#
#     def transform(self, examples: List[Example]) -> List[Example]:
#         res = []
#         for x in examples:
#             x_enc = self.transform_example(x)
#             res.append(x_enc)
#         return res
#
#     def transform_example(self, example: Example) -> Example:
#         """
#         Кодирование категориальных атрибутов примеров:
#         * tokens - List[str] (остаётся неизменным)
#         * labels - List[int]
#         * entities - List[Tuple[start, end]]
#         * arcs - List[Tuple[head, dep, id_relation]]
#         """
#         example_enc = Example(
#             filename=example.filename,
#             id=example.id,
#             text=example.text
#         )
#
#         # tokens
#         example_enc.tokens = example.tokens.copy()
#         if self.add_seq_bounds:
#             example_enc.tokens = ["[START]"] + example_enc.tokens + ["[END]"]
#
#         # tokens spans
#         example_enc.tokens_spans = example.tokens_spans.copy()
#         if self.add_seq_bounds:
#             example_enc.tokens_spans = [(-1, -1)] + example_enc.tokens_spans + [(-1, -1)]  # TODO: ок ли так делать?
#
#         # labels
#         def encode_labels(labels, vocab, add_seq_bounds, ner_label_other):
#             labels_encoded = []
#             for label in labels:
#                 label_enc = vocab.get_id(label)
#                 labels_encoded.append(label_enc)
#             if add_seq_bounds:
#                 label = vocab.get_id(ner_label_other)
#                 labels_encoded = [label] + labels_encoded + [label]
#             # example_enc.labels = labels_encoded
#             return labels_encoded
#
#         example_enc.labels = encode_labels(
#             labels=example.labels, vocab=self.vocab_ner,
#             add_seq_bounds=self.add_seq_bounds, ner_label_other=self.ner_label_other
#         )
#         example_enc.labels_events = {}
#         for k, v in example.labels_events.items():
#             example_enc.labels_events[k] = encode_labels(
#                 labels=v, vocab=self.vocabs_events[k],
#                 add_seq_bounds=self.add_seq_bounds, ner_label_other=self.ner_label_other
#             )
#
#         # entities
#         example_enc.entities = deepcopy(example.entities)
#         if self.add_seq_bounds:
#             # потому что в начало добавлен токен начала строки
#             for entity in example_enc.entities:
#                 entity.start_token_id += 1
#                 entity.end_token_id += 1
#
#         # arcs
#         arcs_encoded = []
#         for arc in example.arcs:
#             id_rel = self.vocab_re.get_id(arc.rel)
#             arc_enc = Arc(id=arc.id, head=arc.head, dep=arc.dep, rel=id_rel)
#             arcs_encoded.append(arc_enc)
#         example_enc.arcs = arcs_encoded
#         return example_enc
#
#     def save(self, encoder_dir):
#         d = {
#             "ner_encoding": self.ner_encoding,
#             "ner_label_other": self.ner_label_other,
#             "re_label_other": self.re_label_other,
#             "ner_prefix_joiner": self.ner_prefix_joiner,
#             "add_seq_bounds": self.add_seq_bounds
#         }
#         with open(os.path.join(encoder_dir, "encoder_config.json"), "w") as f:
#             json.dump(d, f, indent=4)
#
#         with open(os.path.join(encoder_dir, "ner_encodings.json"), "w") as f:
#             json.dump(self.vocab_ner.encodings, f, indent=4)
#
#         with open(os.path.join(encoder_dir, "ner_encodings_events.json"), "w") as f:
#             json.dump({k: v.encodings for k, v in self.vocabs_events.items()}, f, indent=4)
#
#         with open(os.path.join(encoder_dir, "re_encodings.json"), "w") as f:
#             json.dump(self.vocab_re.encodings, f, indent=4)
#
#     @classmethod
#     def load(cls, encoder_dir):
#         config = json.load(open(os.path.join(encoder_dir, "encoder_config.json")))
#         enc = cls(**config)
#
#         ner_encodings = json.load(open(os.path.join(encoder_dir, "ner_encodings.json")))
#         enc.vocab_ner = Vocab(values=ner_encodings)
#
#         re_encodings = json.load(open(os.path.join(encoder_dir, "re_encodings.json")))
#         enc.vocab_re = Vocab(values=re_encodings)
#
#         d = json.load(open(os.path.join(encoder_dir, "ner_encodings_events.json")))
#         enc.vocabs_events = {k: Vocab(values=v) for k, v in d.items()}
#
#         return enc

# def change_tokens_and_entities(x: Example) -> Example:
#     """
#     tokens = [иван иванов живёт в деревне жопа]
#     labels = [B_PER I_PER O O O B_LOC]
#     entities = [
#         Entity(tokens=[иван иванов], labels=[B_PER, I_PER], start_token_id=0, end_token_id=2),
#         Entity(tokens=[жопа], labels=[B_LOC], start_token_id=5, end_token_id=5),
#     ]
#
#
#     tokens = [PER живёт в деревне LOC]
#     labels = [B_PER I_PER O O O B_LOC]
#     entities = [
#         Entity(tokens=[иван иванов], labels=[B_PER, I_PER], start_token_id=0, end_token_id=0),
#         Entity(tokens=[жопа], labels=[B_LOC], start_token_id=4, end_token_id=4),
#     ]
#     """
#     x_new = deepcopy(x)
#     if x_new.entities:
#         entities_sorted = sorted(x_new.entities, key=lambda e: e.start_token_id)
#         pointers = [0]
#         tokens_new = []
#         for i, entity in enumerate(entities_sorted, 1):
#             end_prev = pointers[i - 1]
#             start_curr = entity.start_token_id
#             tokens_new += x.tokens[end_prev:start_curr]
#             label = '[{}]'.format(entity.labels[0].split('-')[1])
#             tokens_new.append(label)
#             start_new = end_new = len(tokens_new) - 1
#             end_curr = entity.end_token_id
#             pointers.append(end_curr + 1)
#             if i == len(entities_sorted):
#                 start = entity.end_token_id + 1
#                 end = len(x.tokens)
#                 tokens_new += x.tokens[start:end]
#             entity.start_token_id = start_new
#             entity.end_token_id = end_new
#         x_new.tokens = tokens_new
#     return x_new
#
#
# def convert_example_for_bert(x: Example, tokenizer, tag2token: dict, mode: str, no_rel_id: int = 0) -> List[Example]:
#     """
#     https://github.com/facebookresearch/SpanBERT/blob/10641ea3795771dd96e9e3e9ef0ead4f4f6a29d2/code/run_tacred.py#L116
#
#     tokens = [иван иванов живёт в деревне жопа]
#     labels = [B_PER I_PER O O O B_LOC]
#     entities = [
#         Entity(id=T1, tokens=[иван иванов], labels=[B_PER, I_PER], start_token_id=0, end_token_id=2),
#         Entity(id=T2, tokens=[жопа], labels=[B_LOC], start_token_id=5, end_token_id=5),
#     ]
#     arc = [ARC(id=R1, head=T1, dep=T2, rel=3)]
#
#     # [CLS] <контекст> [START_{HEAD или DEP}] <токены HEAD или DEP> [END_{HEAD или DEP}]
#     # <контекст> [START_{DEP или HEAD}] <токены DEP или HEAD> [END_{DEP или HEAD}] <контекст>
#     # [SEP] [таг левого операнда отношения (head)] [SEP] [таг правого операнда отношения (dep)] [SEP]
#     [
#         Example(
#             tokens = [
#                 [CLS] [START_HEAD] иван иванов [END_HEAD] живёт в деревне
#                 [START_DEP] жопа [END_DEP] [SEP] [HEAD-PER] [SEP] [DEP-LOC] [SEP]
#             ],
#             label = 3
#         ),
#         Example(
#             tokens = [
#                 [CLS] [START_DEP] иван иванов [END_DEP] живёт в деревне
#                 [START_HEAD] жопа [END_HEAD] [SEP] [HEAD-LOC] [SEP] [DEP-PER] [SEP]
#             ],
#             label = 0
#         )
#     ]
#
#     в tag2token должны быть токены берта для следующих спец. токенов:
#     1) '[START_HEAD]', '[END_HEAD]', '[START_DEP]', '[END_DEP]'
#     2) таги именных сущностей
#     """
#     assert mode in {BertEncodings.TEXT, BertEncodings.NER, BertEncodings.TEXT_NER, BertEncodings.NER_TEXT}
#
#     arc2rel = {}
#     for arc in x.arcs:
#         arc2rel[(arc.head, arc.dep)] = arc.rel
#
#     examples_new = []
#
#     START_HEAD_TOKEN = tag2token[SpecialSymbols.START_HEAD]
#     END_HEAD_TOKEN = tag2token[SpecialSymbols.END_HEAD]
#     START_DEP_TOKEN = tag2token[SpecialSymbols.START_DEP]
#     END_DEP_TOKEN = tag2token[SpecialSymbols.END_DEP]
#
#     token2pieces = {}
#
#     def get_pieces(token):
#         if token not in token2pieces:
#             token2pieces[token] = tokenizer.tokenize(token)
#         return token2pieces[token]
#
#     id_new = 0
#     num_entities = len(x.entities)
#     for id_head in range(num_entities):
#         for id_dep in range(num_entities):
#             if id_head == id_dep:
#                 continue
#             head = x.entities[id_head]
#             dep = x.entities[id_dep]
#
#             tag_head = "HEAD_" + head.labels[0].split('-')[1]
#             tag_dep = "DEP_" + dep.labels[0].split('-')[1]
#
#             TAG_HEAD_TOKEN = tag2token[tag_head]
#             TAG_DEP_TOKEN = tag2token[tag_dep]
#
#             tokens_new = [SpecialSymbols.CLS]
#
#             if mode in {BertEncodings.TEXT, BertEncodings.TEXT_NER}:
#                 # [HEAD_START] иван иванов [HEAD_END] живёт в деревне [DEP_START] жопа [DEP_END] [SEP]
#
#                 for i, pieces in enumerate(map(get_pieces, x.tokens)):
#
#                     if i == head.start_token_id:
#                         tokens_new.append(START_HEAD_TOKEN)
#                     if i == dep.start_token_id:
#                         tokens_new.append(START_DEP_TOKEN)
#
#                     tokens_new += pieces
#
#                     if i == head.end_token_id:
#                         tokens_new.append(END_HEAD_TOKEN)
#                     if i == dep.end_token_id:
#                         tokens_new.append(END_DEP_TOKEN)
#
#                 tokens_new.append(SpecialSymbols.SEP)
#
#                 if mode == BertEncodings.TEXT_NER:
#                     # + [HEAD_PER] [SEP] [DEP_LOC] [SEP]
#                     tokens_new += [TAG_HEAD_TOKEN, SpecialSymbols.SEP, TAG_DEP_TOKEN, SpecialSymbols.SEP]
#
#             else:
#                 # [HEAD_PER] живёт в деревне [DEP_LOC] [SEP]
#                 head_pieces = []
#                 dep_pieces = []
#
#                 for i, pieces in enumerate(map(get_pieces, x.tokens)):
#
#                     if i == head.start_token_id:
#                         tokens_new.append(TAG_HEAD_TOKEN)
#                     if i == dep.start_token_id:
#                         tokens_new.append(TAG_DEP_TOKEN)
#
#                     if head.start_token_id <= i <= head.end_token_id:
#                         head_pieces += pieces
#                     elif dep.start_token_id <= i <= dep.end_token_id:
#                         dep_pieces += pieces
#                     else:
#                         tokens_new += pieces
#
#                 tokens_new.append(SpecialSymbols.SEP)
#
#                 if mode == BertEncodings.NER_TEXT:
#                     # + [иван иванов [SEP] жопа [SEP]]
#                     tokens_new += head_pieces
#                     tokens_new.append(SpecialSymbols.SEP)
#                     tokens_new += dep_pieces
#                     tokens_new.append(SpecialSymbols.SEP)
#
#             token_ids = tokenizer.convert_tokens_to_ids(tokens_new)
#             rel = arc2rel.get((head.id, dep.id), no_rel_id)
#             # x.id = <название файла>_<порядковый номер предложения>
#             # id_new = <x.id>_<порядковый номер отношения>
#             x_new = Example(id=f'{x.id}_{id_new}', tokens=token_ids, label=rel)
#             examples_new.append(x_new)
#             id_new += 1
#     return examples_new
