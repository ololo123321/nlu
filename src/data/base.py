import re
from typing import Union, List
from collections import namedtuple


TOKENS_EXPRESSION = re.compile(r"\w+|[^\w\s]")


class SpecialSymbols:
    CLS = '[CLS]'
    SEP = '[SEP]'
    START_HEAD = '[START_HEAD]'
    END_HEAD = '[END_HEAD]'
    START_DEP = '[START_DEP]'
    END_DEP = '[END_DEP]'


class BertEncodings:
    TEXT = "text"
    NER = "ner"
    TEXT_NER = "text_ner"
    NER_TEXT = "ner_text"


class NerEncodings:
    BIO = "bio"
    BILOU = "bilou"


class NerPrefixJoiners:
    UNDERSCORE = "_"  # WARNING: этот символ может встречаться в названиях сущностей/событий/отношений
    HYPHEN = "-"


class LineTypes:
    ENTITY = "T"
    EVENT = "E"
    RELATION = "R"
    ATTRIBUTE = "A"
    # https://brat.nlplab.org/standoff.html
    # For backward compatibility with existing standoff formats,
    # brat also recognizes the ID prefix "M" for attributes.
    ATTRIBUTE_OLD = "M"
    COMMENT = "#"
    EQUIV = "*"  # TODO: что это??


class Languages:
    EN = "en",
    RU = "ru"


# immutable structs

Attribute = namedtuple("Attribute", ["id", "type", "value"])
EventArgument = namedtuple("EventArgument", ["id", "role"])
Span = namedtuple("Span", ["start", "end"])
SpanExtended = namedtuple("Span", ["start", "end", "label", "score"])


# mutable structs


class ReprMixin:
    def __repr__(self):
        class_name = self.__class__.__name__
        params_str = ', '.join(f"{k}={v}" for k, v in self.__dict__.items())
        return f'{class_name}({params_str})'


class Token(ReprMixin):
    def __init__(
            self,
            text: str = None,
            span_abs: Span = None,
            span_rel: Span = None,
            index_abs: int = None,
            index_rel: int = None,
            labels: List[Union[str, int]] = None,
            pieces: List[Union[str, int]] = None,  # TODO: нужно ли?
            token_ids: List[int] = None,
            id_sent: int = None,
            id_head: int = None,  # for dependency parsing
            rel: str = None,  # for dependency parsing
            pos: str = None  # part of speech, пока не надо
    ):
        """

        :param text: текст
        :param span_abs: абсолютный* спан
        :param span_rel: относительный** спан
        :param index_abs: абсолютный* порядковый номер
        :param index_rel: относительный** порядковый номер
        :param labels: лейблы
        :param pieces: bpe-кусочки

        * на уровне документа
        ** на уровне примера
        """
        self.text = text
        self.span_abs = span_abs
        self.span_rel = span_rel
        self.index_abs = index_abs
        self.index_rel = index_rel  # пока не нужно
        self.labels = labels if labels is not None else []
        self.pieces = pieces if pieces is not None else []
        self.token_ids = token_ids if token_ids is not None else []
        self.id_sent = id_sent
        self.id_head = id_head
        self.rel = rel
        self.pos = pos

        self.labels_pieces = []
        self.label_ids = []

    @property
    def copy(self):
        t = Token()
        for attr, value in self.__dict__.items():
            if isinstance(value, list):
                setattr(t, attr, value.copy())
            else:
                setattr(t, attr, value)
        return t

    def reset(self):
        self.labels = []
        self.id_head = None
        self.rel = None
        self.pos = None
        self.labels_pieces = []
        self.label_ids = []


class Entity(ReprMixin):
    def __init__(
            self,
            id: Union[int, str] = None,
            label: Union[int, str] = None,
            text: str = None,
            tokens: List[Token] = None,
            is_event_trigger: bool = False,
            attrs: List[Attribute] = None,  # атрибуты сущности
            comment: str = None,
            index: int = None,
            id_chain: int = None  # для coreference resolution
    ):
        """

        :param id:
        :param label:
        :param text:
        :param tokens:
        :param is_event_trigger:
        :param attrs:
        :param comment:
        """
        self.id = id
        self.label = label
        self.text = text
        self.tokens = tokens if tokens is not None else []
        self.is_event_trigger = is_event_trigger
        self.attrs = attrs if attrs is not None else []
        self.comment = comment
        self.index = index
        self.id_chain = id_chain

        self.label_id = None  # deprecated


class Event(ReprMixin):
    def __init__(
            self,
            id: Union[str, int] = None,
            trigger: str = None,
            label: str = None,
            args: List[EventArgument] = None,
            attrs: List[Attribute] = None,
            comment: str = None
    ):
        self.id = id
        self.trigger = trigger
        self.label = label
        self.args = args if args is not None else []
        self.attrs = attrs if attrs is not None else []
        self.comment = comment


class Arc(ReprMixin):
    def __init__(
            self,
            id: Union[str, int],
            head: Union[str, int],  # int in case of dependency parsing
            dep: Union[str, int],  # int in case of dependency parsing
            rel: str,
            comment: str = None,
            score: float = None  # мб пока не нужно
    ):
        self.id = id
        self.head = head
        self.dep = dep
        self.rel = rel
        self.comment = comment
        self.score = score

        self.rel_id = None
        self.head_index = None
        self.dep_index = None


class Example(ReprMixin):
    def __init__(
            self,
            filename: str = None,
            id: str = None,
            text: str = None,
            tokens: List[Token] = None,
            entities: List[Entity] = None,
            arcs: List[Arc] = None,
            events: List[Event] = None,  # пока только для дебага
            label: int = None,
            parent: str = None,
            chunks: List = None
    ):
        self.filename = filename
        self.id = id
        self.text = text
        self.tokens = tokens if tokens is not None else []
        self.entities = entities if entities is not None else []
        self.arcs = arcs if arcs is not None else []
        self.events = events if events is not None else []
        self.label = label  # в случае классификации предложений
        self.parent = parent
        self.chunks = chunks if chunks is not None else []  # список инстансов класса Example
