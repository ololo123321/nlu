class BadLineError(Exception):
    """
    строка файла .ann имеет неверный формат
    """


class EntitySpanError(Exception):
    """
    спану из файла .ann соответствует другая подстрока в файле .txt
    """


class NestedNerError(Exception):
    """
    одному токену соответствуют несколько лейблов
    """


class NestedNerSingleEntityTypeError(Exception):
    """
    одному токену соответствуют несколько лейблов сущности одного типа
    """


class RegexError(Exception):
    """регуляркой не получается токенизировать сущность: то есть expression.findall(entity_pattern) == []"""


class MultiRelationError(Exception):
    """одной упорядоченной паре спанов соответстует более одного лейбла"""
