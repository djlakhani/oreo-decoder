"""Errors raised by the Wikidata5M storage subsystem."""


class Wikidata5MError(Exception):
    """Base class for Wikidata5M storage failures."""


class IndexBuildError(Wikidata5MError):
    """The index could not be built safely."""


class MalformedSourceError(IndexBuildError):
    """A source dataset row does not satisfy the expected format."""


class CorruptIndexError(Wikidata5MError):
    """An index artifact is missing, truncated, or internally inconsistent."""


class StaleIndexError(Wikidata5MError):
    """The indexed source files have changed since the build."""


class UnknownEntityError(Wikidata5MError, KeyError):
    """A QID or dense entity ID is not present in the index."""


class UnknownRelationError(Wikidata5MError, KeyError):
    """A PID or dense relation ID is not present in the index."""


class MissingDataError(Wikidata5MError):
    """A known identifier has no requested passage or alias payload."""
