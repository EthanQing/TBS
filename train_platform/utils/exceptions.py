from __future__ import annotations


class ValidationError(ValueError):
    pass


class NotFoundError(ValueError):
    pass


class ConflictError(ValueError):
    pass
