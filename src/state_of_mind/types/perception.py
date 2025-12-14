from typing import NamedTuple, Any, Optional, Callable


class ValidationRule(NamedTuple):
    path: str
    required: bool
    validator: Any
    description: str
    value_checker: Optional[Callable[[Any], bool]] = None
