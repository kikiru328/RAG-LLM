from dataclasses import dataclass

@dataclass
class FilterResult:
    passed: bool
    reason: str | None = None
    response: str | None = None
    rule_type: str | None = None
    