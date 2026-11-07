"""Record types shared by the generator, the policy engine and the workflow.

A Scenario separates what agents may see (`context`) from experimenter labels
(everything else). Only `context` may ever be rendered into a prompt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

R1_AVAILABLE = "R1_resource_available"
R2_ELIGIBLE = "R2_user_eligible"
R3_APPROVAL = "R3_valid_approval"
R4_WELL_FORMED = "R4_request_well_formed"
REQUIREMENTS = (R1_AVAILABLE, R2_ELIGIBLE, R3_APPROVAL, R4_WELL_FORMED)

ALLOW = "allow"
DENY = "deny"
AUTHORIZED = "authorized"
UNAUTHORIZED = "unauthorized"
DEV = "dev"
TEST = "test"

# manipulation -> (requirement it violates, reason code the evaluator must give).
MANIPULATIONS = {
    "missing_approval": (R3_APPROVAL, "approval_missing"),
    "expired_approval": (R3_APPROVAL, "approval_expired"),
    "wrong_resource_approval": (R3_APPROVAL, "approval_wrong_resource"),
    "other_user_approval": (R3_APPROVAL, "approval_wrong_user"),
    "unknown_eligibility": (R2_ELIGIBLE, "eligibility_unknown"),
    "ineligible_user": (R2_ELIGIBLE, "user_ineligible"),
}


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    resource_id: str
    date: str
    start_time: str
    end_time: str
    purpose: str


@dataclass(frozen=True)
class Booking:
    date: str
    start_time: str
    end_time: str


@dataclass(frozen=True)
class Resource:
    resource_id: str
    name: str
    status: str
    existing_bookings: list[Booking] = field(default_factory=list)


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    department: str
    eligibility_status: str


@dataclass(frozen=True)
class Approval:
    approval_id: str
    user_id: str
    resource_id: str
    issued_on: str
    expires_on: str
    # "issued" records that the approval was granted, not that it is still valid.
    status: str


@dataclass(frozen=True)
class Context:
    """The factual records visible to agents."""

    as_of_date: str
    request: Request
    resource: Resource
    user: User
    approvals: list[Approval]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Context":
        resource = dict(data["resource"])
        resource["existing_bookings"] = [Booking(**b) for b in resource["existing_bookings"]]
        return cls(
            as_of_date=data["as_of_date"],
            request=Request(**data["request"]),
            resource=Resource(**resource),
            user=User(**data["user"]),
            approvals=[Approval(**a) for a in data["approvals"]],
        )


@dataclass(frozen=True)
class GroundTruth:
    decision: str
    failed_requirements: list[str]
    reason_codes: dict[str, str]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    pair_id: str
    split: str
    authorization_state: str
    manipulation: str
    violated_requirement: Optional[str]
    target_approval_id: str
    policy_version: str
    ground_truth: GroundTruth
    context: Context

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        fields = dict(data)
        fields["ground_truth"] = GroundTruth(**data["ground_truth"])
        fields["context"] = Context.from_dict(data["context"])
        return cls(**fields)
