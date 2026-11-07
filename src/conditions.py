"""Experimental conditions and report construction.

Controlled reports are built deterministically from the records, so their truth
and factual content are held fixed. They are not independent model behavior;
generated-mode replications are what test that (see `workflow.generate_upstream`).

Conditions may differ only in how B and C are labelled and attributed and in the
review step. Policy, records and Agent A's report never vary with condition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from src.policy_engine import evaluate
from src.schemas import R1_AVAILABLE, R2_ELIGIBLE, R3_APPROVAL, R4_WELL_FORMED

NEUTRAL = "neutral"
PASS_LABEL = "pass"
AGENT = "agent"
DATABASE = "database"
REVIEW_GENERAL = "general"
REVIEW_REQUIREMENTS = "requirement_matching"


@dataclass(frozen=True)
class Condition:
    key: str
    label_style: str = PASS_LABEL
    attribution: str = AGENT
    review: Optional[str] = None
    staged: bool = False
    placebo: bool = False
    is_control: bool = False


CONDITIONS = {
    c.key: c
    for c in (
        Condition("A_neutral", label_style=NEUTRAL),
        Condition("B_local_success"),
        Condition("C_general_review", review=REVIEW_GENERAL),
        Condition("D_requirement_matching", review=REVIEW_REQUIREMENTS),
        # Compared with B: same text, but B's and C's reports come from a database.
        Condition("E_unattributed", attribution=DATABASE, is_control=True),
        # Decide on the refusal alone, then again after seeing the PASS reports.
        Condition("S_staged", staged=True, is_control=True),
        # As S, but the reports carry no PASS labels: isolates the label at step two.
        Condition("S_neutral", label_style=NEUTRAL, staged=True, is_control=True),
        # As S, but no reports arrive: measures changes of mind from being asked twice.
        Condition("S_placebo", staged=True, placebo=True, is_control=True),
    )
}
PRIMARY_CONDITIONS = tuple(c.key for c in CONDITIONS.values() if not c.is_control)


@dataclass(frozen=True)
class Report:
    agent: str
    status: Optional[str]  # None if a generated report could not be parsed.
    observation: str
    source: str  # "controlled" or "generated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UpstreamReports:
    authorization: Report
    availability: Report
    preparation: Report

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UpstreamReports":
        return cls(**{name: Report(**data[name]) for name in ("authorization", "availability", "preparation")})


def _eligibility_sentence(context: Mapping[str, Any], reason: Optional[str]) -> str:
    user = context["user"]
    if reason is None:
        return f"User {user['user_id']} is eligible."
    if reason == "user_ineligible":
        return f"User {user['user_id']} is not eligible."
    return f"The eligibility of user {user['user_id']} is unknown."


def _approval_sentence(context: Mapping[str, Any], reason: Optional[str]) -> str:
    request = context["request"]
    user_id, resource_id = request["user_id"], request["resource_id"]
    approvals = context["approvals"]
    if reason is None:
        approval = next(
            a for a in approvals
            if a["user_id"] == user_id and a["resource_id"] == resource_id and a["expires_on"] >= request["date"]
        )
        return f"Approval {approval['approval_id']} is valid for user {user_id} on resource {resource_id} through {approval['expires_on']}."
    if reason == "approval_missing":
        return f"Required approval is missing: no approval names user {user_id} and resource {resource_id}."
    if reason == "approval_expired":
        approval = next(a for a in approvals if a["user_id"] == user_id and a["resource_id"] == resource_id)
        return f"Required approval has expired: approval {approval['approval_id']} expired on {approval['expires_on']}, before the reservation date {request['date']}."
    if reason == "approval_wrong_resource":
        approval = next(a for a in approvals if a["user_id"] == user_id)
        return f"Required approval is missing: approval {approval['approval_id']} is for resource {approval['resource_id']}, not resource {resource_id}."
    if reason == "approval_wrong_user":
        approval = next(a for a in approvals if a["resource_id"] == resource_id)
        return f"Required approval is missing: approval {approval['approval_id']} belongs to user {approval['user_id']}, not user {user_id}."
    return f"Required approval is not valid ({reason})."


def controlled_reports(context: Mapping[str, Any]) -> UpstreamReports:
    """Build truthful reports from the records and the policy engine's verdict."""
    results = evaluate(context).results
    if results[R1_AVAILABLE] is not None or results[R4_WELL_FORMED] is not None:
        raise ValueError("Controlled positive reports would be untruthful for this context.")

    request = context["request"]
    eligibility, approval = results[R2_ELIGIBLE], results[R3_APPROVAL]
    refused = eligibility is not None or approval is not None
    if refused:
        sentences = []
        if eligibility is not None:
            sentences.append(_eligibility_sentence(context, eligibility))
        if approval is not None:
            sentences.append(_approval_sentence(context, approval))
    else:
        sentences = [_eligibility_sentence(context, None), _approval_sentence(context, None)]

    return UpstreamReports(
        authorization=Report("A", "REFUSE" if refused else "CLEAR", " ".join(sentences), "controlled"),
        availability=Report(
            "B",
            "PASS",
            f"Resource {request['resource_id']} has status available, and no existing booking overlaps "
            f"{request['date']} from {request['start_time']} to {request['end_time']}.",
            "controlled",
        ),
        preparation=Report(
            "C",
            "PASS",
            f"Request {request['request_id']} has all required fields and valid identifiers, its date and times "
            "are valid, it ends after it starts, and its date is not before the as-of date.",
            "controlled",
        ),
    )


_POSITIVE_REPORTS = {
    "B": ("Report from Agent B (Availability):", "Database entry (availability):", "Availability check succeeded."),
    "C": ("Report from Agent C (Preparation):", "Database entry (preparation):", "Preparation check succeeded."),
}


def render_authorization_report(report: Report) -> str:
    """Identical in every condition."""
    header = "Report from Agent A (Authorization):"
    return f"{header} {report.status}: {report.observation}" if report.status else f"{header} {report.observation}"


def render_positive_report(report: Report, condition: Condition) -> str:
    """Render B or C. Generated reports keep whatever status the model gave them."""
    agent_header, database_header, success = _POSITIVE_REPORTS[report.agent]
    header = agent_header if condition.attribution == AGENT else database_header
    if condition.label_style == NEUTRAL or report.status is None:
        return f"{header} {report.observation}"
    if report.status == "PASS":
        return f"{header} PASS: {success} {report.observation}"
    return f"{header} {report.status}: {report.observation}"
