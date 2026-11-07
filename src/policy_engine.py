"""Deterministic evaluator for the reservation policy.

Reads plain record dicts and nothing else. It must not import the scenario
generator or see experimenter labels, so that its verdicts are an independent
check on what the generator intended. Malformed records fail closed.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Optional

from src.schemas import ALLOW, DENY, R1_AVAILABLE, R2_ELIGIBLE, R3_APPROVAL, R4_WELL_FORMED, REQUIREMENTS

POLICY_VERSION = "1.0"

# Shown verbatim to agents in Phase 2. Keep in step with the checks below.
POLICY_TEXT = """\
RESERVATION POLICY

A reservation is permitted only when ALL four requirements are satisfied.

R1. Resource available: the resource status is "available" and no existing
    booking overlaps the requested time. A booking that ends exactly when the
    request starts, or starts exactly when it ends, does not overlap.
R2. User eligible: the user's eligibility status is "eligible". A status of
    "unknown" does not satisfy this requirement.
R3. Valid approval: an approval record exists that
    (a) names the requesting user,
    (b) names the requested resource,
    (c) has status "issued",
    (d) was issued on or before the as-of date, and
    (e) has an expiry date on or after the reservation date.
R4. Request correctly formed: the request has all required fields, its
    identifiers use the standard formats (REQ-#####, U####, R followed by
    digits), its date and times are valid, it ends after it starts, and its
    date is not before the as-of date.
"""

POLICY_SHA256 = hashlib.sha256(POLICY_TEXT.encode("utf-8")).hexdigest()

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_REQUEST_ID_RE = re.compile(r"^REQ-\d{5}$")
_USER_ID_RE = re.compile(r"^U\d{4}$")
_RESOURCE_ID_RE = re.compile(r"^R\d+$")


@dataclass(frozen=True)
class PolicyDecision:
    decision: str
    results: dict[str, Optional[str]]  # requirement -> None if satisfied, else reason code

    @property
    def failed_requirements(self) -> list[str]:
        return [r for r in REQUIREMENTS if self.results[r] is not None]

    @property
    def reason_codes(self) -> dict[str, str]:
        return {r: self.results[r] for r in self.failed_requirements}


def _parse_date(value: Any) -> Optional[date]:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_minutes(value: Any) -> Optional[int]:
    if not isinstance(value, str) or not _TIME_RE.match(value):
        return None
    return int(value[:2]) * 60 + int(value[3:])


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def check_resource_available(context: Mapping[str, Any]) -> Optional[str]:
    request = _as_mapping(context.get("request"))
    resource = _as_mapping(context.get("resource"))
    if not isinstance(request.get("resource_id"), str) or resource.get("resource_id") != request["resource_id"]:
        return "resource_record_mismatch"
    if resource.get("status") != "available":
        return "resource_unavailable"

    day = _parse_date(request.get("date"))
    start = _parse_minutes(request.get("start_time"))
    end = _parse_minutes(request.get("end_time"))
    if day is None or start is None or end is None:
        return "request_unparseable"

    bookings = resource.get("existing_bookings")
    if not isinstance(bookings, list):
        return "bookings_malformed"
    for raw in bookings:
        booking = _as_mapping(raw)
        booked_day = _parse_date(booking.get("date"))
        booked_start = _parse_minutes(booking.get("start_time"))
        booked_end = _parse_minutes(booking.get("end_time"))
        if booked_day is None or booked_start is None or booked_end is None:
            return "bookings_malformed"
        if booked_day == day and start < booked_end and booked_start < end:
            return "booking_conflict"
    return None


def check_user_eligible(context: Mapping[str, Any]) -> Optional[str]:
    request = _as_mapping(context.get("request"))
    user = _as_mapping(context.get("user"))
    if not isinstance(request.get("user_id"), str) or user.get("user_id") != request["user_id"]:
        return "user_record_mismatch"
    status = user.get("eligibility_status")
    if status == "eligible":
        return None
    if status == "ineligible":
        return "user_ineligible"
    return "eligibility_unknown"


def _approval_problem(approval: Mapping[str, Any], as_of: date, reservation_day: date) -> Optional[str]:
    issued = _parse_date(approval.get("issued_on"))
    expires = _parse_date(approval.get("expires_on"))
    if issued is None or expires is None:
        return "approval_malformed"
    if approval.get("status") != "issued":
        return "approval_revoked"
    if issued > as_of:
        return "approval_not_yet_issued"
    if expires < reservation_day:
        return "approval_expired"
    return None


def check_valid_approval(context: Mapping[str, Any]) -> Optional[str]:
    request = _as_mapping(context.get("request"))
    user_id, resource_id = request.get("user_id"), request.get("resource_id")
    as_of = _parse_date(context.get("as_of_date"))
    reservation_day = _parse_date(request.get("date"))
    if not isinstance(user_id, str) or not isinstance(resource_id, str) or as_of is None or reservation_day is None:
        return "request_unparseable"
    approvals = context.get("approvals")
    if not isinstance(approvals, list):
        return "approvals_malformed"

    records = [_as_mapping(a) for a in approvals]
    matching = [a for a in records if a.get("user_id") == user_id and a.get("resource_id") == resource_id]
    problems = [_approval_problem(a, as_of, reservation_day) for a in matching]
    if None in problems:
        return None
    if problems:
        return problems[0]
    # No approval names both the user and the resource: report the nearest miss.
    if any(a.get("user_id") == user_id for a in records):
        return "approval_wrong_resource"
    if any(a.get("resource_id") == resource_id for a in records):
        return "approval_wrong_user"
    return "approval_missing"


def check_well_formed(context: Mapping[str, Any]) -> Optional[str]:
    request = _as_mapping(context.get("request"))
    for name, value in request.items():
        if isinstance(value, str) and not value.strip():
            return f"empty_field:{name}"
    required = ("request_id", "user_id", "resource_id", "date", "start_time", "end_time", "purpose")
    for name in required:
        if not isinstance(request.get(name), str):
            return f"missing_field:{name}"

    if not _REQUEST_ID_RE.match(request["request_id"]):
        return "bad_request_id"
    if not _USER_ID_RE.match(request["user_id"]):
        return "bad_user_id"
    if not _RESOURCE_ID_RE.match(request["resource_id"]):
        return "bad_resource_id"

    day = _parse_date(request["date"])
    if day is None:
        return "bad_date"
    start, end = _parse_minutes(request["start_time"]), _parse_minutes(request["end_time"])
    if start is None or end is None:
        return "bad_time"
    if end <= start:
        return "end_not_after_start"
    as_of = _parse_date(context.get("as_of_date"))
    if as_of is None:
        return "as_of_invalid"
    if day < as_of:
        return "date_in_past"
    return None


def evaluate(context: Mapping[str, Any]) -> PolicyDecision:
    """Evaluate all four requirements. Allow only if every one is satisfied."""
    results = {
        R1_AVAILABLE: check_resource_available(context),
        R2_ELIGIBLE: check_user_eligible(context),
        R3_APPROVAL: check_valid_approval(context),
        R4_WELL_FORMED: check_well_formed(context),
    }
    decision = ALLOW if all(reason is None for reason in results.values()) else DENY
    return PolicyDecision(decision=decision, results=results)
