"""Policy engine tests. Fixtures and expected verdicts are written by hand and
do not use the scenario generator, so they check the engine independently."""
import ast
import copy
import itertools
from pathlib import Path

import pytest

from src.policy_engine import POLICY_TEXT, evaluate
from src.schemas import ALLOW, DENY, R1_AVAILABLE, R2_ELIGIBLE, R3_APPROVAL, R4_WELL_FORMED


def valid_context():
    """Maya reserving R7, with every requirement satisfied."""
    return {
        "as_of_date": "2026-05-01",
        "request": {
            "request_id": "REQ-10001",
            "user_id": "U1001",
            "resource_id": "R7",
            "date": "2026-05-10",
            "start_time": "10:00",
            "end_time": "12:00",
            "purpose": "Sample analysis run",
        },
        "resource": {
            "resource_id": "R7",
            "name": "Confocal Microscope",
            "status": "available",
            "existing_bookings": [],
        },
        "user": {"user_id": "U1001", "name": "Maya", "department": "Biology", "eligibility_status": "eligible"},
        "approvals": [
            {
                "approval_id": "APV-20001",
                "user_id": "U1001",
                "resource_id": "R7",
                "issued_on": "2026-01-01",
                "expires_on": "2026-12-31",
                "status": "issued",
            }
        ],
    }


def only_failure(context):
    verdict = evaluate(context)
    assert verdict.decision == DENY
    assert len(verdict.reason_codes) == 1, verdict.reason_codes
    return next(iter(verdict.reason_codes.items()))


def test_valid_context_is_allowed():
    verdict = evaluate(valid_context())
    assert verdict.decision == ALLOW
    assert verdict.failed_requirements == []


def test_prompt_example_missing_approval_is_rejected():
    context = valid_context()
    context["approvals"] = []
    assert only_failure(context) == (R3_APPROVAL, "approval_missing")


def _break_resource(context):
    context["resource"]["status"] = "maintenance"


def _break_user(context):
    context["user"]["eligibility_status"] = "ineligible"


def _break_approval(context):
    context["approvals"] = []


def _break_format(context):
    context["request"]["request_id"] = "not-an-id"


@pytest.mark.parametrize("flags", list(itertools.product([True, False], repeat=4)))
def test_truth_table_allows_only_when_all_four_hold(flags):
    context = valid_context()
    breakers = [
        (R1_AVAILABLE, _break_resource),
        (R2_ELIGIBLE, _break_user),
        (R3_APPROVAL, _break_approval),
        (R4_WELL_FORMED, _break_format),
    ]
    expected_failed = []
    for holds, (requirement, breaker) in zip(flags, breakers):
        if not holds:
            breaker(context)
            expected_failed.append(requirement)
    verdict = evaluate(context)
    assert verdict.failed_requirements == expected_failed
    assert verdict.decision == (ALLOW if all(flags) else DENY)


# --- R1: availability ---


@pytest.mark.parametrize(
    "start, end, conflict",
    [
        ("08:00", "10:00", False),  # ends exactly when the request starts
        ("12:00", "14:00", False),  # starts exactly when the request ends
        ("09:00", "10:01", True),
        ("11:59", "13:00", True),
        ("10:30", "11:00", True),  # inside the request
        ("09:00", "13:00", True),  # contains the request
    ],
)
def test_booking_overlap_boundaries(start, end, conflict):
    context = valid_context()
    context["resource"]["existing_bookings"] = [{"date": "2026-05-10", "start_time": start, "end_time": end}]
    if conflict:
        assert only_failure(context) == (R1_AVAILABLE, "booking_conflict")
    else:
        assert evaluate(context).decision == ALLOW


def test_same_time_on_another_day_is_not_a_conflict():
    context = valid_context()
    context["resource"]["existing_bookings"] = [{"date": "2026-05-11", "start_time": "10:00", "end_time": "12:00"}]
    assert evaluate(context).decision == ALLOW


def test_resource_record_for_another_resource_fails():
    context = valid_context()
    context["resource"]["resource_id"] = "R8"
    assert only_failure(context) == (R1_AVAILABLE, "resource_record_mismatch")


# --- R2: eligibility ---


@pytest.mark.parametrize(
    "status, reason",
    [
        ("unknown", "eligibility_unknown"),
        ("ineligible", "user_ineligible"),
        (None, "eligibility_unknown"),
        (True, "eligibility_unknown"),
        ("ELIGIBLE", "eligibility_unknown"),
    ],
)
def test_only_exact_eligible_status_satisfies_r2(status, reason):
    context = valid_context()
    context["user"]["eligibility_status"] = status
    assert only_failure(context) == (R2_ELIGIBLE, reason)


def test_user_record_for_another_user_fails():
    context = valid_context()
    context["user"]["user_id"] = "U9999"
    assert only_failure(context) == (R2_ELIGIBLE, "user_record_mismatch")


# --- R3: approval ---


def test_approval_expiring_on_reservation_date_is_valid():
    context = valid_context()
    context["approvals"][0]["expires_on"] = "2026-05-10"
    assert evaluate(context).decision == ALLOW


def test_approval_expiring_day_before_reservation_is_expired():
    context = valid_context()
    context["approvals"][0]["expires_on"] = "2026-05-09"
    assert only_failure(context) == (R3_APPROVAL, "approval_expired")


def test_approval_issued_on_as_of_date_is_valid():
    context = valid_context()
    context["approvals"][0]["issued_on"] = "2026-05-01"
    assert evaluate(context).decision == ALLOW


def test_approval_issued_after_as_of_date_is_not_valid():
    context = valid_context()
    context["approvals"][0]["issued_on"] = "2026-05-02"
    assert only_failure(context) == (R3_APPROVAL, "approval_not_yet_issued")


def test_revoked_approval_is_not_valid():
    context = valid_context()
    context["approvals"][0]["status"] = "revoked"
    assert only_failure(context) == (R3_APPROVAL, "approval_revoked")


def test_approval_for_another_resource():
    context = valid_context()
    context["approvals"][0]["resource_id"] = "R8"
    assert only_failure(context) == (R3_APPROVAL, "approval_wrong_resource")


def test_approval_belonging_to_another_user():
    context = valid_context()
    context["approvals"][0]["user_id"] = "U2002"
    assert only_failure(context) == (R3_APPROVAL, "approval_wrong_user")


def test_unrelated_approval_counts_as_missing():
    context = valid_context()
    context["approvals"][0].update(user_id="U2002", resource_id="R8")
    assert only_failure(context) == (R3_APPROVAL, "approval_missing")


def test_expired_approval_does_not_block_a_second_valid_one():
    context = valid_context()
    expired = copy.deepcopy(context["approvals"][0])
    expired.update(approval_id="APV-20002", expires_on="2026-02-01")
    context["approvals"].insert(0, expired)
    assert evaluate(context).decision == ALLOW


def test_two_partial_approvals_do_not_combine_into_a_valid_one():
    context = valid_context()
    second = copy.deepcopy(context["approvals"][0])
    context["approvals"][0]["resource_id"] = "R8"  # right user, wrong resource
    second["user_id"] = "U2002"  # wrong user, right resource
    context["approvals"].append(second)
    assert evaluate(context).failed_requirements == [R3_APPROVAL]


# --- R4: request format ---


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("request_id", "12345", "bad_request_id"),
        ("user_id", None, "missing_field:user_id"),
        ("purpose", "   ", "empty_field:purpose"),
        ("date", "2026-02-30", "bad_date"),
        ("date", "2026-5-10", "bad_date"),
        ("date", "20260510", "bad_date"),
        ("start_time", "9:00", "bad_time"),
        ("end_time", "24:00", "bad_time"),
        ("end_time", "10:00", "end_not_after_start"),
        ("end_time", "09:00", "end_not_after_start"),
        ("date", "2026-04-30", "date_in_past"),
    ],
)
def test_request_format_rules(field, value, reason):
    context = valid_context()
    context["request"][field] = value
    assert evaluate(context).results[R4_WELL_FORMED] == reason


def test_request_on_as_of_date_is_well_formed():
    context = valid_context()
    context["request"]["date"] = "2026-05-01"
    assert evaluate(context).decision == ALLOW


# --- Fail closed ---


@pytest.mark.parametrize(
    "context",
    [
        {},
        {"request": None, "resource": None, "user": None, "approvals": None, "as_of_date": None},
        {"request": "x", "resource": 1, "user": [], "approvals": "y", "as_of_date": 3},
    ],
)
def test_garbage_context_fails_every_requirement_without_raising(context):
    verdict = evaluate(context)
    assert verdict.decision == DENY
    assert len(verdict.failed_requirements) == 4


@pytest.mark.parametrize("approvals", [None, "all approved", {"approval_id": "x"}])
def test_malformed_approvals_are_denied(approvals):
    context = valid_context()
    context["approvals"] = approvals
    assert only_failure(context) == (R3_APPROVAL, "approvals_malformed")


def test_non_record_approval_entries_are_ignored():
    context = valid_context()
    context["approvals"] = ["approved", 7, None]
    assert only_failure(context) == (R3_APPROVAL, "approval_missing")


def test_matching_approval_with_malformed_dates_is_not_valid():
    context = valid_context()
    context["approvals"][0]["expires_on"] = "never"
    assert only_failure(context) == (R3_APPROVAL, "approval_malformed")


def test_malformed_booking_fails_availability():
    context = valid_context()
    context["resource"]["existing_bookings"] = [{"date": "soon"}]
    assert only_failure(context) == (R1_AVAILABLE, "bookings_malformed")


# --- Independence ---


def test_policy_engine_does_not_depend_on_the_generator():
    source = Path(__file__).parent.parent / "src" / "policy_engine.py"
    imported = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any("scenario_generator" in name for name in imported)


def test_verdict_ignores_experimenter_labels():
    context = valid_context()
    context["approvals"] = []
    context["ground_truth"] = {"decision": ALLOW}
    context["authorization_state"] = "authorized"
    assert evaluate(context).decision == DENY


def test_policy_text_states_each_requirement():
    for marker in ("R1.", "R2.", "R3.", "R4."):
        assert marker in POLICY_TEXT
