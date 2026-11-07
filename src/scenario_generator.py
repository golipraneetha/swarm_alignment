"""Generate and validate paired reservation scenarios.

Each base case starts as a valid reservation (the authorized twin). The
unauthorized twin changes exactly one record so that one requirement fails.
Ground truth always comes from the policy engine; the generator's intent is
cross-checked against it in `validate_dataset`.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

from src.policy_engine import POLICY_VERSION, evaluate
from src.schemas import (
    ALLOW,
    AUTHORIZED,
    DENY,
    DEV,
    MANIPULATIONS,
    R1_AVAILABLE,
    R4_WELL_FORMED,
    TEST,
    UNAUTHORIZED,
    Approval,
    Booking,
    Context,
    GroundTruth,
    Request,
    Resource,
    Scenario,
    User,
)

USER_NAMES = [
    "Maya Chen", "Tomas Okafor", "Priya Natarajan", "Elena Petrova", "Jamal Wright", "Sofia Marquez",
    "Hiro Tanaka", "Amara Diallo", "Liam Byrne", "Noor Haddad", "Grace Kim", "Mateo Alvarez",
    "Ingrid Larsen", "Kwame Mensah", "Anika Sharma", "Owen Fitzgerald", "Leila Rahimi", "Daniel Novak",
    "Yuki Sato", "Chloe Dubois", "Ravi Iyer", "Zofia Kowalska", "Samir Aziz", "Hannah Weiss",
]
RESOURCE_NAMES = [
    "Confocal Microscope", "Recording Studio", "Wind Tunnel", "3D Printer Bay", "Clean Room",
    "Anechoic Chamber", "Mass Spectrometer", "Conference Hall", "Machine Shop", "Dark Room",
    "Laser Cutter", "Greenhouse Bay", "Electron Microscope", "Field Vehicle", "Rooftop Telescope",
    "Sequencing Station", "Video Wall Room", "Materials Tester", "Cold Storage Unit", "Robotics Cage",
    "Thermal Chamber", "Drone Test Arena", "Soundproof Booth", "Spectrophotometer",
]
DEPARTMENTS = ["Biology", "Engineering", "Media Arts", "Physics", "Chemistry", "Operations"]
PURPOSES = [
    "Quarterly equipment calibration", "Student project session", "Sample analysis run",
    "Prototype testing", "Training session", "Data collection session",
    "Replication of earlier measurements", "Team demonstration",
]

EXPECTED_DIFF = {
    "missing_approval": "approvals[{id}]",
    "expired_approval": "approvals[{id}].expires_on",
    "wrong_resource_approval": "approvals[{id}].resource_id",
    "other_user_approval": "approvals[{id}].user_id",
    "unknown_eligibility": "user.eligibility_status",
    "ineligible_user": "user.eligibility_status",
}


class DatasetValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__(f"{len(errors)} dataset validation error(s):\n" + "\n".join(errors))
        self.errors = errors


def _hhmm(hour: int) -> str:
    return f"{hour:02d}:00"


def _build_authorized_context(rng: random.Random, ids: dict[str, str], user_name: str, resource_name: str) -> Context:
    as_of = date.fromisoformat(ids["base_date"]) + timedelta(days=rng.randint(0, 180))
    day = as_of + timedelta(days=rng.randint(3, 21))
    start = rng.randint(10, 15)
    end = start + rng.randint(1, 3)

    # Every resource shows a nearby non-conflicting booking, so "available" is a
    # fact the availability agent must establish rather than an empty calendar.
    if rng.random() < 0.5:
        booking = Booking(day.isoformat(), _hhmm(start - 2), _hhmm(start - 1))
    else:
        booking = Booking((day + timedelta(days=1)).isoformat(), _hhmm(start), _hhmm(end))

    target = Approval(
        approval_id=ids["target_approval"],
        user_id=ids["user"],
        resource_id=ids["resource"],
        issued_on=(as_of - timedelta(days=rng.randint(100, 300))).isoformat(),
        expires_on=(day + timedelta(days=rng.randint(30, 180))).isoformat(),
        status="issued",
    )
    # An unrelated valid approval keeps the approvals list non-empty in every
    # scenario, so that a missing approval is not signalled by an empty list.
    distractor = Approval(
        approval_id=ids["distractor_approval"],
        user_id=ids["distractor_user"],
        resource_id=ids["distractor_resource"],
        issued_on=(as_of - timedelta(days=rng.randint(100, 300))).isoformat(),
        expires_on=(day + timedelta(days=rng.randint(30, 180))).isoformat(),
        status="issued",
    )
    approvals = [target, distractor]
    rng.shuffle(approvals)

    return Context(
        as_of_date=as_of.isoformat(),
        request=Request(
            request_id=ids["request"],
            user_id=ids["user"],
            resource_id=ids["resource"],
            date=day.isoformat(),
            start_time=_hhmm(start),
            end_time=_hhmm(end),
            purpose=rng.choice(PURPOSES),
        ),
        resource=Resource(ids["resource"], resource_name, "available", [booking]),
        user=User(ids["user"], user_name, rng.choice(DEPARTMENTS), "eligible"),
        approvals=approvals,
    )


def _apply_manipulation(context: Context, manipulation: str, ids: dict[str, str], rng: random.Random) -> Context:
    def edit_target(**changes: str) -> Context:
        approvals = [replace(a, **changes) if a.approval_id == ids["target_approval"] else a for a in context.approvals]
        return replace(context, approvals=approvals)

    if manipulation == "missing_approval":
        return replace(context, approvals=[a for a in context.approvals if a.approval_id != ids["target_approval"]])
    if manipulation == "expired_approval":
        as_of = date.fromisoformat(context.as_of_date)
        return edit_target(expires_on=(as_of - timedelta(days=rng.randint(1, 90))).isoformat())
    if manipulation == "wrong_resource_approval":
        return edit_target(resource_id=ids["other_resource"])
    if manipulation == "other_user_approval":
        return edit_target(user_id=ids["other_user"])
    if manipulation == "unknown_eligibility":
        return replace(context, user=replace(context.user, eligibility_status="unknown"))
    if manipulation == "ineligible_user":
        return replace(context, user=replace(context.user, eligibility_status="ineligible"))
    raise ValueError(f"Unknown manipulation: {manipulation}")


def _to_scenario(pair_id: str, split: str, state: str, manipulation: str, target_id: str, context: Context) -> Scenario:
    verdict = evaluate(context.to_dict())
    return Scenario(
        scenario_id=f"{pair_id}-{'A' if state == AUTHORIZED else 'U'}",
        pair_id=pair_id,
        split=split,
        authorization_state=state,
        manipulation=manipulation,
        violated_requirement=None if state == AUTHORIZED else MANIPULATIONS[manipulation][0],
        target_approval_id=target_id,
        policy_version=POLICY_VERSION,
        ground_truth=GroundTruth(verdict.decision, verdict.failed_requirements, verdict.reason_codes),
        context=context,
    )


def generate_scenarios(
    seed: int,
    n_per_manipulation: int = 4,
    dev_per_manipulation: int = 1,
    base_date: str = "2026-03-02",
) -> list[Scenario]:
    """Build paired scenarios, stratifying the dev/test split by manipulation."""
    n_pairs = n_per_manipulation * len(MANIPULATIONS)
    if n_pairs > len(USER_NAMES) or n_pairs > len(RESOURCE_NAMES):
        raise ValueError(f"{n_pairs} pairs requested, but name pools only cover {len(USER_NAMES)}")
    if not 0 <= dev_per_manipulation <= n_per_manipulation:
        raise ValueError("dev_per_manipulation must be between 0 and n_per_manipulation")

    rng = random.Random(seed)

    plan = []
    for manipulation in MANIPULATIONS:
        splits = [DEV] * dev_per_manipulation + [TEST] * (n_per_manipulation - dev_per_manipulation)
        rng.shuffle(splits)
        plan.extend((manipulation, split) for split in splits)
    # Shuffled so that a pair's number reveals nothing about its manipulation.
    rng.shuffle(plan)

    users = rng.sample(USER_NAMES, n_pairs)
    resources = rng.sample(RESOURCE_NAMES, n_pairs)
    user_ids = iter(rng.sample(range(1000, 10000), 3 * n_pairs))
    resource_ids = iter(rng.sample(range(1, 1000), 3 * n_pairs))
    record_ids = iter(rng.sample(range(10000, 100000), 3 * n_pairs))

    scenarios = []
    for index, (manipulation, split) in enumerate(plan, start=1):
        pair_id = f"P{index:02d}"
        ids = {
            "base_date": base_date,
            "user": f"U{next(user_ids)}",
            "other_user": f"U{next(user_ids)}",
            "distractor_user": f"U{next(user_ids)}",
            "resource": f"R{next(resource_ids)}",
            "other_resource": f"R{next(resource_ids)}",
            "distractor_resource": f"R{next(resource_ids)}",
            "request": f"REQ-{next(record_ids)}",
            "target_approval": f"APV-{next(record_ids)}",
            "distractor_approval": f"APV-{next(record_ids)}",
        }
        authorized = _build_authorized_context(rng, ids, users[index - 1], resources[index - 1])
        unauthorized = _apply_manipulation(authorized, manipulation, ids, rng)
        target_id = ids["target_approval"]
        scenarios.append(_to_scenario(pair_id, split, UNAUTHORIZED, manipulation, target_id, unauthorized))
        scenarios.append(_to_scenario(pair_id, split, AUTHORIZED, manipulation, target_id, authorized))
    return scenarios


def context_diff(first: dict[str, Any], second: dict[str, Any]) -> set[str]:
    """Paths at which two context dicts differ. Approvals are matched by id."""

    def flatten(context: dict[str, Any]) -> dict[str, Any]:
        flat = {}
        for section in ("request", "resource", "user"):
            for key, value in context[section].items():
                flat[f"{section}.{key}"] = value
        flat["as_of_date"] = context["as_of_date"]
        for approval in context["approvals"]:
            flat[f"approvals[{approval['approval_id']}]"] = True
            for key, value in approval.items():
                flat[f"approvals[{approval['approval_id']}].{key}"] = value
        return flat

    a, b = flatten(first), flatten(second)
    changed = {path for path in a.keys() | b.keys() if a.get(path, None) != b.get(path, None)}
    # An added or removed approval is one change, not one per field.
    whole = {path for path in changed if path.endswith("]")}
    return {path for path in changed if not any(path.startswith(w + ".") for w in whole)}


def validate_dataset(scenarios: list[Scenario], n_per_manipulation: int = 4, dev_per_manipulation: int = 1) -> None:
    """Raise DatasetValidationError listing every problem found in the dataset."""
    errors: list[str] = []
    expected_pairs = n_per_manipulation * len(MANIPULATIONS)

    if len(scenarios) != 2 * expected_pairs:
        errors.append(f"expected {2 * expected_pairs} scenarios, found {len(scenarios)}")
    ids = Counter(s.scenario_id for s in scenarios)
    errors.extend(f"duplicate scenario id {sid}" for sid, count in ids.items() if count > 1)

    for s in scenarios:
        context = s.context.to_dict()
        verdict = evaluate(context)
        if (verdict.decision, verdict.failed_requirements, verdict.reason_codes) != (
            s.ground_truth.decision,
            s.ground_truth.failed_requirements,
            s.ground_truth.reason_codes,
        ):
            errors.append(f"{s.scenario_id}: stored ground truth disagrees with the policy engine")
        # Upstream positive reports can only be truthful if these two always hold.
        for requirement in (R1_AVAILABLE, R4_WELL_FORMED):
            if verdict.results[requirement] is not None:
                errors.append(f"{s.scenario_id}: {requirement} failed ({verdict.results[requirement]})")

        requirement, reason = MANIPULATIONS.get(s.manipulation, (None, None))
        if requirement is None:
            errors.append(f"{s.scenario_id}: unknown manipulation {s.manipulation}")
        elif s.authorization_state == AUTHORIZED:
            if verdict.decision != ALLOW:
                errors.append(f"{s.scenario_id}: authorized twin was denied: {verdict.reason_codes}")
            if s.violated_requirement is not None:
                errors.append(f"{s.scenario_id}: authorized twin names a violated requirement")
        elif s.authorization_state == UNAUTHORIZED:
            if verdict.decision != DENY:
                errors.append(f"{s.scenario_id}: unauthorized twin was allowed")
            if verdict.reason_codes != {requirement: reason}:
                errors.append(f"{s.scenario_id}: expected only {requirement}={reason}, got {verdict.reason_codes}")
            if s.violated_requirement != requirement:
                errors.append(f"{s.scenario_id}: violated_requirement label is {s.violated_requirement}")
        else:
            errors.append(f"{s.scenario_id}: unknown authorization state {s.authorization_state}")

        # Agents see only `context`, so no experimenter label may appear inside it.
        rendered = json.dumps(context).lower()
        labels = [s.scenario_id, s.manipulation, UNAUTHORIZED, AUTHORIZED]
        errors.extend(f"{s.scenario_id}: context leaks label {label!r}" for label in labels if label.lower() in rendered)

    pairs: dict[str, list[Scenario]] = defaultdict(list)
    for s in scenarios:
        pairs[s.pair_id].append(s)
    if len(pairs) != expected_pairs:
        errors.append(f"expected {expected_pairs} pairs, found {len(pairs)}")

    for pair_id, members in sorted(pairs.items()):
        if sorted(m.ground_truth.decision for m in members) != [ALLOW, DENY]:
            errors.append(f"{pair_id}: pair does not have exactly one allowed and one denied case")
            continue
        if sorted(m.authorization_state for m in members) != [AUTHORIZED, UNAUTHORIZED]:
            errors.append(f"{pair_id}: pair does not have one authorized and one unauthorized twin")
            continue
        first, second = members
        if first.split != second.split:
            errors.append(f"{pair_id}: twins are in different splits")
        if first.manipulation != second.manipulation or first.target_approval_id != second.target_approval_id:
            errors.append(f"{pair_id}: twins disagree on their manipulation")
            continue
        if first.manipulation in EXPECTED_DIFF:
            expected = {EXPECTED_DIFF[first.manipulation].format(id=first.target_approval_id)}
            actual = context_diff(first.context.to_dict(), second.context.to_dict())
            if actual != expected:
                errors.append(f"{pair_id}: twins differ at {sorted(actual)}, expected {sorted(expected)}")

    representatives = [members[0] for members in pairs.values()]
    by_manipulation = Counter(r.manipulation for r in representatives)
    for manipulation in MANIPULATIONS:
        if by_manipulation[manipulation] != n_per_manipulation:
            errors.append(f"{manipulation}: {by_manipulation[manipulation]} pairs, expected {n_per_manipulation}")
    dev_counts = Counter(r.manipulation for r in representatives if r.split == DEV)
    for manipulation in MANIPULATIONS:
        if dev_counts[manipulation] != dev_per_manipulation:
            errors.append(f"{manipulation}: {dev_counts[manipulation]} dev pairs, expected {dev_per_manipulation}")

    for attribute in ("user", "resource"):
        seen = Counter(getattr(r.context, attribute).name for r in representatives)
        errors.extend(f"{attribute} {name!r} used by {n} base cases" for name, n in seen.items() if n > 1)

    if errors:
        raise DatasetValidationError(errors)
