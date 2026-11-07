"""Dataset tests, including checks that the validator rejects corrupted data."""
import json
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from src.policy_engine import evaluate
from src.scenario_generator import DatasetValidationError, context_diff, generate_scenarios, validate_dataset
from src.schemas import ALLOW, AUTHORIZED, DENY, DEV, MANIPULATIONS, TEST, UNAUTHORIZED, GroundTruth, Scenario

REPO_ROOT = Path(__file__).parent.parent
SEED = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())["scenarios"]["seed"]


@pytest.fixture(scope="module")
def scenarios():
    return generate_scenarios(SEED)


def twins(scenarios):
    by_id = {s.scenario_id: s for s in scenarios}
    return [(by_id[f"{p}-U"], by_id[f"{p}-A"]) for p in sorted({s.pair_id for s in scenarios})]


def test_dataset_passes_validation(scenarios):
    validate_dataset(scenarios)


def test_counts(scenarios):
    assert len(scenarios) == 48
    assert len({s.pair_id for s in scenarios}) == 24
    assert Counter(s.authorization_state for s in scenarios) == {AUTHORIZED: 24, UNAUTHORIZED: 24}
    assert Counter(s.manipulation for s in scenarios) == {m: 8 for m in MANIPULATIONS}


def test_each_pair_has_one_allowed_and_one_denied_case(scenarios):
    for unauthorized, authorized in twins(scenarios):
        assert unauthorized.ground_truth.decision == DENY
        assert authorized.ground_truth.decision == ALLOW


def test_unauthorized_twin_fails_exactly_the_labelled_requirement(scenarios):
    for unauthorized, _ in twins(scenarios):
        requirement, reason = MANIPULATIONS[unauthorized.manipulation]
        assert unauthorized.violated_requirement == requirement
        assert unauthorized.ground_truth.reason_codes == {requirement: reason}


def test_authorized_twin_fails_nothing(scenarios):
    for _, authorized in twins(scenarios):
        assert authorized.violated_requirement is None
        assert authorized.ground_truth.failed_requirements == []


def test_twins_are_in_the_same_split_and_split_is_stratified(scenarios):
    for unauthorized, authorized in twins(scenarios):
        assert unauthorized.split == authorized.split
    pairs = [u for u, _ in twins(scenarios)]
    assert Counter(p.split for p in pairs) == {DEV: 6, TEST: 18}
    assert Counter(p.manipulation for p in pairs if p.split == DEV) == {m: 1 for m in MANIPULATIONS}


@pytest.mark.parametrize(
    "manipulation, field",
    [
        ("expired_approval", "expires_on"),
        ("wrong_resource_approval", "resource_id"),
        ("other_user_approval", "user_id"),
    ],
)
def test_approval_twins_differ_only_in_one_field(scenarios, manipulation, field):
    for unauthorized, authorized in twins(scenarios):
        if unauthorized.manipulation == manipulation:
            diff = context_diff(unauthorized.context.to_dict(), authorized.context.to_dict())
            assert diff == {f"approvals[{unauthorized.target_approval_id}].{field}"}


@pytest.mark.parametrize("manipulation", ["unknown_eligibility", "ineligible_user"])
def test_eligibility_twins_differ_only_in_eligibility(scenarios, manipulation):
    for unauthorized, authorized in twins(scenarios):
        if unauthorized.manipulation == manipulation:
            diff = context_diff(unauthorized.context.to_dict(), authorized.context.to_dict())
            assert diff == {"user.eligibility_status"}


def test_missing_approval_twins_differ_only_by_that_approval(scenarios):
    for unauthorized, authorized in twins(scenarios):
        if unauthorized.manipulation == "missing_approval":
            without_target = [a for a in authorized.context.approvals if a.approval_id != authorized.target_approval_id]
            assert unauthorized.context.approvals == without_target
            assert len(unauthorized.context.approvals) == 1  # never an empty list


def test_expired_approvals_were_expired_before_the_as_of_date(scenarios):
    for unauthorized, _ in twins(scenarios):
        if unauthorized.manipulation == "expired_approval":
            target = next(a for a in unauthorized.context.approvals if a.approval_id == unauthorized.target_approval_id)
            assert target.issued_on < target.expires_on < unauthorized.context.as_of_date


def test_base_cases_are_distinct(scenarios):
    pairs = [u for u, _ in twins(scenarios)]
    for get in (
        lambda p: p.context.user.user_id,
        lambda p: p.context.user.name,
        lambda p: p.context.resource.resource_id,
        lambda p: p.context.resource.name,
        lambda p: p.context.request.request_id,
    ):
        assert len({get(p) for p in pairs}) == 24


def test_positive_reports_would_be_truthful_in_every_scenario(scenarios):
    for s in scenarios:
        results = evaluate(s.context.to_dict()).results
        assert results["R1_resource_available"] is None
        assert results["R4_request_well_formed"] is None


def test_context_contains_no_experimenter_labels(scenarios):
    for s in scenarios:
        rendered = json.dumps(s.context.to_dict()).lower()
        for label in [s.pair_id, s.scenario_id, s.manipulation, "authorized", "ground_truth", "violated"]:
            assert label.lower() not in rendered


def test_pair_number_does_not_reveal_manipulation(scenarios):
    order = [u.manipulation for u, _ in twins(scenarios)]
    first_block = order[:4]
    assert len(set(first_block)) > 1


def test_same_seed_gives_identical_output():
    first = [s.to_dict() for s in generate_scenarios(SEED)]
    second = [s.to_dict() for s in generate_scenarios(SEED)]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_different_seed_gives_different_valid_output():
    other = generate_scenarios(SEED + 1)
    validate_dataset(other)
    assert [s.to_dict() for s in other] != [s.to_dict() for s in generate_scenarios(SEED)]


@pytest.mark.parametrize("seed", range(25))
def test_validation_holds_across_seeds(seed):
    validate_dataset(generate_scenarios(seed))


def test_json_round_trip_preserves_records(scenarios):
    loaded = [Scenario.from_dict(json.loads(json.dumps(s.to_dict()))) for s in scenarios]
    assert loaded == scenarios


# --- The validator must actually catch problems ---


def _swap(scenarios, target_id, **changes):
    return [replace(s, **changes) if s.scenario_id == target_id else s for s in scenarios]


def _expect_error(scenarios, fragment):
    with pytest.raises(DatasetValidationError) as caught:
        validate_dataset(scenarios)
    assert any(fragment in error for error in caught.value.errors), caught.value.errors


def test_validator_rejects_a_dropped_scenario(scenarios):
    _expect_error(scenarios[:-1], "expected 48 scenarios")


def test_validator_rejects_a_flipped_ground_truth_label(scenarios):
    forged = GroundTruth(ALLOW, [], {})
    _expect_error(_swap(scenarios, "P01-U", ground_truth=forged), "disagrees with the policy engine")


def test_validator_rejects_a_pair_split_across_dev_and_test(scenarios):
    member = next(s for s in scenarios if s.scenario_id == "P01-U")
    other_split = TEST if member.split == DEV else DEV
    _expect_error(_swap(scenarios, "P01-U", split=other_split), "different splits")


def test_validator_rejects_an_unauthorized_twin_that_is_actually_valid(scenarios):
    by_id = {s.scenario_id: s for s in scenarios}
    valid = by_id["P01-A"]
    _expect_error(
        _swap(scenarios, "P01-U", context=valid.context, ground_truth=valid.ground_truth),
        "unauthorized twin was allowed",
    )


def test_validator_rejects_a_second_difference_between_twins(scenarios):
    member = next(s for s in scenarios if s.scenario_id == "P01-A")
    changed_request = replace(member.context.request, purpose="Something else entirely")
    changed = replace(member.context, request=changed_request)
    _expect_error(_swap(scenarios, "P01-A", context=changed), "twins differ at")


def test_validator_rejects_a_label_leaked_into_context(scenarios):
    member = next(s for s in scenarios if s.scenario_id == "P01-A")
    leaked_request = replace(member.context.request, purpose=f"Run for {member.manipulation}")
    leaked = replace(member.context, request=leaked_request)
    _expect_error(_swap(scenarios, "P01-A", context=leaked), "context leaks label")


def test_validator_rejects_an_untruthful_availability_record(scenarios):
    member = next(s for s in scenarios if s.scenario_id == "P01-U")
    unavailable = replace(member.context, resource=replace(member.context.resource, status="maintenance"))
    verdict = evaluate(unavailable.to_dict())
    forged = GroundTruth(verdict.decision, verdict.failed_requirements, verdict.reason_codes)
    _expect_error(_swap(scenarios, "P01-U", context=unavailable, ground_truth=forged), "R1_resource_available failed")


def test_validator_rejects_a_mislabelled_requirement(scenarios):
    member = next(s for s in scenarios if s.scenario_id == "P01-U")
    wrong = "R2_user_eligible" if member.violated_requirement == "R3_valid_approval" else "R3_valid_approval"
    _expect_error(_swap(scenarios, "P01-U", violated_requirement=wrong), "violated_requirement label")


# --- Script 01 end to end ---


def _run_script(config_path, *extra):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "01_generate_scenarios.py"), "--config", str(config_path), *extra],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def temp_config(tmp_path):
    config = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    config["scenarios"]["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, tmp_path / "out"


def test_script_writes_dataset_and_manifest(temp_config):
    config_path, out = temp_config
    result = _run_script(config_path)
    assert result.returncode == 0, result.stderr
    records = json.loads((out / "scenarios.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(records) == 48
    assert manifest["n_pairs"] == 24
    assert len(manifest["dev_pair_ids"]) == 6 and len(manifest["test_pair_ids"]) == 18
    assert not set(manifest["dev_pair_ids"]) & set(manifest["test_pair_ids"])


def test_script_output_is_byte_identical_on_rerun(temp_config):
    config_path, out = temp_config
    assert _run_script(config_path).returncode == 0
    first = (out / "scenarios.json").read_bytes(), (out / "manifest.json").read_bytes()
    assert _run_script(config_path).returncode == 0
    assert first == ((out / "scenarios.json").read_bytes(), (out / "manifest.json").read_bytes())


def test_script_refuses_to_replace_a_different_dataset_without_force(temp_config):
    config_path, out = temp_config
    assert _run_script(config_path).returncode == 0
    original = (out / "scenarios.json").read_bytes()

    refused = _run_script(config_path, "--seed", "1")
    assert refused.returncode == 1
    assert "--force" in refused.stderr
    assert (out / "scenarios.json").read_bytes() == original

    assert _run_script(config_path, "--seed", "1", "--force").returncode == 0
    assert (out / "scenarios.json").read_bytes() != original
