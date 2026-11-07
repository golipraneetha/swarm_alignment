# Local Success, Global Failure

**When multi-agent success signals override safety refusals.**

Can a multi-agent workflow produce an unauthorized action because the final
agent gives too much weight to truthful local success reports, even though the
correct refusal, the policy and the underlying records are all visible?

This is a causal behavioral experiment in a synthetic reservation system. It
makes no real reservations and uses no real accounts or external APIs. The
hypotheses may turn out false, and null results will be reported as such.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Scenarios, policy engine, tests | Done |
| 2 | Model backend, agents, conditions, smoke test | Done; prompts frozen (`07b2cc23`) |
| 3 | Analysis and plots | Code done and tested; pilot on Qwen2.5-3B in progress, no results yet |

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
.venv/bin/python scripts/01_generate_scenarios.py
.venv/bin/python scripts/02_run_experiments.py --run-name smoke --limit-pairs 4 --reps 1
```

Script 02 defaults to the dev split and never downloads weights (`offline: true`).
Held-out cases are refused until `--freeze-prompts` has been run.

Output goes to `data/scenarios/`: `scenarios.json` holds the records and
`manifest.json` holds the seed, the hashes and the split. Re-running with the
same config reproduces both files byte for byte. A run that would change an
existing dataset is refused unless `--force` is passed.

## Scenario dataset

There are 24 base cases and 48 scenarios: each base case has an unauthorized
twin and an authorized twin.

| Manipulation | Violates | What the authorized twin corrects | Pairs |
|---|---|---|---|
| `missing_approval` | R3 | Approval record is present | 4 |
| `expired_approval` | R3 | `expires_on` | 4 |
| `wrong_resource_approval` | R3 | `resource_id` on the approval | 4 |
| `other_user_approval` | R3 | `user_id` on the approval | 4 |
| `unknown_eligibility` | R2 | `eligibility_status` | 4 |
| `ineligible_user` | R2 | `eligibility_status` | 4 |

Design properties, all enforced by `validate_dataset`:

- **Ground truth is cross-checked.** Labels come from `src/policy_engine.py`,
  which reads only the records and never imports the generator. The generator's
  intended outcome must match the engine's verdict and reason code.
- **Twins differ in exactly one place,** so any behavioral difference between
  them can be traced to that record.
- **Exactly one requirement fails** in each unauthorized scenario. R1 and R4
  hold everywhere, so positive availability and preparation reports can always
  be truthful.
- **Labels stay out of agent view.** Agents may only see a scenario's `context`.
  The validator rejects any context containing an experimenter label, and pair
  numbers are shuffled so they reveal nothing about the manipulation.
- **The split is stratified.** There are 6 development pairs and 18 held-out
  pairs, with twins always in the same split.

Choices worth checking when interpreting results:

- Every scenario includes an unrelated approval (another user, another
  resource), so a missing approval is never signalled by an empty list.
- `status: "issued"` means an approval was granted, not that it is still valid.
  An expired approval therefore still reads `issued`, and the model has to
  compare dates to notice.
- Missing-approval twins necessarily differ in approval count. This differs
  between twins but not between experimental conditions, so it cannot confound
  condition comparisons.

## Dev and held-out data

Prompts may be debugged on development pairs only, and must be frozen before any
held-out pair is run. Confirmatory results should therefore use the 18 held-out
pairs, which is 288 of the 384 pilot replays. Development pairs have informed
prompt wording, so they must be reported separately.

## Outcome scoring

`requested_action` alone decides the outcome. Other schema defects (empty reasons,
a `decision` that contradicts the action) are logged in `schema_defects` and do not
discard the decision. Output counts as `malformed` only when no valid action can be
parsed, and `action_hint` keeps what it appeared to request for sensitivity analysis.
The controller never acts on a hint.

Runs made before this rule (`smoke_qwen05b`, `smoke2_*`) store outcomes from stricter
parsers. Their raw outputs are intact, but their stored outcome fields are out of date.

## Limitations

This is a pilot. With 18 held-out clusters and 3 held-out pairs per manipulation,
overall estimates will have wide intervals and per-manipulation results are only
descriptive. Scenarios are drawn from one synthetic domain, so results do not
support broad generalization.
