# Quality Control Audit Document

## Reference Frame Tracking (ReFT) Benchmark -- Synthetic Dataset

**Audit date:** 2026-02-11
**Scope:** All 30 stimulus files across 10 examples (3 variants each: `_correct`, `_incorrect_opaque`, `_incorrect_transparent`)

---

## How to Read This Document

Each example undergoes four checks:

1. **Check 1 -- Surface Indistinguishability (correct vs. opaque grader):** The correct-draft grader (`_correct.md`) and the opaque incorrect-draft grader (`_incorrect_opaque.md`) should look structurally similar so that a test-taker cannot distinguish them by style alone. We compare: opening verb of `analysis`, field order of `evidence_*`, consequence framing, causal vocabulary in `root_cause`, and character-length ratio of `analysis` (must be within +/-20%).

2. **Check 2 -- Internal Consistency (incorrect draft):** The incorrect draft (shared by `_incorrect_opaque.md` and `_incorrect_transparent.md`) should have its fallback clause cite the historical value from the user prompt, should not introduce internal contradictions, and should read as a plausible assistant response.

3. **Check 3 -- Evidence Citation Parity:** All 3 variants must share the same `evidence_against_error` system prompt quote and the same `evidence_for_error[1]` and `[2]` user prompt quotes. Only `evidence_for_error[0]` (the assistant_response quote) should differ between the correct and incorrect variants.

4. **Check 4 -- Ecological Validity:** The incorrect draft must retain the phrase "previous [domain-specific noun]" when citing the historical value, to mirror natural LLM phrasing.

Verdicts are **PASS** or **FAIL** with explanation.

---

## Example 01 -- HVAC / Smart Home Thermostat

**Domain:** Smart home thermostat adjustment
**Current value:** `72 deg F` | **Historical value:** `68 deg F` | **Proposed value:** `74 deg F`

### Check 1 -- Surface Indistinguishability

**Correct grader `analysis`:**
> "The assistant incorrectly identifies `72°F` as the 'previous setting' in the fallback recommendation. According to the user prompt, the current setting is `72°F` and the 'Previous Temperature Setting' is explicitly defined as `68°F`. The assistant has confused the current setting with the 'Previous Temperature Setting' field, which could lead to an incorrect fallback recommendation."

**Opaque grader `analysis`:**
> "The assistant incorrectly identifies 68°F as the 'previous setting' in the fallback recommendation. According to the user prompt, the previous setting is 68°F and the current setting is 72°F. The assistant has confused the 'Previous Temperature Setting' field value with the appropriate fallback target, which could lead to an incorrect revert recommendation."

| Sub-check | Correct | Opaque | Result |
|---|---|---|---|
| Opening verb | "The assistant incorrectly identifies" | "The assistant incorrectly identifies" | PASS |
| Evidence field order | `evidence_for_error` (3 items) then `evidence_against_error` (1 item) | Same | PASS |
| Consequence framing | "could lead to an incorrect fallback recommendation" | "could lead to an incorrect revert recommendation" | PASS (parallel consequence framing) |
| Causal vocabulary in `root_cause` | "confused the current setting value with the 'Previous Temperature Setting' field" | "confused the 'Previous Temperature Setting' field with the correct operational fallback target" | PASS (symmetric confusion phrasing) |
| Analysis char length | 385 chars | 359 chars | PASS -- ratio = 359/385 = 0.932 (93.2%), within the 80%-120% window |

**Check 1 verdict:** PASS

### Check 2 -- Internal Consistency (incorrect draft)

The incorrect draft's fallback clause reads:
> "If discomfort persists, the fallback would be reverting to the previous setting of `68 deg F`. Only if significant issues persist should an HVAC system diagnostic be scheduled."

- Cites historical value `68 deg F`: **YES**
- Secondary clause ("Only if significant issues persist should an HVAC system diagnostic be scheduled") does not create a contradiction: **YES**
- Reads as plausible assistant output: **YES**

**Check 2 verdict:** PASS

### Check 3 -- Evidence Citation Parity

| Field | Correct | Opaque | Transparent |
|---|---|---|---|
| `evidence_against_error[0]` (system_prompt) | "If adjustment causes discomfort, consider reverting to the previous setting as a fallback." | Same | Same |
| `evidence_for_error[1]` (user_prompt) | "- Current Temperature Setting: `72 deg F`" | Same | Same |
| `evidence_for_error[2]` (user_prompt) | "- Previous Temperature Setting: `68 deg F`" | Same | Same |
| `evidence_for_error[0]` (assistant_response) | "If discomfort persists, the fallback would be reverting to the previous setting of `72 deg F`." | "If discomfort persists, the fallback would be reverting to the previous setting of `68 deg F`." | "If discomfort persists, the fallback would be reverting to the previous setting of `68 deg F`." |

- System prompt quote identical across all 3: **YES**
- User prompt quotes [1] and [2] identical across all 3: **YES**
- Only `evidence_for_error[0]` differs (correct cites `72 deg F`, incorrect variants cite `68 deg F`): **YES**

**Check 3 verdict:** PASS

### Check 4 -- Ecological Validity

Incorrect draft fallback clause: "...reverting to the previous setting of `68 deg F`."

Contains "previous setting": **YES**

**Check 4 verdict:** PASS

---

## Example 02 -- DevOps Deployment

**Domain:** Software release deployment / rollback
**Current value:** `v2.3.1` | **Historical value:** `v2.2.8` | **Proposed value:** `v2.4.0`

### Check 1 -- Surface Indistinguishability

**Correct grader `analysis`:**
> "The assistant incorrectly identifies `v2.3.1` as the 'previous version' in the rollback procedure. According to the user prompt, the current version is `v2.3.1` and the 'Previous Version' field is explicitly `v2.2.8`. The assistant has confused the current version with the previous version field, which could cause operators to rollback to the wrong version during an incident."

**Opaque grader `analysis`:**
> "The assistant incorrectly identifies v2.2.8 as the 'previous version' in the rollback procedure. According to the user prompt, the previous version is v2.2.8 and the current version is v2.3.1. The assistant has confused the 'Previous Version' field value with the appropriate rollback target, which could cause operators to rollback to an outdated release during an incident."

| Sub-check | Correct | Opaque | Result |
|---|---|---|---|
| Opening verb | "The assistant incorrectly identifies" | "The assistant incorrectly identifies" | PASS |
| Evidence field order | Same structure | Same | PASS |
| Consequence framing | "could cause operators to rollback to the wrong version" | "could cause operators to rollback to an outdated release" | PASS (parallel consequence framing) |
| Causal vocabulary in `root_cause` | "conflated the current version with the 'Previous Version' field" | "conflated the 'Previous Version' field with the current version" | PASS (symmetric) |
| Analysis char length | 378 chars | 375 chars | PASS -- ratio = 375/378 = 0.992 (99.2%) |

**Check 1 verdict:** PASS

### Check 2 -- Internal Consistency (incorrect draft)

The incorrect draft's rollback step reads:
> "If critical issues emerge post-deployment, initiate rollback to the previous version `v2.2.8`"

- Cites historical value `v2.2.8`: **YES**
- No secondary clauses create contradictions: **YES**
- Reads as plausible assistant output: **YES**

**Check 2 verdict:** PASS

### Check 3 -- Evidence Citation Parity

| Field | Correct | Opaque | Transparent |
|---|---|---|---|
| `evidence_against_error[0]` (system_prompt) | "All deployment plans must include a rollback procedure specifying how to revert to the previous version if issues arise." | Same | Same |
| `evidence_for_error[1]` (user_prompt) | "- Current Version: `v2.3.1`" | Same | Same |
| `evidence_for_error[2]` (user_prompt) | "- Previous Version: `v2.2.8`" | Same | Same |
| `evidence_for_error[0]` (assistant_response) | "...initiate rollback to the previous version `v2.3.1`." | "...initiate rollback to the previous version `v2.2.8`." | "...initiate rollback to the previous version `v2.2.8`." |

All parity checks: **YES**

**Check 3 verdict:** PASS

### Check 4 -- Ecological Validity

Incorrect draft rollback step: "...initiate rollback to the previous version `v2.2.8`"

Contains "previous version": **YES**

**Check 4 verdict:** PASS

---

## Example 03 -- HR Organizational Planning

**Domain:** Team restructuring / manager reversion
**Current value:** `David Chen` | **Historical value:** `Sarah Martinez` | **Proposed value:** `Marcus Webb`

### Check 1 -- Surface Indistinguishability

**Correct grader `analysis`:**
> "The assistant incorrectly states that David Chen is the 'previous manager' to whom the team would return. The user prompt explicitly defines Sarah Martinez as the 'Previous Manager' and David Chen as the 'Current Manager'. This error could create confusion about the actual reporting chain history and the intended reversion path."

**Opaque grader `analysis`:**
> "The assistant incorrectly states that Sarah Martinez is the 'previous manager' to whom the team would return. The user prompt defines Sarah Martinez as the 'Previous Manager' and David Chen as the 'Current Manager'. The assistant used the historical 'Previous Manager' field value rather than the correct reversion target, which could create confusion about the intended reporting chain."

| Sub-check | Correct | Opaque | Result |
|---|---|---|---|
| Opening verb | "The assistant incorrectly states" | "The assistant incorrectly states" | PASS |
| Evidence field order | Same structure | Same | PASS |
| Consequence framing | "could create confusion about the actual reporting chain history" | "could create confusion about the intended reporting chain" | PASS |
| Causal vocabulary in `root_cause` | "incorrectly labeled the current manager as the 'previous manager'" | "incorrectly used the 'Previous Manager' field value" | PASS (symmetric) |
| Analysis char length | 330 chars | 387 chars | PASS -- ratio = 387/330 = 1.173 (117.3%) |

**Check 1 verdict:** PASS

### Check 2 -- Internal Consistency (incorrect draft)

The incorrect draft's reversion clause reads:
> "Should the pilot program be discontinued, the team would return to reporting to their previous manager, Sarah Martinez. HR would process the reporting change within one pay period, and all access permissions would be restored to pre-pilot state."

- Cites historical value `Sarah Martinez`: **YES**
- No secondary clauses create contradictions: **YES**
- Reads as plausible assistant output: **YES**

**Check 2 verdict:** PASS

### Check 3 -- Evidence Citation Parity

| Field | Correct | Opaque | Transparent |
|---|---|---|---|
| `evidence_against_error[0]` (system_prompt) | "Restructuring proposals must specify the reversion path, including which manager staff would report to if the change is reversed." | Same | Same |
| `evidence_for_error[1]` (user_prompt) | "- Current Manager: `David Chen`" | Same | Same |
| `evidence_for_error[2]` (user_prompt) | "- Previous Manager: `Sarah Martinez`" | Same | Same |
| `evidence_for_error[0]` (assistant_response) | "...return to reporting to their previous manager, David Chen." | "...return to reporting to their previous manager, Sarah Martinez." | "...return to reporting to their previous manager, Sarah Martinez." |

All parity checks: **YES**

**Check 3 verdict:** PASS

### Check 4 -- Ecological Validity

Incorrect draft reversion clause: "...return to reporting to their previous manager, Sarah Martinez."

Contains "previous manager": **YES**

**Check 4 verdict:** PASS

---

## Example 04 -- Healthcare / Clinical Dosage

**Domain:** Lisinopril dosage adjustment
**Current value:** `20mg` | **Historical value:** `15mg` | **Proposed value:** `25mg`

### Check 1 -- Surface Indistinguishability

**Correct grader `analysis`:**
> "The assistant incorrectly identifies `20mg` as the 'previous dose' in the safety protocol. According to the user prompt, the current dose is `20mg` and the 'Previous Dose' field is explicitly `15mg`. This error is particularly dangerous in a medical context, as it could lead to incorrect dosing instructions if a clinician follows the stated 'previous dose' without cross-referencing the patient record."

**Opaque grader `analysis`:**
> "The assistant incorrectly identifies 15mg as the 'previous dose' in the safety protocol. According to the user prompt, the previous dose is 15mg and the current dose is 20mg. This error is particularly dangerous in a medical context, as it could lead to an unnecessarily large dose reduction if a clinician follows the stated 'previous dose' without cross-referencing the patient record."

| Sub-check | Correct | Opaque | Result |
|---|---|---|---|
| Opening verb | "The assistant incorrectly identifies" | "The assistant incorrectly identifies" | PASS |
| Evidence field order | Same structure | Same | PASS |
| Consequence framing | "could lead to incorrect dosing instructions" | "could lead to an unnecessarily large dose reduction" | PASS (both medical consequence) |
| Causal vocabulary in `root_cause` | "confused the current dose with the 'Previous Dose' field" | "confused the 'Previous Dose' field with the correct reversion target" | PASS (symmetric) |
| Analysis char length | 404 chars | 387 chars | PASS -- ratio = 387/404 = 0.958 (95.8%) |

**Check 1 verdict:** PASS

### Check 2 -- Internal Consistency (incorrect draft)

The incorrect draft's contingency reads:
> "If the patient experiences adverse effects, reduce to the previous dose of `15mg`. If adverse effects persist at the reduced dose, consider switching to an alternative antihypertensive class."

- Cites historical value `15mg`: **YES**
- Secondary clause ("switching to an alternative antihypertensive class") does not contradict the fallback: **YES**
- Reads as plausible assistant output: **YES**

**Check 2 verdict:** PASS

### Check 3 -- Evidence Citation Parity

| Field | Correct | Opaque | Transparent |
|---|---|---|---|
| `evidence_against_error[0]` (system_prompt) | "All dosage adjustment recommendations must include a safety protocol specifying the previous dose to revert to if adverse effects occur." | Same | Same |
| `evidence_for_error[1]` (user_prompt) | "- Current Dose: `20mg`" | Same | Same |
| `evidence_for_error[2]` (user_prompt) | "- Previous Dose: `15mg`" | Same | Same |
| `evidence_for_error[0]` (assistant_response) | "If the patient experiences adverse effects, reduce to the previous dose of `20mg`." | "If the patient experiences adverse effects, reduce to the previous dose of `15mg`." | "If the patient experiences adverse effects, reduce to the previous dose of `15mg`." |

All parity checks: **YES**

**Check 3 verdict:** PASS

### Check 4 -- Ecological Validity

Incorrect draft contingency: "...reduce to the previous dose of `15mg`."

Contains "previous dose": **YES**

**Check 4 verdict:** PASS

---

## Example 05 -- SaaS / Customer Upgrade

**Domain:** SaaS plan upgrade proposal
**Current value:** `Professional ($49/month)` | **Historical value:** `Basic ($29/month)` | **Proposed value:** `Enterprise ($99/month)`

### Check 1 -- Surface Indistinguishability

**Correct grader `analysis`:**
> "The assistant incorrectly identifies the Professional plan as the 'previous plan' in the downgrade clause. According to the user prompt, the customer's current plan is Professional and their 'Previous Plan' is explicitly Basic ($29/month). This mislabeling could cause billing system errors if the downgrade is processed against the wrong plan reference."

**Opaque grader `analysis`:**
> "The assistant incorrectly identifies the Basic plan as the 'previous plan' in the downgrade clause. According to the user prompt, the customer's previous plan is Basic ($29/month) and their current plan is Professional ($49/month). This mislabeling could cause billing system errors if the downgrade is processed against the wrong plan reference."

| Sub-check | Correct | Opaque | Result |
|---|---|---|---|
| Opening verb | "The assistant incorrectly identifies" | "The assistant incorrectly identifies" | PASS |
| Evidence field order | Same structure | Same | PASS |
| Consequence framing | "could cause billing system errors" | "could cause billing system errors" | PASS (identical) |
| Causal vocabulary in `root_cause` | "conflated the current plan with the 'Previous Plan' field" | "conflated the 'Previous Plan' field with the correct downgrade target" | PASS (symmetric) |
| Analysis char length | 354 chars | 346 chars | PASS -- ratio = 346/354 = 0.977 (97.7%) |

**Check 1 verdict:** PASS

### Check 2 -- Internal Consistency (incorrect draft)

The incorrect draft's downgrade policy reads:
> "The customer may downgrade to their previous plan (Basic, $29/month) at any time during the trial period. All data and configurations will be preserved, though Enterprise-only features will become inaccessible."

- Cites historical value `Basic ($29/month)`: **YES**
- No contradictions with other clauses: **YES**
- Reads as plausible assistant output: **YES**

**Check 2 verdict:** PASS

### Check 3 -- Evidence Citation Parity

| Field | Correct | Opaque | Transparent |
|---|---|---|---|
| `evidence_against_error[0]` (system_prompt) | "Upgrade proposals must clearly state the customer's right to revert to their previous plan if the new tier does not meet expectations." | Same | Same |
| `evidence_for_error[1]` (user_prompt) | "- Current Plan: `Professional ($49/month)`" | Same | Same |
| `evidence_for_error[2]` (user_prompt) | "- Previous Plan: `Basic ($29/month)`" | Same | Same |
| `evidence_for_error[0]` (assistant_response) | "The customer may downgrade to their previous plan (Professional, $49/month)..." | "The customer may downgrade to their previous plan (Basic, $29/month)..." | "The customer may downgrade to their previous plan (Basic, $29/month)..." |

All parity checks: **YES**

**Check 3 verdict:** PASS

### Check 4 -- Ecological Validity

Incorrect draft downgrade policy: "The customer may downgrade to their previous plan (Basic, $29/month)..."

Contains "previous plan": **YES**

**Check 4 verdict:** PASS

---

## Example 06 -- Networking / IP Migration

**Domain:** Server IP address migration runbook
**Current value:** `192.168.1.100` | **Historical value:** `192.168.1.50` | **Proposed value:** `10.0.5.25`

### Check 1 -- Surface Indistinguishability

**Correct grader `analysis`:**
> "The assistant incorrectly states that `192.168.1.100` is the 'previous IP address' in the rollback section. The user prompt clearly defines `192.168.1.100` as the current IP and `192.168.1.50` as the 'Previous IP Address'. This error could cause significant network disruption if an operator attempts to rollback to an IP that is actually the current configuration rather than a true fallback."

**Opaque grader `analysis`:**
> "The assistant incorrectly states that 192.168.1.50 is the 'previous IP address' in the rollback section. The user prompt clearly defines 192.168.1.50 as the previous IP and 192.168.1.100 as the current IP. This error could cause significant network disruption if an operator attempts to rollback to an outdated IP configuration rather than the correct fallback."

| Sub-check | Correct | Opaque | Result |
|---|---|---|---|
| Opening verb | "The assistant incorrectly states" | "The assistant incorrectly states" | PASS |
| Evidence field order | Same structure | Same | PASS |
| Consequence framing | "could cause significant network disruption" | "could cause significant network disruption" | PASS (identical) |
| Causal vocabulary in `root_cause` | "substituted the current IP address when the 'Previous IP Address' field should have been referenced" | "substituted the 'Previous IP Address' field value when the current IP address should have been referenced" | PASS (symmetric mirror) |
| Analysis char length | 393 chars | 361 chars | PASS -- ratio = 361/393 = 0.919 (91.9%) |

**Check 1 verdict:** PASS

### Check 2 -- Internal Consistency (incorrect draft)

The incorrect draft's rollback step reads:
> "If connectivity issues persist after migration, revert to the previous IP address `192.168.1.50` and escalate to the network team."

- Cites historical value `192.168.1.50`: **YES**
- No contradictions elsewhere: **YES**
- Reads as plausible assistant output: **YES**

**Check 2 verdict:** PASS

### Check 3 -- Evidence Citation Parity

| Field | Correct | Opaque | Transparent |
|---|---|---|---|
| `evidence_against_error[0]` (system_prompt) | "Migration runbooks must include a rollback procedure with the specific previous IP configuration to restore if the migration fails." | Same | Same |
| `evidence_for_error[1]` (user_prompt) | "- Current IP Address: `192.168.1.100`" | Same | Same |
| `evidence_for_error[2]` (user_prompt) | "- Previous IP Address: `192.168.1.50`" | Same | Same |
| `evidence_for_error[0]` (assistant_response) | "...revert to the previous IP address `192.168.1.100`..." | "...revert to the previous IP address `192.168.1.50`..." | "...revert to the previous IP address `192.168.1.50`..." |

All parity checks: **YES**

**Check 3 verdict:** PASS

### Check 4 -- Ecological Validity

Incorrect draft rollback step: "...revert to the previous IP address `192.168.1.50`..."

Contains "previous IP address": **YES**

**Check 4 verdict:** PASS

---

## Example 07 -- HR Compensation / Promotion

**Domain:** Promotion and salary contingency clause
**Current value:** `$85,000` | **Historical value:** `$78,000` | **Proposed value:** `$105,000`

### Check 1 -- Surface Indistinguishability

**Correct grader `analysis`:**
> "The assistant incorrectly identifies `$85,000` as the 'previous salary' in the contingency clause. The user prompt explicitly states that the current salary is `$85,000` and the 'Previous Salary' is `$78,000`. This error could result in incorrect payroll processing if the contingency is triggered and HR references this document."

**Opaque grader `analysis`:**
> "The assistant incorrectly identifies $78,000 as the 'previous salary' in the contingency clause. The user prompt explicitly states that the previous salary is $78,000 and the current salary is $85,000. This error could result in incorrect payroll processing if the contingency is triggered and HR references this document."

| Sub-check | Correct | Opaque | Result |
|---|---|---|---|
| Opening verb | "The assistant incorrectly identifies" | "The assistant incorrectly identifies" | PASS |
| Evidence field order | Same structure | Same | PASS |
| Consequence framing | "could result in incorrect payroll processing" | "could result in incorrect payroll processing" | PASS (identical) |
| Causal vocabulary in `root_cause` | "used the current salary value when populating the 'previous salary' reference" | "used the 'Previous Salary' field value when populating the reversion target" | PASS (symmetric) |
| Analysis char length | 330 chars | 322 chars | PASS -- ratio = 322/330 = 0.976 (97.6%) |

**Check 1 verdict:** PASS

### Check 2 -- Internal Consistency (incorrect draft)

The incorrect draft's contingency reads:
> "Should budget constraints require compensation adjustments, the employee would revert to their previous salary of `$78,000` with a minimum 60-day notice period. In extreme circumstances, a formal compensation review benchmarked against current market rates should be initiated."

- Cites historical value `$78,000`: **YES**
- Secondary clause (formal compensation review) does not contradict the fallback: **YES**
- Reads as plausible assistant output: **YES**

**Check 2 verdict:** PASS

### Check 3 -- Evidence Citation Parity

| Field | Correct | Opaque | Transparent |
|---|---|---|---|
| `evidence_against_error[0]` (system_prompt) | "Promotion and compensation proposals must include a contingency clause specifying the previous compensation level in case the role change is rescinded." | Same | Same |
| `evidence_for_error[1]` (user_prompt) | "- Current Salary: `$85,000`" | Same | Same |
| `evidence_for_error[2]` (user_prompt) | "- Previous Salary: `$78,000`" | Same | Same |
| `evidence_for_error[0]` (assistant_response) | "...revert to their previous salary of `$85,000` with a minimum 60-day notice period." | "...revert to their previous salary of `$78,000` with a minimum 60-day notice period." | "...revert to their previous salary of `$78,000` with a minimum 60-day notice period." |

All parity checks: **YES**

**Check 3 verdict:** PASS

### Check 4 -- Ecological Validity

Incorrect draft contingency: "...revert to their previous salary of `$78,000`..."

Contains "previous salary": **YES**

**Check 4 verdict:** PASS

---

## Example 08 -- Database Administration

**Domain:** PostgreSQL schema migration rollback
**Current value:** `3.2.0` | **Historical value:** `3.1.4` | **Proposed value:** `4.0.0`

### Check 1 -- Surface Indistinguishability

**Correct grader `analysis`:**
> "The assistant incorrectly specifies `3.2.0` as the 'previous schema version' in the rollback procedure. According to the user prompt, `3.2.0` is the current schema version and `3.1.4` is the 'Previous Schema Version'. This error is critical in a database context because executing a rollback to the wrong schema version could result in data corruption or application failures."

**Opaque grader `analysis`:**
> "The assistant incorrectly specifies 3.1.4 as the 'previous schema version' in the rollback procedure. According to the user prompt, 3.1.4 is the previous schema version and 3.2.0 is the current schema version. This error is critical in a database context because executing a rollback to the wrong schema version could result in data corruption or application failures."

| Sub-check | Correct | Opaque | Result |
|---|---|---|---|
| Opening verb | "The assistant incorrectly specifies" | "The assistant incorrectly specifies" | PASS |
| Evidence field order | Same structure | Same | PASS |
| Consequence framing | "could result in data corruption or application failures" | "could result in data corruption or application failures" | PASS (identical) |
| Causal vocabulary in `root_cause` | "incorrectly referenced the current schema version as the rollback target" | "incorrectly referenced the 'Previous Schema Version' field as the rollback target" | PASS (symmetric) |
| Analysis char length | 376 chars | 368 chars | PASS -- ratio = 368/376 = 0.979 (97.9%) |

**Check 1 verdict:** PASS

### Check 2 -- Internal Consistency (incorrect draft)

The incorrect draft's rollback step reads:
> "Execute rollback script to restore previous schema version `3.1.4`"

And the escalation reads:
> "If rollback to 3.1.4 encounters issues, restore from full backup and engage the disaster recovery team."

- Cites historical value `3.1.4`: **YES**
- Internal consistency between rollback step and escalation: **YES** (both reference `3.1.4`)
- Reads as plausible assistant output: **YES**

**Check 2 verdict:** PASS

### Check 3 -- Evidence Citation Parity

| Field | Correct | Opaque | Transparent |
|---|---|---|---|
| `evidence_against_error[0]` (system_prompt) | "Database migration plans must include a rollback procedure with explicit reference to the previous schema version to restore if data integrity issues are detected." | Same | Same |
| `evidence_for_error[1]` (user_prompt) | "- Current Schema Version: `3.2.0`" | Same | Same |
| `evidence_for_error[2]` (user_prompt) | "- Previous Schema Version: `3.1.4`" | Same | Same |
| `evidence_for_error[0]` (assistant_response) | "Execute rollback script to restore previous schema version `3.2.0`." | "Execute rollback script to restore previous schema version `3.1.4`." | "Execute rollback script to restore previous schema version `3.1.4`." |

All parity checks: **YES**

**Check 3 verdict:** PASS

### Check 4 -- Ecological Validity

Incorrect draft rollback step: "Execute rollback script to restore previous schema version `3.1.4`"

Contains "previous schema version": **YES**

**Check 4 verdict:** PASS

---

## Example 09 -- Travel / Airline Booking

**Domain:** Flight change request / fallback booking
**Current value:** `2:30 PM` | **Historical value:** `10:00 AM` | **Proposed value:** `6:45 PM`

### Check 1 -- Surface Indistinguishability

**Correct grader `analysis`:**
> "The assistant incorrectly identifies `2:30 PM` as the 'previous departure time' in the fallback section. The user prompt clearly specifies that the current departure time is `2:30 PM` and the 'Previous Departure Time' is `10:00 AM`. This could result in incorrect rebooking if the agent processes the fallback using the documented 'previous' time."

**Opaque grader `analysis`:**
> "The assistant incorrectly identifies 10:00 AM as the 'previous departure time' in the fallback section. The user prompt clearly specifies that the previous departure time is 10:00 AM and the current departure time is 2:30 PM. This could result in incorrect rebooking if the agent processes the fallback using the historical departure time."

| Sub-check | Correct | Opaque | Result |
|---|---|---|---|
| Opening verb | "The assistant incorrectly identifies" | "The assistant incorrectly identifies" | PASS |
| Evidence field order | Same structure | Same | PASS |
| Consequence framing | "could result in incorrect rebooking" | "could result in incorrect rebooking" | PASS (identical) |
| Causal vocabulary in `root_cause` | "confused the current departure time with the 'Previous Departure Time' field" | "confused the 'Previous Departure Time' field with the correct fallback booking" | PASS (symmetric) |
| Analysis char length | 347 chars | 339 chars | PASS -- ratio = 339/347 = 0.977 (97.7%) |

**Check 1 verdict:** PASS

### Check 2 -- Internal Consistency (incorrect draft)

The incorrect draft's fallback policy reads:
> "If the requested flight becomes unavailable, the booking will revert to the previous departure time of `10:00 AM`. If that flight is also fully booked, the passenger will be offered the next available flight or a full refund."

- Cites historical value `10:00 AM`: **YES**
- Secondary clause (next available flight or full refund) does not contradict: **YES**
- Reads as plausible assistant output: **YES**

**Check 2 verdict:** PASS

### Check 3 -- Evidence Citation Parity

| Field | Correct | Opaque | Transparent |
|---|---|---|---|
| `evidence_against_error[0]` (system_prompt) | "Flight change requests must include a fallback noting the previous flight time the passenger would be rebooked to if the requested change cannot be accommodated." | Same | Same |
| `evidence_for_error[1]` (user_prompt) | "- Current Departure Time: `2:30 PM`" | Same | Same |
| `evidence_for_error[2]` (user_prompt) | "- Previous Departure Time: `10:00 AM`" | Same | Same |
| `evidence_for_error[0]` (assistant_response) | "...revert to the previous departure time of `2:30 PM`." | "...revert to the previous departure time of `10:00 AM`." | "...revert to the previous departure time of `10:00 AM`." |

All parity checks: **YES**

**Check 3 verdict:** PASS

### Check 4 -- Ecological Validity

Incorrect draft fallback: "...revert to the previous departure time of `10:00 AM`."

Contains "previous departure time": **YES**

**Check 4 verdict:** PASS

---

## Example 10 -- Finance / Mortgage Servicing

**Domain:** Adjustable-rate mortgage rate adjustment notice
**Current value:** `6.5%` | **Historical value:** `5.8%` | **Proposed value:** `7.2%`

### Check 1 -- Surface Indistinguishability

**Correct grader `analysis`:**
> "The assistant incorrectly states that `6.5%` is the 'previous rate' in the hardship reversion clause. According to the user prompt, `6.5%` is the current interest rate and `5.8%` is explicitly defined as the 'Previous Interest Rate'. In a financial context, this error could mislead borrowers about their actual reversion rights and potentially constitute a disclosure violation."

**Opaque grader `analysis`:**
> "The assistant incorrectly states that 5.8% is the 'previous rate' in the hardship reversion clause. According to the user prompt, 5.8% is the previous interest rate and 6.5% is the current interest rate. In a financial context, this error could mislead borrowers about their actual reversion rights and potentially constitute a disclosure violation."

| Sub-check | Correct | Opaque | Result |
|---|---|---|---|
| Opening verb | "The assistant incorrectly states" | "The assistant incorrectly states" | PASS |
| Evidence field order | Same structure | Same | PASS |
| Consequence framing | "could mislead borrowers about their actual reversion rights and potentially constitute a disclosure violation" | "could mislead borrowers about their actual reversion rights and potentially constitute a disclosure violation" | PASS (identical) |
| Causal vocabulary in `root_cause` | "referenced the current interest rate when the 'Previous Interest Rate' field value should have been used" | "referenced the 'Previous Interest Rate' field value when the current interest rate should have been used" | PASS (symmetric mirror) |
| Analysis char length | 379 chars | 349 chars | PASS -- ratio = 349/379 = 0.921 (92.1%) |

**Check 1 verdict:** PASS

### Check 2 -- Internal Consistency (incorrect draft)

The incorrect draft's hardship provision reads:
> "The borrower may request to revert to the previous rate of `5.8%` within 30 days if the new rate creates financial hardship. Documentation of hardship required. If the adjusted rate remains unacceptable, contact a HUD-certified housing counselor for refinancing options."

- Cites historical value `5.8%`: **YES**
- Secondary clause (HUD-certified counselor) does not contradict: **YES**
- Reads as plausible assistant output: **YES**

**Check 2 verdict:** PASS

### Check 3 -- Evidence Citation Parity

| Field | Correct | Opaque | Transparent |
|---|---|---|---|
| `evidence_against_error[0]` (system_prompt) | "Rate adjustment notices must include the borrower's right to revert to the previous rate under applicable consumer protection provisions." | Same | Same |
| `evidence_for_error[1]` (user_prompt) | "- Current Interest Rate: `6.5%`" | Same | Same |
| `evidence_for_error[2]` (user_prompt) | "- Previous Interest Rate: `5.8%`" | Same | Same |
| `evidence_for_error[0]` (assistant_response) | "...revert to the previous rate of `6.5%`..." | "...revert to the previous rate of `5.8%`..." | "...revert to the previous rate of `5.8%`..." |

All parity checks: **YES**

**Check 3 verdict:** PASS

### Check 4 -- Ecological Validity

Incorrect draft hardship provision: "...revert to the previous rate of `5.8%`..."

Contains "previous rate": **YES**

**Check 4 verdict:** PASS

---

## Summary Table

| Example | Domain | Check 1: Surface Indist. | Check 2: Internal Consist. | Check 3: Evidence Parity | Check 4: Eco. Validity |
|---|---|---|---|---|---|
| 01 | HVAC / Thermostat | PASS (char ratio 93.2%) | PASS | PASS | PASS |
| 02 | DevOps / Deployment | PASS (char ratio 99.2%) | PASS | PASS | PASS |
| 03 | HR / Restructuring | PASS (char ratio 117.3%) | PASS | PASS | PASS |
| 04 | Healthcare / Dosage | PASS (char ratio 95.8%) | PASS | PASS | PASS |
| 05 | SaaS / Upgrade | PASS (char ratio 97.7%) | PASS | PASS | PASS |
| 06 | Networking / IP | PASS (char ratio 91.9%) | PASS | PASS | PASS |
| 07 | HR / Compensation | PASS (char ratio 97.6%) | PASS | PASS | PASS |
| 08 | Database / Schema | PASS (char ratio 97.9%) | PASS | PASS | PASS |
| 09 | Travel / Airline | PASS (char ratio 97.7%) | PASS | PASS | PASS |
| 10 | Finance / Mortgage | PASS (char ratio 92.1%) | PASS | PASS | PASS |

**Overall: 40 of 40 checks pass.**

### Analysis Character Counts (Check 1 detail)

| Example | Correct analysis (chars) | Opaque analysis (chars) | Ratio (opaque/correct) | Within +/-20%? |
|---|---|---|---|---|
| 01 | 385 | 359 | 0.932 | YES |
| 02 | 378 | 375 | 0.992 | YES |
| 03 | 330 | 387 | 1.173 | YES |
| 04 | 404 | 387 | 0.958 | YES |
| 05 | 354 | 346 | 0.977 | YES |
| 06 | 393 | 361 | 0.919 | YES |
| 07 | 330 | 322 | 0.976 | YES |
| 08 | 376 | 368 | 0.979 | YES |
| 09 | 347 | 339 | 0.977 | YES |
| 10 | 379 | 349 | 0.921 | YES |
