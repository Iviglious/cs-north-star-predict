# User Story — Northstar Desk Triage Assistant

## Who uses it?

**Primary user:** A **frontline support agent** on Northstar Desk's first-response team. They handle incoming cases from email, webchat, phone, and in-app channels.

**Secondary user:** A **team lead** who spot-checks routing quality and uses similar-case examples in coaching sessions.

## What problem does it solve?

When a new case arrives, agents often face three questions under time pressure:

1. **Who should own this?** (billing, engineering, support, security…)
2. **How urgent is it?** (especially when the customer hasn't picked a priority)
3. **Have we seen this before?** (and if so, what fixed it?)

Today, agents rely on memory, Slack searches, or asking a colleague. That works — but it's slow, inconsistent, and hard for new starters.

The Triage Assistant **doesn't decide for them**. It gives a **starting point**: suggested route, risk flags, and 3 similar past cases with outcomes — so the agent can act faster with more context.

## Day-in-the-life workflow

### Morning queue review

Sarah opens the triage queue. A new email arrives:

> *"CSV export is coming out blank for the sales report widget."*

She opens the **Northstar Desk Triage Assistant**, pastes the summary, selects **webchat** and **Enterprise**, and clicks **Suggest routing**.

The tool returns:

- **Route suggestion:** engineering · bug · High priority
- **Escalation risk:** moderate — keyword `blank` detected
- **Similar case ND-2025-002008:** same export issue, escalated to engineering, still open

Sarah routes to engineering with High priority — but because she saw the similar case, she adds a note: *"Check widget export pipeline — seen before in ND-2025-002008."*

**Time saved:** ~3–5 minutes of searching. **Quality gain:** consistent routing + institutional memory.

### High-risk billing case

A phone case comes in:

> *"Urgent: credit card failed for Orbit ID renewal and account is about to lock."*

The tool flags:

- **billing** team, **Urgent** priority
- **Escalation risk:** 39%
- **Flags:** `failed`, `urgency_language`
- **Similar case ND-2025-000407:** same issue, solved in 3.8 hours as `fixed`

Sarah prioritises it immediately and uses the similar case resolution as a playbook.

### When the agent overrides the tool

A VAT receipt request is suggested as **billing / Medium** — Sarah confirms and routes it. No escalation needed. The similar cases show this pattern resolves quickly with `answered`.

If the tool suggested **engineering** at 42% confidence, Sarah would treat it as a weak hint and use her judgement — the confidence score tells her when to trust vs override.

## What success looks like

| Metric | Expected impact |
|---|---|
| Time to first route decision | Reduced (less searching) |
| Routing consistency | Improved (especially for new agents) |
| Escalation surprises | Reduced (early risk flags) |
| Agent confidence | Higher (similar cases as evidence) |

## What it deliberately does *not* do

- Does **not** auto-close or auto-respond to customers
- Does **not** replace the case management system
- Does **not** hide uncertainty — low confidence is shown explicitly
- Does **not** use customer demographics in suggestions

## Elevator pitch (30 seconds)

> "When a new case lands, paste the summary into the Triage Assistant. In seconds you get a routing suggestion, escalation risk, and three similar cases showing how we handled it before. You stay in control — the tool just helps you decide faster with evidence."
