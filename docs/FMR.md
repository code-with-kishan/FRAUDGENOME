# Failure Mode & Remediation (FMR) Template

Purpose: quick, repeatable Failure Mode & Remediation notes for security/compliance reviews.

- Title: Brief title of the failure mode
- Date: YYYY-MM-DD
- Reporter: Name / Role
- Severity: Low / Medium / High / Critical

Description:

- Summary: one-paragraph description of the failure or gap
- Affected components: list of services, models, datasets, endpoints
- Root cause: technical cause

Impact:

- User-facing impact
- Security/compliance impact
- Estimated affected records / accounts

Remediation Plan:

1. Immediate mitigation: steps to stop bleeding (e.g., disable endpoint, revoke token)
2. Short-term fix: patch, config change, toggle
3. Long-term fix: architectural or process change
4. Validation steps: tests, audits, monitoring to confirm fix

Owner & timeline:

- Owner: name
- ETA: date

Notes & Audit Trail:

- Links to logs, ticket numbers, PRs, and any artifacts
