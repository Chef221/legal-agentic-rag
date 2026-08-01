# Private Submission Checklist

## Before packaging

- [ ] Official question file identity and SHA-256 recorded.
- [ ] Official artifact validation passed with exact corpus lineage.
- [ ] Code commit and working tree state recorded.
- [ ] Application config hash recorded; no secret present.
- [ ] Every active model has immutable revision, verified license and BTC
      registration evidence.
- [ ] Batch is complete, ordered, checksum-valid and contains one result per ID.
- [ ] No warm-up/reference answer leaked into prediction input.
- [ ] Data Statement and Model Card snapshots completed.

## Package preflight

- [ ] Output filename is exactly `submission.zip`.
- [ ] ZIP contains only UTF-8 `submission.json`.
- [ ] Root is an ID-keyed JSON object; every value is exactly `{"answer": string}`.
- [ ] No missing, extra, duplicate or reordered ID.
- [ ] No internal citation marker, trace, warning or metadata field leaked.
- [ ] Final archive SHA-256 recorded.
- [ ] Local diagnostic scoring/report completed when references are available.

## Upload gate

- [ ] Correct Task 2 organizer portal verified.
- [ ] Competition phase and deadline verified.
- [ ] Submission ledger checked: fewer than 3 private submissions today.
- [ ] Team leader account and team name verified.
- [ ] One teammate independently reviewed archive identity and checklist.

## After upload

- [ ] Organizer submission ID, timestamp and status recorded in ledger.
- [ ] Portal response/screenshot retained outside Git if needed.
- [ ] Failed upload also recorded because quota behavior may require BTC
      clarification.
- [ ] No archive is silently replaced; next attempt uses a new release ID.
