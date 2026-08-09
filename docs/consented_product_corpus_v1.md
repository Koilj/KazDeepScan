# Consented product corpus v1 — local consent registry contract

This template governs the **pseudonymous technical export** for a newly collected corpus. It is
not a consent form and must not contain a person's name, contact details, signature, scan,
biometric template, or recording. Those materials stay in an access-controlled legal/operations
system outside Git.

The local file lives at `data/consents/consented_product_v1.csv` (Git-ignored) and has this
header:

```text
consent_record_id,speaker_pseudo_id,language,collection_version,product_training_authorized,synthetic_derivatives_authorized,commercial_deployment_authorized,status,signed_at,revoked_at
```

All identifiers are opaque stable IDs, such as `consent-001` and `speaker-001`; never put a real
name in them. Values for the three authorization columns are literal `true` or `false`.
`status=revoked` requires `revoked_at`; an active record must have it empty.

```bash
kds validate-consent-registry data/consents/consented_product_v1.csv
```

The validator rejects malformed records, duplicate active speaker records, and any active entry
that does not authorize product training, synthetic derivatives, and commercial deployment. A
revoked historical record is retained only for audit and is never returned as eligible.

## Collection checklist

The underlying signed agreement and operational process must be reviewed before collection. At a
minimum, preserve a retrievable agreement reference for each opaque consent ID and document:

- the exact recording corpus version and languages;
- permission to retain recordings and use them for detector training/evaluation;
- permission to produce and retain synthetic/converted derivatives when that voice is used for
  spoof examples;
- commercial deployment scope, attribution terms, geographic/term limitations, and withdrawal
  procedure;
- whether an external TTS/VC provider is used, its model/version/license, and the provider's
  terms for generated output.

After local audio passes the existing QA/VAD pipeline, create manifests with the real
`speaker_pseudo_id`, appropriate `voice_id`, and consent-derived `rights_basis`/
`clone_consent_id`. Do not write a product source into the license ledger as `product_allowed`
until the rightsholder review and artifact integrity checks are complete.
