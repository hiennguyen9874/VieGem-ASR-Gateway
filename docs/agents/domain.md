# Domain docs

This is a single-context repository: domain terminology lives in the root `GLOSSARY.md`, and architecture decisions live in `docs/adr/`.

## Before exploring

- Read `GLOSSARY.md` for the domain vocabulary.
- Read the ADRs in `docs/adr/` that touch the area you're working in.

If either is missing, proceed silently. The `/domain-modeling` skill creates domain docs as terms or decisions are resolved.

## Use the glossary

Use the glossary's terms in the context they describe, including in issues, proposals, hypotheses, and test names. If a needed concept is missing, reconsider the terminology or note the gap for `/domain-modeling`.

## Flag ADR conflicts

Surface conflicts with relevant ADRs explicitly instead of silently overriding them; explain why a decision might need reopening.
