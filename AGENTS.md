## Agent skills

### Issue tracker

Issues and specs live as local Markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/` (created lazily). See `docs/agents/domain.md`.

## Engineering rules

BlueprintDL is a personal experiment operating system: a small, familiar core that constrains how ideas enter, not a client delivery artifact. Delivery is a separate minimal package.

### Core contracts (keep these small)

1. Config schema (`type + params`, project/name, imports)
2. Registry
3. Run identity (`runs/<project>/<experiment>/<timestamp>/`, `run.json`)
4. Trainer / hooks / lifecycle
5. Extensions and `imports` (project-specific components and Recipes)
6. Experiment tools: `visualize`, `compare`, `project`

Everything else is an extension and does not need to grow the core.

### Rules for changes

- An idea enters as one of: config change, registered component, Recipe, or a `scratch/` prototype. Never add a new `train_*.py` or a second training entry point.
- New top-level patterns or new config semantics require an ADR and explicit user approval before implementation.
- Every change includes a test or a parity check; prefer the smallest diff that proves the behavior.
- AI implements within existing patterns. The user owns the spec, interface decisions, final review, and makes at least one hands-on modification per feature.
- If a change makes experiments slower or less reproducible, simplify or revert before adding more features.

