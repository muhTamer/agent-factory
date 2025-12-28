# Blueprint Prompt for `complaint` (auto-generated)
Capability: complaint
Vertical: fintech
Mode: offline-fallback

Goals:
- Generate an agent that implements the IAgent contract (load, handle, metadata).
- Use shared libraries per `entrypoint` where possible.
- Never write files outside its own generated/<id>/ directory at runtime.
- No network calls unless via declared tools.

Inputs (declared): ['policy.yaml', 'sop.md']
Output contract: {'eligibility': 'string', 'steps': 'list', 'ticket': 'object'}
Entrypoint: app.shared.fsm.build_agent