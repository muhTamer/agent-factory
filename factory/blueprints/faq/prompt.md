# Blueprint Prompt for `faq` (auto-generated)
Capability: faq
Vertical: fintech
Mode: offline-fallback

Goals:
- Generate an agent that implements the IAgent contract (load, handle, metadata).
- Use shared libraries per `entrypoint` where possible.
- Never write files outside its own generated/<id>/ directory at runtime.
- No network calls unless via declared tools.

Inputs (declared): ['csv', 'md', 'txt']
Output contract: {'answer': 'string', 'citations': 'list', 'score': 'float'}
Entrypoint: app.shared.rag.build_agent