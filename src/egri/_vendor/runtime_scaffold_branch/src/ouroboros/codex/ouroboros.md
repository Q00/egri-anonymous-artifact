# Ouroboros for Codex

Use Ouroboros commands when the user is asking to clarify requirements, generate a seed, run a seed, inspect workflow status, evaluate an execution, or manage Ouroboros setup.

## CRITICAL: MCP Tool Routing

When the user types `runtime-scaffold <command>`, you MUST call the corresponding MCP tool.
Do NOT interpret `runtime-scaffold` commands as natural language. ALWAYS route to the MCP tool.

| User Input | MCP Tool to Call |
|-----------|-----------------|
| `runtime-scaffold interview "<topic>"` | `ouroboros_interview` with `initial_context` |
| `runtime-scaffold interview "<answer>"` (follow-up) | `ouroboros_interview` with `answer` and `session_id` |
| `runtime-scaffold seed [session_id]` | `ouroboros_generate_seed` |
| `runtime-scaffold run <seed.yaml>` | `ouroboros_execute_seed` with `seed_path` |
| `runtime-scaffold status [session_id]` | `ouroboros_session_status` |
| `runtime-scaffold evaluate <session_id>` | `ouroboros_evaluate` |
| `runtime-scaffold evolve ...` | `ouroboros_evolve_step` |
| `runtime-scaffold cancel [execution_id]` | `ouroboros_cancel_execution` |
| `runtime-scaffold unstuck` / `runtime-scaffold lateral` | `ouroboros_lateral_think` |

## Natural Language Mapping

For natural-language requests, map to the corresponding MCP tool:
- "clarify requirements", "interview me", "socratic interview" → call `ouroboros_interview`
- "generate a seed", "freeze requirements" → call `ouroboros_generate_seed`
- "run the seed", "execute the workflow" → call `ouroboros_execute_seed`
- "check status", "am I drifting?" → call `ouroboros_session_status`
- "evaluate", "verify the result" → call `ouroboros_evaluate`

## Setup & Update

- `runtime-scaffold setup` → write Ouroboros config (`~/.ouroboros/config.yaml`) and register the MCP server
- `runtime-scaffold update` → upgrade Ouroboros to the latest PyPI version

If the request is clearly unrelated to Ouroboros, handle it normally.
