# Research session contract

Brilliance's client is a view of a durable research session, not the owner of scientific work.
The current web workbench uses the same `ResearchChatService` and project archive as the
configured authoring adapter and managed numerical workers. No MBA-specific route belongs here.
New topics continue to enter through expert modules and the generic accelerator API.

## Ownership

- The scientist owns requirements and may steer the active session with any message.
- The model selects methods, reads local expertise, authors Python, and interprets evidence.
- The configured runner owns interpreter/Julia/JuTrack setup and execution receipts.
- Numerical jobs own their lifetime, progress, checkpoints, and cooperative cancellation.
- The client renders public text, source, evidence, and activity. It never turns a progress
  counter or a stable candidate into a scientific success claim.

## Native-provider turn semantics

Recent public user/assistant messages are included with the current request, expert notes,
and saved lattice/result references. This history is bounded to 24,000 characters; canonical
project state is supplied separately. Raw traces and private model reasoning are excluded.

Pending input is checked before and after a model response and between tools. If input arrives
during generation, the unexecuted proposal is discarded and the model sees the new request.
If input arrives during a batch of calls, completed calls retain their receipts and remaining
calls are skipped. Every advertised call receives a reply before new user input is appended,
preserving the provider's tool-message protocol.

Input arriving in the narrow interval between the final model check and job completion is
picked up by the job supervisor and queued as a follow-up turn, without duplicating user chat.
An explicit pause prevents this automatic inbox drain until the scientist resumes.

Pause is cooperative. It never terminates a compute process or disconnects its pipes. The
current provider request or foreground operation may finish first. Subsequent calls do not
execute. Managed jobs continue independently. The scientist resumes with a new message.

## Client protocol

`GET /api/projects/{id}` returns a complete snapshot. `GET .../events` first returns the same
shape as `event: snapshot`, then `event: delta` with changed messages in `upsert`, the complete
ordered ID list in `message_order`, and current metadata/live/job/session fields. Every event
has a version ID. A reconnect always receives a full snapshot; clients do not need an unbounded
event log or rely on a particular web-server process surviving.

Unchanged message DOM remains attached so figures, selections, and expanded tables survive live
updates. Public response text is streamed; provider `reasoning_content` is never rendered.
Provider endpoints lacking SSE can use `BRILLIANCE_NATIVE_AGENT_STREAM=false`.

Source inspection is confined to `workspace/programs/*.py`. Draft creation is exclusive and
does not overwrite, run, or accept code. Existing project files and lattice evidence are unchanged.

## Validation and limits

`tests/test_research_session.py` exercises context bounds, steering, pause, streamed call
assembly, incomplete streams, deltas, source isolation, and offline connection guards.
`tests/workbench_ui.cjs` runs a real Chromium client against entirely intercepted local fixtures:
no provider, web search, or internet request is made. It requires an already installed Playwright
package/browser (`BRILLIANCE_PLAYWRIGHT_PACKAGE` can identify the package).

The UI/session rebuild does not establish the quality of an autonomous lattice design. Physics
capability still requires JuTrack validation and expert methods. Only MBA currently has developed
local design knowledge; other topics use the general tools and need additional expert content.
