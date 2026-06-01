# claude-memory (portable copies)

Verbatim copies of Claude's file-based *memory* for this project, which normally
lives **outside** the repo at `~/.claude/projects/<slug>/memory/` and therefore
does **not** travel with a `git clone` / USB copy.

These are kept here so the context survives a move to another machine. Two ways to
use them on a fresh machine:

1. **Simplest:** just tell the new Claude session to read [`../../HANDOFF.md`](../../HANDOFF.md)
   (which summarizes all of this and the next steps). No re-seeding needed.
2. **Re-seed the real memory (optional):** copy `copri-project-overview.md` and
   `bucket-sieve-next-step.md` into that machine's
   `~/.claude/projects/<slug-for-this-repo-path>/memory/` and add matching lines to
   its `MEMORY.md`. The `<slug>` is the repo's absolute path with `/` and spaces
   replaced by `-`. (The `originSessionId` in the frontmatter is cosmetic.)

Authoritative project state and the Windows/CUDA plan are in
[`../../HANDOFF.md`](../../HANDOFF.md).
