# babble-on

## Experiment log (required)

Every experiment gets an entry appended to `docs/EXPERIMENTS.md`, in the same
commit as its code, results, or revert. "Experiment" means any spike, audit,
sweep, smoke test, model trial, or a UI change built to be judged by eye.

- One file, strictly sequential, newest entry at the bottom. Never create a
  second log or per-experiment result files in place of an entry.
- Each entry: date, title, **Question**, **Done** (what ran, where the code
  and raw outputs live, commit hash), **Result** (numbers and observations),
  **Decision** (shipped, rejected, open, and why).
- Rejected experiments stay in the log with the reason, so they are not
  re-proposed.
- Longer write-ups (audits, full result dumps) may live in their own file
  under `docs/` or the spike directory, but the log entry must summarise them
  and link to them.
