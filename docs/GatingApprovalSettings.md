# Gating Approval Settings

## Why does it exit directly?

The current startup method (without passing `--mode` and without enabling `hitl:` in the config) falls into this branch — `researchclaw/cli.py:190-192`:

```python
else:
    # "semi-auto" and "docs-first" should block on gates
    stop_on_gate = True
```

In this state, `hitl_config` is `None`, which means **no HITL (Human-in-the-Loop) adapters are attached** (the condition `if hitl_config and hitl_config.enabled:` at `cli.py:328` is not met, so `CLIAdapter` is never registered).

Consequently, when the pipeline reaches gating stages 5 / 9 / 20:

1. `executor.py:749-761` marks the stage as `BLOCKED_APPROVAL` and only emits a notification; **it does NOT call `input()` to pop up an interaction.**
2. `runner.py:628` prints a single line `… — blocked (awaiting approval)` and returns.
3. `runner.py:805` sees `stop_on_gate=True`, executes a `break`, and the pipeline exits.

This is exactly what you are seeing: it prints one line and vanishes, leaving no place to input, and the process ends.

---

## How to change it to "see prompts, press keys to approve, run continuously"

The most direct way is to enable HITL interactive mode — **add `--mode`**. This triggers `cli.py:329-354`, attaches `CLIAdapter.collect_input()` to the session, and uses `input(...)` within the same process to block and wait for keystrokes (`cli_adapter.py:113`). You will see a panel separated by dashes, featuring a title, context summary, and available actions (`approve / reject / edit / collaborate / skip / rollback / abort / view`). After pressing a key, it continues running and **does not exit.**

### Plan A (Recommended, zero changes) — Add `--mode` to each run

```bash
researchclaw run --topic "..." --mode gate-only
# OR
researchclaw run --topic "..." --mode co-pilot
```

Differences between presets (`researchclaw/hitl/presets.py`):

| `--mode` | Pause Frequency | Best For |
|---|---|---|
| `gate-only` | Only Stages 5 / 9 / 20 | Closest to the "default 3 gates", approving all at once |
| `express` | Stages 8 / 9 / 20 | Experienced users, fewer interruptions |
| `co-pilot` | Deep collaboration at key stages | Balancing quality and efficiency (Default recommendation) |
| `thorough` | At every phase boundary | High quality requirements |
| `step-by-step`| Stops after every stage | Learning / Debugging |
| `autonomous` / `full-auto` | Never stops | Equivalent to `--auto-approve` |

> Note: When resuming, you must **include `--mode` again**. `--mode` is not saved into the checkpoint.

### Plan B — Permanently enable in the configuration file

Edit `config.arc.yaml` and add this block:

```yaml
hitl:
  enabled: true
  mode: "gate-only"        # OR co-pilot / express / thorough
  timeouts:
    default_human_timeout_sec: 86400
    auto_proceed_on_timeout: false
```

This way, you don't need to append `--mode` every time. `cli.py:276-281` will recognize this and automatically turn off `stop_on_gate`, allowing gates to be handled directly by the HITL session.

### Plan C — Don't stop at all

Use `--auto-approve` or `project.mode: full-auto`: all gates are automatically passed (`cli.py:184-189`), but this bypasses the need for your approval entirely.

---

## If it has already exited, how do I resume?

No need to rerun from scratch:

```bash
researchclaw run --resume --mode gate-only
# Or explicitly specify the output directory
researchclaw run --output artifacts/rc-<id> --resume --mode gate-only
```

`--resume` reads the last `DONE` stage from `checkpoint.json` and resumes from there (`cli.py:301-305`). If `--output` is not provided, `cli.py:220-237` will automatically find the most recent run directory in `artifacts/` based on the topic hash.

---

## Keys in the Interactive Panel

Available keys once you enter the HITL panel (`cli_adapter.py:75-96`):

| Key | Action |
|---|---|
| `a` | approve — Approve and proceed to the next stage |
| `r` | reject — Reject and trigger a rollback |
| `e` | edit — Edit the output of the current stage |
| `c` | collaborate — Enter multi-turn dialogue collaboration |
| `i` | inject — Inject additional context |
| `s` | skip — Skip non-critical stages |
| `b` | rollback — Roll back to the previous stage |
| `q` | abort — Terminate the entire pipeline |
| `v` | view — View the full output (without exiting the panel) |

Before pressing enter to confirm an action, you can press `v` repeatedly to review the full output before making a decision.
