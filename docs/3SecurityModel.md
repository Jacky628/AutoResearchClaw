# Critical Analysis of the AutoResearchClaw Security Model

> **Why this document**: AutoResearchClaw faces a fundamental risk—it **executes LLM-generated, unvetted Python code** and **fetches web resources based on URLs provided by LLMs or external data.** Both are textbook attack surfaces. This document does not just list "features"; it performs a **Red Team style critical assessment**: deconstructing the **implementation mechanisms** of each defense layer and questioning **what they stop, what they miss, why, where the trust boundaries lie, and the extent of residual risk.** It also incorporates the recent SSRF fix (commit `bc9dd50`). All assertions regarding "bypassable" or "incomplete" defenses are **verified by source code review or live testing**; reproduction commands are provided for key findings.
>
> This document complements `ArchitectureResearch.md` (static structure) and `EndToEndTracing.md` (data flow) by focusing exclusively on the **security dimension.**

---

## 0. Threat Model: Scope and Boundaries

Before evaluating defenses, we must define the threats. AutoResearchClaw has three primary attack surfaces, ordered by risk:

**Attack Surface 1: Executing LLM-generated experimental code (Highest Risk).** Stage 10 generates thousands of lines of Python, and Stage 12 runs them on the host or in a container. The core realization is that **the code to be executed is inherently untrusted.** It is dangerous not necessarily because of a malicious actor, but due to three common scenarios: ① LLM alignment failure or simple bugs (e.g., `shutil.rmtree(home)` instead of a temp dir); ② Malicious instructions injected into upstream prompts/retrieved content; ③ Experiments that intentionally involve untrusted third-party code/models. The system must assume "this code could do anything."

**Attack Surface 2: Fetching web resources via URLs (SSRF).** Literature search, PDF extraction, and web crawling visit URLs that may originate from LLM queries or external search results. The classic Server-Side Request Forgery (SSRF) risk is that an attacker influencing the URL could cause the **server** to access internal resources it shouldn't—cloud metadata endpoints (`169.254.169.254`), internal databases, or management consoles.

**Attack Surface 3: Exfiltration of Results and Logs.** Papers/code may be pushed to GitHub/HF; logs may contain API keys. This lower-risk area is covered in the Governance section.

---

## 1. First Line of Defense: Static AST Validation (`experiment/validator.py`)

### 1.1 Sequential Checks

Before generated code enters the sandbox, `validate_code()` (`validator.py:372-404`) runs three checks in order, **short-circuiting on syntax errors**:

1. **Syntax Check** `validate_syntax` (`:314-329`): `ast.parse`. Returns immediately on failure.
2. **Security Scan** `validate_security` (`:332-343`): Parses code into an Abstract Syntax Tree (AST) and uses `_SecurityVisitor` to traverse it against a blacklist.
3. **Import Availability** `validate_imports` (`:346-369`): Checks if imports are in the "available set." **Note**: Missing imports yield a **warning, not an error** (`:362-365`), meaning unknown modules do not block execution—only the ten modules in `BANNED_MODULES` are hard-blocked.

### 1.2 The Four Lists

The security scan relies on four hardcoded lists:

- **`DANGEROUS_CALLS`** (`:67-91`, **error**): Fully-qualified call name strings—`os.system`, `os.popen`, `os.exec*` (entire family), `os.remove/unlink/rmdir`, `subprocess.*`, and `shutil.rmtree`. It matches the "literal attribute chain."
- **`DANGEROUS_BUILTINS`** (`:94-101`, **error**): Bare built-ins—`eval`, `exec`, `compile`, `__import__`.
- **`BANNED_MODULES`** (`:104-117`, **error**): Entire modules—`subprocess`, `shutil`, `socket`, `http`, `urllib`, `requests`, `ftplib`, `smtplib`, `ctypes`, `signal`.
- **`SAFE_STDLIB`** (`:120-165`) + **`COMMON_SCIENCE`** (`:167-201`): Whitelists. Two critical details: First, **`os` itself is whitelisted** (`:145`), only specific calls are banned. Second, **`pickle` is whitelisted** (`:147`), despite being a well-known entry point for RCE.

### 1.3 How the AST Visitor "Sees" Code

`_SecurityVisitor` (`:209-269`) is an `ast.NodeVisitor` covering two nodes:

- `visit_Call` (`:217-239`): For every function call, `_resolve_call_name` resolves the callable into a name string and checks it against the blacklists.
- `visit_Import` / `visit_ImportFrom` (`:243-269`): Checks top-level module names against `BANNED_MODULES`.

`_resolve_call_name` (`:272-281`) is purely static: it only recognizes `ast.Name` and `ast.Attribute`. It recursively joins `os.system(...)` into the string `"os.system"`. Any form that is **not a literal attribute chain** (function return values, subscripts, dynamic attributes) returns an empty string and **evades detection.**

> **The essence of this defense**: A static AST scanner based on "literal string matching." Its efficacy depends entirely on whether a dangerous operation appears literally in the AST.

---

## 2. Blind Spots of AST Scanning (Critical Assessment)

Static AST scanning has a **fundamental flaw that cannot be fixed by adding more names**: it sees literal structure, not runtime behavior. Anything "assembled at runtime" is invisible.

**① `getattr` + String concatenation bypasses `DANGEROUS_CALLS`.** Since `_resolve_call_name` only parses literal chains, the callable for `getattr(os, "sys"+"tem")("ls")` is the result of a `getattr(...)` call, not an `ast.Attribute`, so it doesn't match `"os.system"`. **Verified**:

```python
validate_code('import os\nos.system("ls")')            → errors: ['Dangerous call: os.system()']   ✅ Blocked
validate_code('import os\ngetattr(os, chr(115)+"ystem")("ls")') → errors: []                        ⚠️ Bypassed
```

Similarly, `globals()["eval"](...)` or `getattr(__builtins__, "ev"+"al")(...)` bypass `DANGEROUS_BUILTINS`.

**② `importlib` bypasses `BANNED_MODULES`.** `importlib` is not blacklisted, and `importlib.import_module("subprocess")` is a **function call**, not an `import` statement. `visit_Import` is never triggered. **Verified**:

```python
validate_code('import importlib\nm=importlib.import_module("subprocess")') → errors: []   ⚠️ Bypassed
```

**③ `open()` for arbitrary read/write is not listed.** `open` is a built-in that isn't in `DANGEROUS_BUILTINS` (banning it would prevent experiments from reading/writing data). Since `os` only bans deletions, **writing or overwriting files is unrestricted.** **Verified**:

```python
validate_code('open("/tmp/x","w").write("x")') → errors: []   ⚠️ Bypassed
```
Result: Code can read `~/.ssh/id_rsa` or overwrite `~/.ssh/authorized_keys`.

**④ Connectivity via whitelisted libraries.** While `requests` is banned, `torch`, `datasets`, and `transformers` are whitelisted. These libraries perform internal HTTP requests (e.g., `load_dataset`, `from_pretrained`). Module-level blacklists **cannot control the internal behavior of whitelisted libraries.**

**⑤ `pickle` Deserialization.** `pickle.load` is a classic RCE vector via `__reduce__`, yet `pickle` is in `SAFE_STDLIB`.

**⑥ Resource Exhaustion.** AST cannot analyze runtime overhead. `while True: pass` (CPU) or `x = [0]*10**12` (RAM) are valid syntax with no dangerous calls.

### Summary of this Layer

**AST scanning only stops "naive" dangerous operations.** It is a **"Quality Filter"** to help the LLM correct accidental mistakes during generation (paired with the repair loop), but it is **NOT a security boundary against malicious code.** The true boundary must be provided by the OS layer.

---

## 3. Integration: Gate or Feedback?

Is a validation error a **hard veto** or just **feedback for the LLM**? The code shows it's the **latter**: `validate_code` is called inside Stage 10's `max_repair=5` loop. `format_issues_for_llm` (`:412-422`) feeds the formatted issues back to the LLM for correction. If a malicious snippet uses `getattr` from the start, it never errors, the repair loop isn't triggered, and the code proceeds to execution.

---

## 4. Second Line of Defense: Layered Sandbox Isolation (The True Boundary)

`config.experiment.mode` defines the execution environment. The isolation strength varies wildly.

### 4.1 Isolation Strengths

**`simulated`**: No code execution. Absolutely safe, but no experiment is performed.

**`sandbox` (Local Subprocess) — Almost NO OS-level isolation ⚠️.** This is the most dangerous configuration. It runs generated code using `subprocess` **locally with current user permissions.** It has full filesystem and network access. Its only protection is the bypassable AST scanner. **In `sandbox` mode, an evasive LLM snippet can read any file, write anywhere, and connect to the internet.**

**`docker` (Container + Network Policy) — The True Boundary ✅.** This puts code in a container with three `network_policy` options (`docker_sandbox.py:403-417`):
- `none`: Full `--network none`. Most secure.
- `setup_only`/`pip_only`: Allows network for Phases 0/1 (pip, downloads), then cuts it with `iptables` for Phase 2. **Weakness**: `iptables` requires `NET_ADMIN`/root. If the user lacks this, it **gracefully degrades to leaving the network open** (`:408-413` comment).
- `full`: Debugging only.

The container uses `--memory` limits (`:389`), but lacks `--pids-limit` (fork bombs), `--read-only`, or `cap-drop=ALL`. GPU mode uses `--gpus all` (`:225`), which expands the attack surface.

**`ssh_remote`**: Transfers risk to a remote machine. **`colab_drive`**: Good isolation (disposable environment).

### 4.2 Pre-Execution Measures (`sandbox.py:385-451`)

- **Immutable `experiment_harness.py`**: Injected to prevent code from tampering with measurement logic.
- **Symlink Escape Check** `validate_entry_point_resolved` (`:40-50`): Ensures the entry point doesn't resolve to a location outside the staging directory (blocking `../../etc/passwd`).

> **Core Criticism**: Security depends entirely on the mode. Only `docker(network=none)` provides a real boundary. `sandbox` is effectively running untrusted code on the host.

---

## 5. Third Attack Surface: SSRF and `check_url_ssrf` (`web/_ssrf.py`)

All web requests pass through `check_url_ssrf(url)`. The logic is clear:

```python
if "\\" in url: return "...backslash..."                                  # Added in bc9dd50
if "@" in url.split("//",1)[-1].split("/",1)[0]: return "...userinfo..."  # Added in bc9dd50
parsed = urlparse(url)
if parsed.scheme not in ("http","https"): return "Unsupported scheme"
# Literal IPs parsed; domains resolved via getaddrinfo to IP
if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
    return "Blocked internal/private URL"
```

It **successfully blocks**: non-http(s), RFC1918 private nets, loopback (`127.0.0.1`), link-local (`169.254.169.254`), and reserved addresses.

### 5.1 What commit `bc9dd50` fixed

It closed **URL Parsing Ambiguity** bypasses where different components interpret a URL differently:
- **Backslash**: Rejects any URL containing `\` to prevent interpretation discrepancies.
- **Userinfo**: Rejects URLs where the host section contains `@` to prevent `trusted.com@169.254.169.254` style bypasses.

### 5.2 Residual SSRF Risks (Verified by Source Review)

Two classic SSRF bypasses remain open:

**① DNS Rebinding / TOCTOU.** `check_url_ssrf` resolves DNS once to check (`_ssrf.py:39`). However, `crawler.py` (via `urllib.request.urlopen`) **resolves DNS a second time** when making the request. An attacker-controlled DNS server can return a public IP for the check and `127.0.0.1` for the request.
**② HTTP Redirect Follow.** `check_url_ssrf` only checks the **initial URL** (`crawler.py:74`). `urlopen` follows 30x redirects by default without re-validating the target. An attacker can provide a public URL that redirects to `http://169.254.169.254/`.

---

## 6. Trust Boundary Map and Risk Rating

```
┌─────────────────────────────────────────────────────────────────┐
│  Untrusted Input: LLM Code / LLM or External URLs              │
└─────────────────────────────────────────────────────────────────┘
        │ Code                                  │ URL
        ▼                                       ▼
  [L1 Static AST Scan]                       [check_url_ssrf]
   Bypassable via getattr/importlib/open       Fixed parsing ambiguity (✅)
   Target: Quality Filter, not security        DNS Rebinding/Redirects open (Verified)
        │                                       │
        ▼                                       ▼
  [L2 Sandbox Execution] ← The True Boundary   [urlopen Request]
   ├ sandbox = Host process, no isolation ⚠️     Follows redirects + 2nd DNS resolve
   ├ docker(none) = True Boundary ✅            → Bypasses L1 check
   ├ docker(setup_only) = Network cut requires 
   │   root; degrades to open on non-root ⚠️
```

**Residual Risk Rating**:

| Risk | Level | Basis |
|------|------|------|
| Execution of evasive/injected code in `sandbox` mode | **High** | AST scan is bypassable; host process has no isolation. |
| Exfiltration via whitelisted libs (torch/datasets) | **Med-High** | Blacklists don't cover internal library HTTP. |
| `open()` arbitrary R/W / `pickle` deserialization | **Med-High** | Not in any blacklist. |
| SSRF via DNS Rebinding / Redirects | **Med** | Verified crawler.py follows redirects without re-checking. |
| docker `setup_only` network cut fail (non-root) | **Med** | Explicitly documented as graceful degradation. |

---

## 7. Recommendations for Improvement

1. **Isolation by Default (Highest Priority)**: For untrusted/external topics, force `docker` + `network_policy=none`.
2. **SSRF Closure**: (a) **Lock and direct-connect** to the IP resolved during validation to prevent DNS rebinding; (b) Re-validate **every hop of a redirect.**
3. **Container Tightening**: Add `--pids-limit`, `--read-only`, and `--cap-drop=ALL` to Docker starts.
4. **AST Blacklist Additions**: Include `importlib`, `pickle`, and warn on suspicious `getattr`/`open(..., 'w')`.
5. **Hard Resource Limits**: Use cgroups for CPU/RAM limits instead of just wall-clock timeouts.

---

## 8. Verification (Reproducible)

```bash
# L1 can be bypassed (Expected: empty errors list)
python -c "from researchclaw.experiment.validator import validate_code as v; print('getattr:', [i.message for i in v('import os\ngetattr(os, \"sys\"+\"tem\")(\"ls\")').errors]); print('importlib:', [i.message for i in v('import importlib\nimportlib.import_module(\"subprocess\")').errors])"

# SSRF: Three types of blocking
python -c "from researchclaw.web._ssrf import check_url_ssrf as c; print(c('http://169.254.169.254/latest/meta-data/')); print(c('http://trusted.com@127.0.0.1/')); print(c('http://a\\\\b/'))"

# Verify redirect/DNS risk in crawler.py (read code)
sed -n '72,76p;208,212p' researchclaw/web/crawler.py
```

> **Note**: This file is a research document and does not involve any code changes.
