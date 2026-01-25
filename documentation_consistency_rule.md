# Documentation Consistency Protocol

**Context**: You are working on "Remixatron", a complex system with deep DSP and networking logic. The "Source of Truth" is split between the code (implementation) and several critical markdown files (`DSP.md`, `BROADCAST.md`, `USERS_MANUAL.md`).

**Rule**: Whenever you modify code or are asked to review the project, you MUST enforce the following consistency protocol:

## 1. The Code is Truth
Implementation reality always supersedes documentation.
-   **If Code Changed**: You MUST check if this change invalidates descriptions in `DSP.md` or `BROADCAST.md`.
-   **If Docs Differ**: If you find a discrepancy during review (e.g., "Doc says X, Code does Y"), assume the Code is correct (unless it's obviously a bug) and **update the documentation immediately**.

## 2. "Tri-State" Synchronization
You must maintain consistency across three layers:
1.  **Markdown Docs**: High-level architecture (`DSP.md`, `BROADCAST.md`).
2.  **Code Comments**: Module/Function-level docstrings (`///`).
3.  **Implementation**: The actual Rust/JS code.

**Protocol**:
-   If you disable a feature in code (e.g., commenting out a clustering step), you MUST update the module docstring AND the relevant section in `DSP.md` to reflect this.
-   Do not leave "zombie comments" referring to disabled or deleted functionality.

## 3. Targeted Audit Triggers
When modifying specific areas, you are REQUIRED to audit specific documents:

| Modified Scope | Required Audit Target | Verification Checks |
| :--- | :--- | :--- |
| **Audio / DSP** | `DSP.md` | Check sample rates, buffer sizes, algorithms (FFT, CQT), and thresholds (e.g., Panic Mode %). |
| **Networking / API** | `BROADCAST.md` | Check protocol specs, binary frame formats (byte offsets), and sync strategies. |
| **UI / Features** | `USERS_MANUAL.md` | Update screenshots or workflows if the UI behavior changes. |

## 4. Verification Step
After any consistency update, run relevant checks (e.g., `cargo check` or `npm test`) to ensure comment updates didn't accidentally break syntax (e.g., initialized `/*` without `*/`).
