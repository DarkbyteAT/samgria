# CLAUDE.md

@AGENTS.md

## Project Context

samgria is a standalone composable gradient transform library for PyTorch. It provides a protocol-based pipeline for interventions around the gradient descent step — perturbation before descent (SAM, ASAM), noise injection after descent (LAMP), and arbitrary compositions. Extracted from rltrain's transforms/ package.

## Scope

| In scope | Out of scope |
|----------|-------------|
| GradientTransform protocol | nn.Module building blocks (→ toblox) |
| SAM, ASAM, LAMPRollback | Training loops, agents, environments |
| Gradient utilities (get_grad, set_grad) | RL-specific code |
| Composable transform pipeline | |
