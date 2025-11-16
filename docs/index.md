# rbmpy

This is the official documentation for the `rbmpy` Python package.

## Installation

You can install the package using:

```bash
pip install rbmpy
```

## Getting Started

Here's how to use `rbmpy` in your Python project:

```
from rbmpy import AgentVars, AlAgent, compute_persprob

# Set up agent variables
vars = AgentVars()

# Initialize the agent
agent = AlAgent(vars)

# Example: compute perseveration probability
p = compute_persprob([0.1, 0.2, 0.3])
print(p)
```

For full API details, see the [API Reference](reference.md).