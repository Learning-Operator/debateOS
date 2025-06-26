# DebateOS

A multi-agent debate system powered by AutoGen that orchestrates structured debates between AI agents.

## Overview

DebateOS creates a structured debate environment with four specialized AI agents:
- **Con Debater**: Argues against the resolution
- **Pro Debater**: Argues in favor of the resolution  
- **Judge**: Evaluates arguments and provides final judgment
- **Admin**: Manages debate flow and turn-taking

## Requirements

```bash
pip install 'ag2[openai]' openai pydantic pyyaml numpy
```


## Usage

```python
from debate_os import DebateOS

# Initialize the debate system
debate = DebateOS(
    api_key="your_openai_api_key",
    model="gpt-4o-mini",
    N_rounds=15
)

# Run a debate
resolution = "Artificial intelligence will have a net positive impact on society"
result, context, last_agent = debate.run_debate(resolution)
```

## Configuration

- `api_key`: Your OpenAI API key
- `model`: OpenAI model to use (default: "gpt-4o-mini")  
- `N_rounds`: Number of debate rounds (default: 15)
- `MAX_CONTEXT`: Maximum message history for debaters (default: 6)
- `ADMIN_CONTEXT`: Maximum message history for judge (default: 3)

## Features

- **Structured Debate Flow**: Automatic turn management between debaters
- **Context Management**: Maintains debate history and argument tracking
- **Configurable Rounds**: Set custom debate length
- **Message History Limiting**: Prevents context overflow
- **YAML Configuration**: Easy agent customization through config files

## Output

The system returns:
- Complete chat history
- Context variables with debate state
- Final agent information

The debate concludes with a judge's evaluation of all presented arguments.
