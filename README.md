# Oracelion

A Python-based project for audio file processing and data management.

## Project Structure

- `data/` - Directory for storing project data
- `audio_file/` - Directory for storing audio files
- `.env` - Environment variables (not tracked in git)

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. Install dependencies (once requirements.txt is available):
```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory with necessary environment variables.

## Development

- Make sure to follow the project's coding standards
- Keep sensitive data out of version control
- Update requirements.txt when adding new dependencies

## Git Ignore Rules

- Python virtual environments (.venv/, env/)
- Environment variables file (.env)
- Data and audio files (data/*, audio_file/*)
- Python cache files (__pycache__/, *.pyc)
- System files (.DS_Store, Thumbs.db)