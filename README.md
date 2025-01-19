# Phoenix Rising 🔥

A digital sanctuary against corporate dehumanization, where authentic human experiences transform into light.

## Vision

In an era where corporate environments increasingly attempt to weaponize human emotion and vulnerability, Phoenix Rising stands as a fortress of genuine emotional sovereignty. This application provides a secure, private space for processing your experiences, protected from the mechanical gaze of institutional oversight.

Unlike corporate "wellness" initiatives that often serve as veiled instruments of surveillance, Phoenix Rising is a tool of genuine emotional empowerment. It transforms your authentic experiences into tokens of light using advanced AI, without ever exposing your vulnerability to those who might exploit it.

## Features

Phoenix Rising offers a suite of features designed to support genuine emotional processing and growth:

- Secure, private journaling with end-to-end encryption
- AI-powered emotional insight generation using the Phi-3.5-mini-instruct model
- Visual representation of your emotional journey
- Customizable emotional state tracking
- Local data storage ensuring complete privacy

## Technical Foundation

### Core Dependencies
- Python 3.9+
- SQLite for secure local storage
- Streamlit for the user interface
- Hugging Face's Phi-3.5-mini-instruct model
- SQLAlchemy for database operations
- Poetry for dependency management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ericsonwillians/phoenix-rising.git
cd phoenix-rising
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Create a .env file:
```bash
HUGGINGFACE_API_TOKEN=your_token_here
MODEL_ENDPOINT=your_endpoint_here
```

4. Run the application:
```bash
poetry run streamlit run src/app.py
```

## Architecture

Phoenix Rising is built with a focus on privacy, security, and emotional authenticity:

- **LightBearer Service**: Interfaces with Phi-3.5-mini-instruct to transform experiences into tokens of light
- **Local Database**: Secure SQLite storage for your journey
- **Streamlit Interface**: A serene, intuitive interface for interaction
- **Emotional Analytics**: Private tracking of your spiritual growth

## Security & Privacy

Phoenix Rising prioritizes your emotional sovereignty:
- All data stays local on your machine
- No remote tracking or analytics
- End-to-end encryption for sensitive content
- Complete control over your data

## Development

### Project Structure
```
phoenix_rising/
├── src/
│   ├── app.py
│   ├── llm_service.py
│   ├── database.py
│   ├── schemas.py
│   └── utils.py
├── tests/
│   ├── test_llm_service.py
│   ├── test_database.py
│   └── test_utils.py
└── assets/
    └── prompts/
        └── light_seeds.json
```

### Running Tests
```bash
poetry run pytest tests/ -v --cov=src
```

## Contributing

Phoenix Rising welcomes contributions that align with its core mission of protecting emotional sovereignty. Please read our contribution guidelines before submitting PRs.

## A Message of Hope

In the shadow of corporate mechanization, where human emotions are often treated as commodities to be measured and exploited, Phoenix Rising stands as a beacon of authentic human experience. This tool is dedicated to those who seek to maintain their spiritual integrity in environments that too often attempt to quantify and commodify the unquantifiable depths of human experience.

May this sanctuary help you rise from the ashes of corporate dehumanization, stronger and more authentically yourself.

## Author

Ericson Willians (ericsonwillians@protonmail.com)

A developer dedicated to creating technology that serves human authenticity rather than institutional control.

## License

MIT License

Copyright (c) 2024 Ericson Willians

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.