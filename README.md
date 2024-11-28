

# AI Video Generator

An automated video generation system that transforms news articles into engaging short-form videos with AI-powered translations, narration, and visual content.

### Features

* Article Scraping: Automatically extracts content from news articles
* Multi-language Support: Translates content between languages using Bhashini API
* AI-Powered Script Generation: Creates optimized video scripts using Ollama/Gemma2
* Text-to-Speech: Generates natural-sounding narration using Edge TTS
* Dynamic Video Generation: Creates visually appealing videos with relevant background footage from Pexels
* Automated Captioning: Adds timed captions using Whisper for better accessibility
* Keyword Extraction: Uses KeyBERT and YAKE for intelligent video content matching

### Prerequisites

* Python 3.8+
* FFmpeg installed on your system
* Pexels API key
* Bhashini API configuration

### Installation

1. Clone the repository
2. Install required packages:
```bash
pip install requests beautifulsoup4 moviepy keybert yake langchain edge-tts whisper-timestamped
```

3. Set up configuration:
- Add your Pexels API key to the `PEXELS_API_KEY` constant
- Configure the Bhashini service path in `CONFIG_FILE_PATH`

### Usage

```python
# Run the main script
python video_generator.py

# Basic usage with custom URL
url = "your-news-article-url"
source_lang = "kn"  # Source language code
target_lang = "en"  # Target language code

asyncio.run(main())
```

### Configuration

The system uses several configuration files and environment variables:

```python
PEXELS_API_KEY = "your-api-key"
CONFIG_FILE_PATH = "/path/to/service.json"
```

### Project Structure

```
.
├── video_generator.py    # Main script
├── service.json         # Bhashini API configuration
└── .logs/              # Log directory
    ├── gpt_logs/       # GPT processing logs
    └── pexel_logs/     # Pexels API logs
```

### How It Works

1. Article Scraping: Extracts content from news articles using BeautifulSoup
2. Translation: Converts content to target language using Bhashini API
3. Script Generation: Creates optimized video script using AI
4. Audio Generation: Converts script to speech using Edge TTS
5. Video Creation: 
   - Extracts keywords from content
   - Searches and downloads relevant video clips from Pexels
   - Combines audio, video, and captions into final output




