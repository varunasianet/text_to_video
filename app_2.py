import requests
from bs4 import BeautifulSoup
import json
import logging
import asyncio
import http.client
import os
import re
import tempfile
import platform
import subprocess
import time
import zipfile
from moviepy.editor import (AudioFileClip, CompositeVideoClip, CompositeAudioClip,
                            TextClip, VideoFileClip, ColorClip)
from moviepy.config import change_settings
from moviepy.audio.fx.audio_loop import audio_loop
from moviepy.audio.fx.audio_normalize import audio_normalize
from keybert import KeyBERT
import yake
from langchain_community.llms import Ollama
import edge_tts
import whisper_timestamped as whisper
from datetime import datetime
import warnings

# Suppress specific MoviePy/FFmpeg warnings
# warnings.filterwarnings("ignore", category="moviepy.video.io.ffmpeg_reader")

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(level=logging.DEBUG)

# Constants
PEXELS_API_KEY = "rqpEWCiO53h8dh530BpGrTdAF9nrgKo1D4F6GzmiC7C4fqarNKw9jA92" #Replace the PEXELS_API_KEY with your own Pexels API key
CONFIG_FILE_PATH = "/Users/asianet/Elec-lng/service.json"
LOG_TYPE_GPT = "GPT"
LOG_TYPE_PEXEL = "PEXEL"
DIRECTORY_LOG_GPT = ".logs/gpt_logs"
DIRECTORY_LOG_PEXEL = ".logs/pexel_logs"

def load_config_json(file_path): #Loading the bhasini config file
    with open(file_path, 'r') as file:
        return json.load(file)

config_json = load_config_json(CONFIG_FILE_PATH)

def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    title = soup.find('h1').text.strip()
    description_meta = soup.find('meta', attrs={'name': 'description'})
    description = description_meta['content'].strip() if description_meta else "No description available"
    
    content_div = soup.find('div', class_='article-content')
    web_summary_div = soup.find('div', class_='web-summary')
    if web_summary_div:
        web_summary_div.decompose()

    content_blocks = content_div.find_all('p') if content_div else []
    content = [block.text.strip() for block in content_blocks if 'href' not in str(block) and 'Follow us on:' not in block.text]
    full_content = ' '.join(content).replace("Follow us on:", "")
    
    return title, description, full_content

def find_service_id(task_type, source_language, target_language=None):
    for config in config_json["pipelineResponseConfig"]:
        if config["taskType"] == task_type:
            for task_config in config["config"]:
                if task_config["language"]["sourceLanguage"] == source_language:
                    if target_language:
                        if "targetLanguage" in task_config["language"] and task_config["language"]["targetLanguage"] == target_language:
                            return task_config["serviceId"]
                    else:
                        return task_config["serviceId"]
    return None

def translate_text(text, source_language, target_language):
    service_id = find_service_id("translation", source_language, target_language)
    if service_id:
        conn = http.client.HTTPSConnection("dhruva-api.bhashini.gov.in")
        headers = {
            'Accept': '*/*',
            'Authorization': config_json["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"],
            'Content-Type': 'application/json'
        }
        payload = json.dumps({
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": source_language,
                            "targetLanguage": target_language
                        },
                        "serviceId": service_id
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": text
                    }
                ]
            }
        })
        
        try:
            conn.request("POST", "/services/inference/pipeline", payload, headers)
            res = conn.getresponse()
            data = res.read()
            response_json = json.loads(data.decode("utf-8"))
            
            if res.status == 200:
                try:
                    translated_text = response_json["pipelineResponse"][0]["output"][0]["target"]
                    return translated_text
                except (KeyError, IndexError) as e:
                    logging.error(f"Error parsing response: {e}")
                    return f"Translation failed: Error parsing response"
            else:
                logging.error(f"Translation API request failed with status code: {res.status}")
                logging.debug(f"Response content: {data.decode('utf-8')}")
                return f"Translation failed with status code: {res.status}"
        except Exception as e:
            logging.error(f"Translation API request failed: {e}")
            return f"Translation failed: {str(e)}"
    else:
        return "Translation service not available for the selected languages."

def extract_keywords(translated_content):
    try:
        # KeyBERT
        kw_model_bert = KeyBERT()
        keywords_bert = kw_model_bert.extract_keywords(translated_content, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=10)
        
        # LLM (Ollama)
        ollama_llm = Ollama(model="Gemma2")
        prompt = """
        Extract three visually concrete and specific keywords or short phrases from the given content.
        Each keyword should be short (1-3 words) and capture the main visual elements of the content.
        Use only English in your keywords. Ensure each keyword depicts something visual and concrete.
        Separate each keyword with a comma.Dont say here I've extracted the top 5 most relevant and 
        visually concrete keywords for each time segment. Here are the results. I need only results.

        eg;[ 'shoes', 'chair', 'camera', 'bag', 'French', 'alien', 'car', 'tree', ]

        Content:
        {content}

        Keywords:
        """
        
        llm_response = ollama_llm(prompt.format(content=translated_content))
        keywords_llm = [kw.strip() for kw in llm_response.split(',') if kw.strip()]
        
        logging.info(f"Extracted KeyBERT keywords: {keywords_bert}")
        logging.info(f"Extracted LLM keywords: {keywords_llm}")
        
        return keywords_bert, keywords_llm
    except Exception as e:
        logging.error(f"Error in extract_keywords: {e}")
        return [], []

async def generate_audio(text, outputFilename):
    communicate = edge_tts.Communicate(text, "en-AU-WilliamNeural")
    await communicate.save(outputFilename)

def generate_timed_captions(audio_filename, model_size="base"):
    WHISPER_MODEL = whisper.load_model(model_size)
    gen = whisper.transcribe_timestamped(WHISPER_MODEL, audio_filename, verbose=False, fp16=False)
    return getCaptionsWithTime(gen)

def getCaptionsWithTime(whisper_analysis, maxCaptionSize=15, considerPunctuation=False):
    wordLocationToTime = getTimestampMapping(whisper_analysis)
    position = 0
    start_time = 0
    CaptionsPairs = []
    text = whisper_analysis['text']

    if considerPunctuation:
        sentences = re.split(r'(?<=[.!?]) +', text)
        words = [word for sentence in sentences for word in splitWordsBySize(sentence.split(), maxCaptionSize)]
    else:
        words = text.split()
        words = [cleanWord(word) for word in splitWordsBySize(words, maxCaptionSize)]

    for word in words:
        position += len(word) + 1
        end_time = interpolateTimeFromDict(position, wordLocationToTime)
        if end_time and word:
            CaptionsPairs.append(((start_time, end_time), word))
            start_time = end_time

    return CaptionsPairs

def getTimestampMapping(whisper_analysis):
    index = 0
    locationToTimestamp = {}
    for segment in whisper_analysis['segments']:
        for word in segment['words']:
            newIndex = index + len(word['text']) + 1
            locationToTimestamp[(index, newIndex)] = word['end']
            index = newIndex
    return locationToTimestamp

def splitWordsBySize(words, maxCaptionSize):
    halfCaptionSize = maxCaptionSize / 2
    captions = []
    while words:
        caption = words[0]
        words = words[1:]
        while words and len(caption + ' ' + words[0]) <= maxCaptionSize:
            caption += ' ' + words[0]
            words = words[1:]
            if len(caption) >= halfCaptionSize and words:
                break
        captions.append(caption)
    return captions

def cleanWord(word):
    return re.sub(r'[^\w\s\-_"\'\']', '', word)

def interpolateTimeFromDict(word_position, d):
    for key, value in d.items():
        if key[0] <= word_position <= key[1]:
            return value
    return None

def generate_script(translated_content):
    prompt = (
        """You are a seasoned content writer for a YouTube Shorts channel, specializing in creating engaging video scripts based on article content. 
        Your task is to generate a concise script for a 30-second video that summarizes the key points of the provided article.

        Here are some guidelines to follow:
        - Focus on the most important and interesting information from the article.
        - Keep the script concise and engaging, aiming for a duration of approximately 30 seconds when spoken (about 75-90 words).
        - Use simple and clear language that is easy to understand for a wide audience.
        - Organize the script in a logical flow, with a clear beginning, middle, and end
        - Dont add any extra information in the content
        
        You will be provided with the English translated content of the article. Your task is to create the best possible video script based on this content.

        Strictly output the script in a JSON format like below, and only provide a parsable JSON object with the key 'script'.

        # Output
        {"script": "Here is the script ..."}
        """
    )

    ollama_llm = Ollama(model="Gemma2")
    response = ollama_llm(prompt + "\n\nTranslated Content: " + translated_content)
    
    try:
        script = json.loads(response)["script"]
    except Exception as e:
        json_start_index = response.find('{')
        json_end_index = response.rfind('}')
        content = response[json_start_index:json_end_index+1]
        script = json.loads(content)["script"]
    
    return script

def search_videos(query_string, orientation_landscape=True):
    url = "https://api.pexels.com/videos/search"
    headers = {
        "Authorization": PEXELS_API_KEY
    }
    params = {
        "query": query_string,
        "orientation": "landscape" if orientation_landscape else "portrait",
        "per_page": 15
    }

    response = requests.get(url, headers=headers, params=params)
    json_data = response.json()
    log_response(LOG_TYPE_PEXEL, query_string, response.json())
   
    return json_data

def getBestVideo(query_string, orientation_landscape=True, used_vids=[]):
    vids = search_videos(query_string, orientation_landscape)
    videos = vids.get('videos', [])

    if orientation_landscape:
        filtered_videos = [video for video in videos if video['width'] >= 1920 and video['height'] >= 1080 and video['width']/video['height'] == 16/9]
    else:
        filtered_videos = [video for video in videos if video['width'] >= 1080 and video['height'] >= 1920 and video['height']/video['width'] == 16/9]

    sorted_videos = sorted(filtered_videos, key=lambda x: abs(15-int(x['duration'])))

    for video in sorted_videos:
        for video_file in video['video_files']:
            if orientation_landscape:
                if video_file['width'] == 1920 and video_file['height'] == 1080:
                    if not (video_file['link'].split('.hd')[0] in used_vids):
                        return video_file['link']
            else:
                if video_file['width'] == 1080 and video_file['height'] == 1920:
                    if not (video_file['link'].split('.hd')[0] in used_vids):
                        return video_file['link']
    print("NO LINKS found for this round of search with query:", query_string)
    return None

def generate_video_url(timed_video_searches, video_server):
    timed_video_urls = []
    used_links = []
    if (video_server == "pexel"):
        for (t1, t2), search_terms in timed_video_searches:
            urls = []
            for query in search_terms:
                if len(query.split()) == 1:
                    query += " scene"
                url = getBestVideo(query, orientation_landscape=False, used_vids=used_links)
                if url:
                    used_links.append(url.split('.hd')[0])
                    urls.append(url)
                    if len(urls) >= 10:
                        break
            if not urls:
                urls.append(None)  # Placeholder if no videos found
            timed_video_urls.append([[t1, t2], urls])
    else:
        logging.error(f"Unsupported video server: {video_server}")
    return timed_video_urls
def getVideoSearchQueriesTimed(script, captions_timed):
    try:
        keywords = call_Ollama(script, captions_timed)
        
        if not keywords:
            print("No keywords extracted. Using default keywords.")
            return [[[0, len(script)], ["nature", "city", "technology"]]]
        
        return [[[start, end], kw_list] for start, end, kw_list in keywords]
    except Exception as e:
        print(f"Error in getVideoSearchQueriesTimed: {e}")
        return [[[0, len(script)], ["nature", "city", "technology"]]]

# Example usage
script = "This is a sample script about technology and nature in the city."
captions_timed = [
    ((0, 2), "This is a sample"),
    ((2, 4), "script about technology"),
    ((4, 6), "and nature in the city.")
]

result = getVideoSearchQueriesTimed(script, captions_timed)
print(json.dumps(result, indent=2))

import re
import logging
from langchain_community.llms import Ollama
def call_Ollama(script, captions_timed):
    ollama_llm = Ollama(model="Gemma2")
    
    captions_str = "\n".join([f"{start}-{end}: {caption}" for (start, end), caption in captions_timed])
    
    prompt = f"""Given the following video script and timed captions, extract three visually concrete and specific keywords for each time segment that can be used to search for background videos. The keywords should be short and capture the main essence of the sentence. They can be synonyms or related terms.

Important Guidelines:
- Use only English in your text queries.
- Each keyword must depict something visual and concrete (e.g., 'rainy street', 'cat sleeping').
- Avoid abstract concepts (e.g., 'emotional moment' is BAD, 'crying child' is GOOD).
- Do not include person names as the Pexels API cannot find them.
- Provide exactly 3 keywords per time segment.
- Ensure the output is valid JSON.

Script: {script}

Timed Captions:
{captions_str}

Return the keywords in the following JSON format:
[
  [start_time, end_time, ["keyword1", "keyword2", "keyword3"]],
  [start_time, end_time, ["keyword1", "keyword2", "keyword3"]],
  ...
]
"""

    response = ollama_llm(prompt)
    
    try:
        # Try to parse the entire response as JSON
        keywords_json = json.loads(response)
        return keywords_json
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from the response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                keywords_json = json.loads(json_match.group())
                return keywords_json
            except json.JSONDecodeError:
                logging.error("Error: Unable to parse JSON from the extracted part of the response")
        else:
            logging.error("Error: No JSON found in the response")
    
    # If all parsing attempts fail, return a default value
    return [[[0, len(script)], ["nature", "city", "technology"]]]


def fix_json(json_str):
    json_str = re.sub(r'[^\x00-\x7F]+', '', json_str)
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    json_str = ''.join(ch for ch in json_str if ord(ch) >= 32)
    return json_str

def log_response(log_type, query, response):
    log_entry = {
        "query": query,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }
    if log_type == LOG_TYPE_GPT:
        if not os.path.exists(DIRECTORY_LOG_GPT):
            os.makedirs(DIRECTORY_LOG_GPT)
        filename = '{}_gpt3.txt'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        filepath = os.path.join(DIRECTORY_LOG_GPT, filename)
        with open(filepath, "w") as outfile:
            outfile.write(json.dumps(log_entry) + '\n')

    if log_type == LOG_TYPE_PEXEL:
        if not os.path.exists(DIRECTORY_LOG_PEXEL):
            os.makedirs(DIRECTORY_LOG_PEXEL)
        filename = '{}_pexel.txt'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        filepath = os.path.join(DIRECTORY_LOG_PEXEL, filename)
        with open(filepath, "w") as outfile:
            outfile.write(json.dumps(log_entry) + '\n')

def download_file(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url)
        f.write(response.content)

def search_program(program_name):
    try: 
        search_cmd = "where" if platform.system() == "Windows" else "which"
        return subprocess.check_output([search_cmd, program_name]).decode().strip()
    except subprocess.CalledProcessError:
        return None

def get_program_path(program_name):
    program_path = search_program(program_name)
    return program_path

import os
import tempfile
from moviepy.editor import (AudioFileClip, CompositeVideoClip, CompositeAudioClip,
                            TextClip, VideoFileClip)
from moviepy.config import change_settings

from moviepy.editor import (AudioFileClip, CompositeVideoClip, TextClip, VideoFileClip, concatenate_videoclips, ColorClip)

def get_output_media(audio_file_path, timed_captions, background_video_data, video_server):
    OUTPUT_FILE_NAME = "rendered_video_5thjuly.mp4"
    
    try:
        audio_clip = AudioFileClip(audio_file_path)
        audio_duration = audio_clip.duration

        # Prepare video clips
        visual_clips = []
        for (t1, t2), video_urls in background_video_data:
            if video_urls and video_urls[0]:  # Check if there's at least one valid URL
                video_clip = VideoFileClip(video_urls[0])
                video_duration = video_clip.duration
                segment_duration = t2 - t1
                
                if video_duration > segment_duration:
                    video_clip = video_clip.subclip(0, segment_duration)
                elif video_duration < segment_duration:
                    loops = int(segment_duration / video_duration) + 1
                    video_clip = concatenate_videoclips([video_clip] * loops).subclip(0, segment_duration)
                
                video_clip = video_clip.set_start(t1)
                visual_clips.append(video_clip)
            else:
                # Create a blank clip if no video is available
                blank_clip = ColorClip((1080, 1920), color=(0, 0, 0), duration=t2-t1)
                blank_clip = blank_clip.set_start(t1)
                visual_clips.append(blank_clip)

        # Add captions
        for (t1, t2), caption in timed_captions:
            text_clip = TextClip(caption, fontsize=72, color='white', font='Arial', method='caption')
            text_clip = text_clip.set_duration(t2 - t1).set_start(t1).set_position('center')
            visual_clips.append(text_clip)

        # Combine all clips
        final_video = CompositeVideoClip(visual_clips)
        final_video = final_video.set_duration(audio_duration).set_audio(audio_clip)

        # Write the final video file
        final_video.write_videofile(OUTPUT_FILE_NAME, codec='libx264', audio_codec='aac', fps=24, preset='fast')

        return OUTPUT_FILE_NAME
    except Exception as e:
        logging.error(f"Error in get_output_media: {e}")
        return None

def merge_empty_intervals(segments):
    merged = []
    i = 0
    while i < len(segments):
        interval, url = segments[i]
        if url is None:
            j = i + 1
            while j < len(segments) and segments[j][1] is None:
                j += 1
            
            if i > 0:
                prev_interval, prev_url = merged[-1]
                if prev_url is not None and prev_interval[1] == interval[0]:
                    merged[-1] = [[prev_interval[0], segments[j-1][0][1]], prev_url]
                else:
                    merged.append([interval, prev_url])
            else:
                merged.append([interval, None])
            
            i = j
        else:
            merged.append([interval, url])
            i += 1
    
    return merged
async def main():
    url = "https://kannada.asianetnews.com/cricket-sports/rishabh-pant-gets-emotional-after-virat-kohli-handover-t20-world-cup-trophy-during-victory-parade-ckm-sg4une"    
    source_lang = "kn"
    target_lang = "en"
    
    try:
        # Step 1: Scrape the article
        logging.info("Scraping the article...")
        title, description, full_content = scrape_article(url)
        logging.info(f"Title: {title}")
        logging.info(f"Description: {description}")
        logging.info(f"Content length: {len(full_content)} characters")
        
        # Step 2: Translate the content
        logging.info("Translating the content...")
        translated_content = translate_text(full_content, source_lang, target_lang)
        logging.info(f"Translated content length: {len(translated_content)} characters")
        
        # Step 3: Extract keywords from translated content
        logging.info("Extracting keywords...")
        keywords_llm = extract_keywords(translated_content)
        # logging.info(f"KeyBERT keywords: {keywords_bert}")
        logging.info(f"LLM keywords: {keywords_llm}")
        
        # Step 4: Generate script
        logging.info("Generating script...")
        script = generate_script(translated_content)
        logging.info(f"Generated script: {script}")
        
        # Step 5: Generate audio
        logging.info("Generating audio...")
        SAMPLE_FILE_NAME = "audio_tts.wav"
        await generate_audio(script, SAMPLE_FILE_NAME)
        logging.info(f"Audio generated: {SAMPLE_FILE_NAME}")
        
        # Step 6: Generate timed captions
        logging.info("Generating timed captions...")
        timed_captions = generate_timed_captions(SAMPLE_FILE_NAME)
        logging.info(f"Timed captions: {timed_captions}")
        
        # Step 7: Generate video search queries
        logging.info("Generating video search queries...")
        search_terms = getVideoSearchQueriesTimed(script, timed_captions)
        logging.info(f"Search terms: {search_terms}")
        
        # Step 8: Generate video URLs
        logging.info("Generating video URLs...")
        VIDEO_SERVER = "pexel"
        background_video_urls = generate_video_url(search_terms, VIDEO_SERVER)
        logging.info(f"Background video URLs: {background_video_urls}")
        
        # Step 9: Merge empty intervals (if needed)
        logging.info("Merging empty intervals...")
        merged_video_urls = merge_empty_intervals(background_video_urls)
        logging.info(f"Merged video URLs: {merged_video_urls}")
        
        # Step 10: Generate output media
        logging.info("Generating output media...")
        output_video = get_output_media(SAMPLE_FILE_NAME, timed_captions, merged_video_urls, VIDEO_SERVER)
        
        if output_video:
            logging.info(f"Output video generated successfully: {output_video}")
        else:
            logging.error("Failed to generate output video")
        
        logging.info("Process completed successfully!")
    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())