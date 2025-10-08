import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import time
import logging
import yt_dlp
import requests
import re
import html

from common.streamlit_imports import st

from typing import Optional, List

# ---------- Optional imports ----------
try:
    from transformers import pipeline
    import torch
except Exception:
    pipeline = None
    torch = None


def run_youtube_blog():
    # ---------- Streamlit UI ----------
    st.subheader("📺 YouTube → Blog Writer (Captions Only)")

    video_url = st.text_input("🎥 Paste YouTube Video URL:")
    hf_model_option = st.text_input("🤖 Hugging Face Model", "google/flan-t5-base")

    # Logs + Output containers
    with st.expander("🪶 Logs", expanded=False):
        log_container = st.empty()

    result_container = st.empty()

    # ---------- Logger Setup ----------
    class StreamlitLogger(logging.Handler):
        def __init__(self):
            super().__init__()
            self.logs = ""

        def emit(self, record):
            self.logs += f"\n[{record.levelname}] {record.getMessage()}"
            log_container.text(self.logs)

    logger = logging.getLogger("yt_blog")
    logger.setLevel(logging.INFO)
    logger.addHandler(StreamlitLogger())

    # ---------- Caption Fetch + Cleaner ----------
    def fetch_captions(video_url: str) -> Optional[str]:
        try:
            ydl_opts = {
                "skip_download": True,
                "writeautomaticsub": True,
                "writeinfojson": True,
                "subtitleslangs": ["en"],
                "quiet": True,
                "outtmpl": "%(id)s",
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                subtitles = info.get("automatic_captions") or info.get("subtitles")

                if not subtitles:
                    logger.warning("No captions found.")
                    return None

                en_captions = subtitles.get("en") or subtitles.get("en-US")
                if not en_captions:
                    logger.warning("No English captions found.")
                    return None

                caption_url = en_captions[0]["url"]
                logger.info(f"🎥 Found captions URL:\n{caption_url}")

                # Handle HLS timedtext playlists (m3u8)
                if "manifest.googlevideo.com" in caption_url and "playlist" in caption_url:
                    logger.info("📜 Detected playlist (.m3u8), fetching actual .vtt URL...")
                    m3u8_res = requests.get(caption_url)
                    if m3u8_res.status_code != 200:
                        logger.error("Failed to fetch playlist.")
                        return None
                    # Extract actual .vtt URL
                    matches = re.findall(r"https?://[^\s]+fmt=vtt[^\s]+", m3u8_res.text)
                    if not matches:
                        logger.error("No .vtt link found inside playlist.")
                        return None
                    vtt_url = matches[0]
                    logger.info(f"➡️ Found .vtt URL:\n{vtt_url}")
                    caption_url = vtt_url

                res = requests.get(caption_url)
                if res.status_code != 200:
                    logger.error(f"Failed to fetch captions: {res.status_code}")
                    return None

                cleaned = clean_captions(res.text)
                logger.info("✅ Transcript fetched and cleaned successfully!")
                return cleaned

        except Exception as e:
            logger.error(f"Failed to fetch captions: {e}")
            return None

    def clean_captions(text: str) -> str:
        """Cleans raw WebVTT captions into readable text."""
        text = re.sub(r"WEBVTT.*", "", text)
        text = re.sub(r"Kind:.*", "", text)
        text = re.sub(r"Language:.*", "", text)
        text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3} --> .*", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)
        text = re.sub(r"\n+", "\n", text)
        return text.strip()

    # ---------- Helper Functions ----------
    def chunk_text_by_chars(text: str, max_chars: int = 1000) -> List[str]:
        paras = [p.strip() for p in text.split('\n') if p.strip()]
        chunks, cur, cur_len = [], [], 0
        for p in paras:
            if cur_len + len(p) + 1 > max_chars:
                chunks.append('\n'.join(cur))
                cur, cur_len = [p], len(p)
            else:
                cur.append(p)
                cur_len += len(p) + 1
        if cur:
            chunks.append('\n'.join(cur))
        logger.info(f"Split text into {len(chunks)} chunks.")
        return chunks

    def init_hf_pipeline(model_name: str = "google/flan-t5-base"):
        if pipeline is None:
            raise RuntimeError("transformers not installed.")
        device = 0 if torch and torch.cuda.is_available() else -1
        logger.info(f"Loading model: {model_name} on device: {device}")
        return pipeline("text2text-generation", model=model_name, device=device)

    def summarize_chunks(chunks: List[str], hf_model: str):
        summarizer = init_hf_pipeline(hf_model)
        summaries = []
        for i, chunk in enumerate(chunks):
            prompt = (
                "Summarize the following transcript section in 3 concise, factual bullet points:\n\n" + chunk
            )
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}...")
            out = summarizer(prompt, max_length=256, do_sample=False)
            summaries.append(out[0]['generated_text'])
        return summaries

    def generate_blog(summaries: List[str], hf_model: str):
        generator = init_hf_pipeline(hf_model)
        combined = "\n\n".join(summaries)
        prompt = (
                    "You are an assistant that ALWAYS responds with valid JSON.\n"
                    "Generate a blog post from the following summaries.\n"
                    "Keys: title, seo_description, tags (list), content (markdown).\n\n"
                    f"{combined}\n\nReturn JSON only, no extra text."
                )

        logger.info("Generating final blog content...")
        out = generator(prompt, max_length=1024, do_sample=False)
        txt = out[0]["generated_text"]
        try:
            start, end = txt.find("{"), txt.rfind("}")
            return json.loads(txt[start:end + 1])
        except Exception:
            logger.warning("Model did not return valid JSON; using fallback.")
            return {
                "title": "Auto-generated YouTube Blog",
                "seo_description": combined[:160],
                "tags": ["youtube", "summary"],
                "content": "# Auto-generated Blog\n\n" + combined
            }

    # ---------- UI Trigger ----------
    if st.button("🚀 Generate Blog") and video_url:
        with st.spinner("Fetching captions and generating blog..."):
            start_time = time.time()
            captions = fetch_captions(video_url)
            if not captions:
                st.error("❌ No captions found! Please choose a video with captions.")
                return

            chunks = chunk_text_by_chars(captions)
            summaries = summarize_chunks(chunks, hf_model_option)
            blog = generate_blog(summaries, hf_model_option)
            elapsed = round(time.time() - start_time, 2)

        st.success(f"✅ Blog generated in {elapsed}s!")

        result_container.markdown(f"### 📝 {blog['title']}")
        result_container.markdown(blog['content'])

        st.download_button(
            label="📥 Download Blog (Markdown)",
            data=f"# {blog['title']}\n\n{blog['content']}",
            file_name="youtube_blog.md",
            mime="text/markdown"
        )