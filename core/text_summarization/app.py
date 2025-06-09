import sys
import os
import requests
import validators

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.streamlit_imports import st
from common.langchain_imports import (
    PromptTemplate, load_summarize_chain,
    YoutubeLoader, UnstructuredURLLoader,
    TranscriptsDisabled, NoTranscriptFound, Document
)
from utils.llm import llm
from utils.video_id import extract_video_id
from common.config import load_dotenv
from xml.etree.ElementTree import ParseError
from youtube_transcript_api import YouTubeTranscriptApi
import time

import urllib3
from utils.llm import llm
from utils.video_id import extract_video_id

load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
http = urllib3.PoolManager(headers={"User-Agent": "Mozilla/5.0"})

def run_text_summarization():
    st.subheader("\U0001F9DC LangChain: Summarize Text From YT or Website")
    st.info("\U0001F517 Paste a **YouTube video** or **website** URL to get a summarized version.")

    try:
        st.write("\U0001F527 Testing LLM...")
        result = llm.invoke("Say Hello")
        st.success(result.content)
        model_used = result.response_metadata.get('model_name', 'Unknown')
        st.success("Model used: " + model_used)
        st.write("✅ LLM test successful.")
    except Exception as e:
        st.exception(f"❌ LLM Exception: {e}")
        return

    generic_url = st.text_input("URL", label_visibility="collapsed")

    MAX_RETRIES = 15
    RETRY_DELAY = 2

    if st.button("Summarize the content from YT or Website"):
        st.write("🚀 Button clicked.")

        if not generic_url.strip():
            st.error("Please provide a URL.")
            return

        if not validators.url(generic_url):
            st.error("Please enter a valid URL.")
            return

        st.write(f"\U0001F517 URL received: {generic_url}")

        success = False
        attempt_msg = st.empty()
        parse_error_shown = False

        youtube_msg_shown = False
        analysis_msg_shown = False
        building_chain_msg_shown = False
        running_chain_msg_shown = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                attempt_msg.markdown(f"🔁 **Attempt {attempt} of {MAX_RETRIES}**")
                with st.spinner("\U0001F50D Loading and summarizing content..."):
                    docs = []

                    if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        if not youtube_msg_shown:
                            st.write("\U0001F3A5 Detected YouTube URL.")
                            youtube_msg_shown = True

                        video_id = extract_video_id(generic_url)
                        try:
                            result = YouTubeTranscriptApi.get_transcript(video_id)
                            if not analysis_msg_shown:
                                st.write("\U0001F4E5 Analysing YouTube Video...")
                                analysis_msg_shown = True

                            docs = [Document(page_content="\n".join([line["text"] for line in result]))]

                        except (TranscriptsDisabled, NoTranscriptFound):
                            st.error("❌ This YouTube video doesn't have a transcript available.")
                            break

                        except Exception as e:
                            if not parse_error_shown:
                                st.warning("⚠️ Transcript could not be parsed or loaded. Retrying...")
                                parse_error_shown = True
                            continue

                    else:
                        st.write("\U0001F310 Detected regular website URL.")
                        try:
                            st.write("\U0001F4E5 Sending request to URL...")
                            response = http.request('GET', generic_url)

                            st.write(f"\U0001F4E1 Response status code: {response.status}")
                            if response.status != 200 or len(response.data.strip()) == 0:
                                st.error("❌ Failed to fetch website content.")
                                break

                            loader = UnstructuredURLLoader(
                                urls=[generic_url],
                                ssl_verify=True,
                                headers={"User-Agent": "Mozilla/5.0"}
                            )
                            st.write("\U0001F4C4 Extracting content with UnstructuredURLLoader...")
                            docs = loader.load()
                            st.write(f"✅ Content extracted. Docs count: {len(docs)}")

                            if not docs:
                                st.error("❌ No usable content extracted from the webpage.")
                                break
                        except Exception as e:
                            st.error("❌ Error loading webpage.")
                            st.exception(e)
                            break

                    prompt_template = """
                    Provide a summary of the following content in 300 words:
                    Content: {text}
                    """
                    prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

                    if not building_chain_msg_shown:
                        st.write("\U0001F9E0 Building summarization chain...")
                        building_chain_msg_shown = True

                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

                    if not running_chain_msg_shown:
                        st.write("\U0001F4E6 Running summarization chain...")
                        running_chain_msg_shown = True

                    output_summary = chain.invoke(docs)

                    st.success("✅ Summary:")
                    st.write(output_summary['output_text'])

                    success = True
                    break

            except Exception as e:
                st.warning(f"⚠️ Attempt {attempt} failed with error: {type(e).__name__}: {e}")
                time.sleep(RETRY_DELAY)

        if not success:
            st.error("❌ All attempts to summarize the content failed. Please try again later.")