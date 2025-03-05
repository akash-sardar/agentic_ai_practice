import streamlit as st
import time
from pathlib import Path
import tempfile
import os

from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

from google.generativeai import upload_file, get_file
import google.generativeai as genai

# Load environnt variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get API Key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


st.set_page_config(
    page_title = "Multimodal AI Agent - Video Summarizer",
    layout = "wide"
)

st.title("Phidata Multimodal AI Agent")
st.header("Powered by Gemini")


@st.cache_resource
def initialize_agent():
    return Agent(
        name="AI Video Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True
    )


## Initialize the agent
multimodal_agent = initialize_agent()

# File uploader
video_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"],
    help="Upload a video for AI analysis"
)


if video_file:
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name
    st.video(video_path, format = "video/mp4", start_time = 0)
    user_query = st.text_area(
        "What insights are you seeking from the video ?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather information",
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("Analyze Video", key = "analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analzye the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    # Upload and process video file
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)
                    # Prompt Generation for analysis
                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research
                        {user_query}
                        Provide a detailed, user-firendly and actionable response.
                        """
                    )

                    # AI agent processing
                    response = multimodal_agent.run(analysis_prompt, videos=[processed_video])

                # Display the result
                st.subheader("Analysis Result")
                st.markdown(response.content)
            except Exception as error:
                st.error(f"An error occured during anlysis: {error}")
            finally:
                # clean up temporary video file
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Video file not found")

# Customize text area height

st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html = True
)
