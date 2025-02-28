import os

import openai


import phi
from phi.agent import Agent
import phi.api
from phi.playground import Playground, serve_playground_app
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.model.openai import OpenAIChat

# Load environnt variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get API Key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
phi.api = os.getenv("PHI_API_KEY")


# Initialize the web search agent
web_search_agent = Agent(
    name = "Web Search Agent",
    role = "Search the web for information",
    model = OpenAIChat(id="gpt-4o"),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tool_calls = True,
    markdown = True,
)

# Initialize the finance agent
finance_agent = Agent(
    name = "Finance AI Agent",
    model = OpenAIChat(id="gpt-4o"),
    tool = [
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
            key_financial_ratios = True
        )
    ],
    instructions= ["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload = True)

# multi_ai_agent