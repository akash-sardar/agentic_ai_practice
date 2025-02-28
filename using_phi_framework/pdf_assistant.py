import typer
from typing import Optional, List
import os

import phi
from phi.assistant import Assistant
# storage to run pgVector - creates a session and a temp file to physically store data
from phi.storage.agent.postgres import PgAgentStorage  
from phi.storage.assistant.postgres import PgAssistantStorage
# Access the knowledgebase
from phi.knowledge.pdf import PDFUrlKnowledgeBase
# declare the vector db clas
from phi.vectordb.pgvector import PgVector2



# Load environnt variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get API Key from environment
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
phi.api = os.getenv("PHI_API_KEY")

# Database running on docker (containerized database)
# db connection string
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# PdfURLKnowledge reads the pdfs from URL and stores in VectorDB PgVector2
knowledge_base = PDFUrlKnowledgeBase(
    urls = ["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes", db_url=db_url)
)
# Load the knowledge base: Comment out after first run
knowledge_base.load()

# Create a storage backend using the Postgres database
storage = PgAssistantStorage(
    # store sessions in the ai.sessions table
    table_name="pdf_assistant",
    # db_url: Postgres database URL
    db_url=db_url,
)

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None
    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]
    
    assistant = Assistant(
        run_id = run_id,
        user_id= user,
        knowledge_base=knowledge_base,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # enable the assisstant to search the knowldge base
        search_knowledge=True,
        # Enabe the assisstant to read the chat history
        read_chat_history=True,
    )
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")
    assistant.cli_app(markdown = True)


if __name__ == "__main__":
    typer.run(pdf_assistant)
