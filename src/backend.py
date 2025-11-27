# backend.py
import os
from dotenv import load_dotenv
from llama_cloud_services import LlamaCloudIndex
import google.generativeai as genai

# === Load environment variables from .env (local only) ===
load_dotenv()

# === Keys (NO DEFAULTS – must be provided via env or .env) ===
LLAMACLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not LLAMACLOUD_API_KEY:
    raise ValueError(
        "Missing LLAMACLOUD_API_KEY. "
        "Set it in a .env file or as an environment variable."
    )

if not GEMINI_API_KEY:
    raise ValueError(
        "Missing GEMINI_API_KEY. "
        "Set it in a .env file or as an environment variable."
    )

# === Configure Gemini ===
genai.configure(api_key=GEMINI_API_KEY)

# You can use 1.5-flash (fast, free-tier friendly) or 1.5-pro (stronger, slower)
GEMINI_MODEL_NAME = "gemini-1.5-flash"
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# === Connect to your LlamaCloud index ===
index = LlamaCloudIndex(
    name="math200RAG",
    project_name="Default",
    organization_id="b7f143d8-d0dc-4c2c-b5a5-bbdaa6d72cec",
    api_key=LLAMACLOUD_API_KEY,
)


def retrieve_chunks(question: str, top_k: int = 5) -> list:
    """
    Step A: Retrieve the most relevant chunks from the CLP-3 index
    for a given question.
    """
    retriever = index.as_retriever()
    nodes = retriever.retrieve(question)

    # Optionally limit to top_k if more are returned
    nodes = nodes[:top_k]
    return nodes


def build_context_from_nodes(nodes: list) -> str:
    """
    Turn retrieved nodes into a single context string for Gemini.
    """
    context_pieces = []
    for i, node in enumerate(nodes, start=1):
        text = getattr(node, "text", str(node))
        context_pieces.append(f"[Chunk {i}]\n{text}\n")
    return "\n".join(context_pieces)


def answer_with_rag(question: str) -> str:
    """
    Step B: Use RAG with Gemini 1.5.
    1) Retrieve chunks from math200RAG (CLP-3 textbook).
    2) Build a prompt instructing Gemini to only use that context.
    3) Ask Gemini to answer the question.
    """
    # ---- Step A: Retrieve chunks ----
    nodes = retrieve_chunks(question)
    if not nodes:
        return "Prof. Quack couldn't find any relevant content in the textbook for that question."

    context = build_context_from_nodes(nodes)

    # ---- Step B: Ask Gemini using a RAG-style prompt ----
    prompt = f"""
You are Prof. Quack, a friendly but precise MATH 200 tutor.
You answer questions using ONLY the following textbook context from CLP-3 (Multivariable Calculus).

If the answer is not clearly supported by this context, say:
"I can't find that in the class materials."

IMPORTANT:
- Use the same notation and style implied by the context.
- Be concise but clear.
- If there are formulas, write them plainly in LaTeX-style text.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = gemini_model.generate_content(prompt)
    return response.text
