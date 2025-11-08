import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from agents import Agent, Runner, function_tool
import asyncio
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(override=True)

client = OpenAI()

# Paths
EMBED_PATH = "c://code//agenticai//2_openai_agents//intent_embeddings.npy"
META_PATH = "c://code//agenticai//2_openai_agents//intent_metadata.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOG_FILE = "agent_logs.txt"

model = SentenceTransformer(MODEL_NAME)

# -------------------------------------------------
# Load embeddings + metadata
# -------------------------------------------------
if not (os.path.exists(EMBED_PATH) and os.path.exists(META_PATH)):
    raise FileNotFoundError("Missing embeddings or metadata. Run embedding script first.")

print("Loading embeddings and metadata...")
utterance_embeddings = np.load(EMBED_PATH)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

utterances = meta["utterances"]
intents = meta["intents"]
categories = meta["categories"]

# -------------------------------------------------
# Helper: Log issue
# -------------------------------------------------
def log_issue(issue_type: str, query: str):
    """Append a timestamped line to a shared log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] New {issue_type} issue raised: {query}\n")

# -------------------------------------------------
# Intent classifier using embeddings
# -------------------------------------------------
def classify_intent_semantic(user_query: str):
    """Return top intent and category based on cosine similarity."""
    query_emb = model.encode([user_query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, utterance_embeddings)[0]
    best_idx = int(np.argmax(similarities))
    return {
        "intent": intents[best_idx],
        "category": categories[best_idx],
        "matched_utterance": utterances[best_idx],
        "similarity": round(float(similarities[best_idx]), 3)
    }

# -------------------------------------------------
# Specialized agents (tools)
# -------------------------------------------------
@function_tool
async def handle_account(query: str):
    """Handle account management queries."""
    log_issue("account", query)
    return "Account Agent: assisting with account-related queries."

@function_tool
async def handle_order(query: str):
    """Handle order or purchase queries."""
    log_issue("order", query)
    return "Order Agent: handling purchase and order-related issues."

@function_tool
async def handle_delivery(query: str):
    """Handle delivery or shipment queries."""
    log_issue("delivery", query)
    return "Delivery Agent: tracking or resolving delivery issues."

@function_tool
async def handle_feedback(query: str):
    """Handle feedback, reviews, or complaints."""
    log_issue("feedback", query)
    return "Feedback Agent: managing customer feedback and complaints."

@function_tool
async def handle_payment(query: str):
    """Handle payment, refund, or transaction-related queries."""
    log_issue("payment", query)
    return "Payment Agent: processing payment and transaction issues."

# -------------------------------------------------
# Agent definitions
# -------------------------------------------------
account_agent = Agent(
    name="Account Agent",
    instructions="Handle account management and profile issues.",
    tools=[handle_account]
)

order_agent = Agent(
    name="Order Agent",
    instructions="Handle customer order and purchase issues.",
    tools=[handle_order]
)

delivery_agent = Agent(
    name="Delivery Agent",
    instructions="Handle shipping and delivery inquiries.",
    tools=[handle_delivery]
)

feedback_agent = Agent(
    name="Feedback Agent",
    instructions="Handle customer feedback, reviews, and complaints.",
    tools=[handle_feedback]
)

payment_agent = Agent(
    name="Payment Agent",
    instructions="Handle billing, payments, and refunds.",
    tools=[handle_payment]
)

# -------------------------------------------------
# Triage logic (uses embeddings + LLM)
# -------------------------------------------------
@function_tool
async def triage_logic(query: str):
    """Determine correct agent using embeddings + LLM reasoning."""
    result = classify_intent_semantic(query)

    route_prompt = f"""
    A customer asked: "{query}"
    The embedding model classified this as:
    Intent: {result['intent']}
    Category: {result['category']}
    Similarity: {result['similarity']}

    Which team should handle this — Account, Order, Delivery, Feedback, or Payment?
    Reply with exactly one word.
    """

    llm_response = client.responses.create(
        model="gpt-4o-mini",
        input=route_prompt,
        temperature=0
    )

    route = llm_response.output[0].content[0].text.strip().lower()

    if "account" in route:
        return "account"
    elif "order" in route:
        return "order"
    elif "delivery" in route or "ship" in result["category"].lower():
        return "delivery"
    elif "feedback" in route or "review" in result["category"].lower():
        return "feedback"
    elif "payment" in route or "refund" in result["category"].lower():
        return "payment"
    else:
        return "feedback"  # fallback

# -------------------------------------------------
# Triage Agent
# -------------------------------------------------
triage_agent = Agent(
    name="Triage Agent",
    instructions="Determine which specialized agent should handle the query.",
    tools=[triage_logic],
)

# -------------------------------------------------
# Chat orchestration
# -------------------------------------------------
async def chat_with_customer(query: str):
    """Run triage, select correct agent, and get response."""
    triage_result_session = await Runner.run(triage_agent, query)
    triage_result = triage_result_session.final_output.strip().lower()

    print(f"[Router] → Routed to: {triage_result.capitalize()} Agent")

    if triage_result == "account":
        agent = account_agent
    elif triage_result == "order":
        agent = order_agent
    elif triage_result == "delivery":
        agent = delivery_agent
    elif triage_result == "feedback":
        agent = feedback_agent
    elif triage_result == "payment":
        agent = payment_agent
    else:
        agent = feedback_agent

    response_session = await Runner.run(agent, query)
    return response_session.final_output

# -------------------------------------------------
# Chat loop
# -------------------------------------------------
if __name__ == "__main__":

    async def main():
        print("Customer Service Chatbot (type 'exit' to quit)\n")

        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Chatbot: Goodbye!")
                break

            response = await chat_with_customer(user_input)
            print(f"Chatbot: {response}\n")

    asyncio.run(main())
