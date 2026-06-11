from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
import re
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
# -----------------------
# STATE
# -----------------------
class State(TypedDict):
    question: str
    route: str
    tool_result: str
    final_answer: str


# -----------------------
# LLM (Gemini)
# -----------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# -----------------------
# ROUTER NODE
# -----------------------
def router_node(state: State):
    q = state["question"].lower()

    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", q):
        route = "calculator"

    elif "weather" in q:
        route = "api"

    else:
        route = "llm"

    return {"route": route}


# -----------------------
# TOOL 1: CALCULATOR
# -----------------------
def calculator_tool(state: State):
    try:
        match = re.search(r"\d+(?:\s*[\+\-\*/]\s*\d+)+", state["question"])
        if not match:
            return {"tool_result": "Invalid expression"}
        expression = match.group()
        result = eval(expression)   # demo only (unsafe in prod)
        return {"tool_result": f"🧮 Calculator Result: {result}"}
    except Exception as e:
        return {"tool_result": f"Invalid expression ({e})"}
# -----------------------
# TOOL 2: API TOOL (mock real-world API)
# -----------------------
def api_tool(state: State):
    # Example: pretend weather API response
    q = state["question"]

    return {
        "tool_result": f"🌦️ API Result: Weather in Bhopal is 30°C, Sunny (mocked)"
    }


# -----------------------
# TOOL 3: LLM NODE
# -----------------------
def llm_tool(state: State):
    response = llm.invoke(state["question"])
    return {"tool_result": response.content}


# -----------------------
# FINAL NODE (AGGREGATOR)
# -----------------------
def final_node(state: State):
    prompt = f"""
You are an AI system.

Question:
{state['question']}

Tool Output:
{state['tool_result']}

Generate a clean final answer using the tool output.
"""

    response = llm.invoke(prompt)

    return {"final_answer": response.content}


# -----------------------
# ROUTING FUNCTION
# -----------------------
def route_selector(state: State) -> str:
    return state["route"]


# -----------------------
# BUILD GRAPH
# -----------------------
graph = StateGraph(State)

graph.add_node("router", router_node)
graph.add_node("calculator", calculator_tool)
graph.add_node("api", api_tool)
graph.add_node("llm", llm_tool)
graph.add_node("final", final_node)

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    route_selector,
    {
        "calculator": "calculator",
        "api": "api",
        "llm": "llm"
    }
)

# all tools go to final
graph.add_edge("calculator", "final")
graph.add_edge("api", "final")
graph.add_edge("llm", "final")

graph.add_edge("final", END)


app = graph.compile()




if __name__ == "__main__":
    print("\n🚀 REAL Tool-Based LangGraph System\n")

    while True:
        q = input("Ask: ")

        if q.lower() == "exit":
            break

        result = app.invoke({
            "question": q,
            "route": "",
            "tool_result": "",
            "final_answer": ""
        })

        print("\n👉 FINAL ANSWER:")
        print(result["final_answer"])
        print("-" * 50)