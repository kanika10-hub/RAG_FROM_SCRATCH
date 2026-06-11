from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. STATE (shared memory)
class State(TypedDict):
    text: str
    word_count: int


# 2. NODE 1: Uppercase
def uppercase_node(state: State):
    return {
        "text": state["text"].upper()
    }


# 3. NODE 2: Word Counter
def word_count_node(state: State):
    return {
        "word_count": len(state["text"].split())
    }


# 4. FINAL OUTPUT NODE
def output_node(state: State):
    print("\nFINAL STATE:")
    print(state)
    return state


# 5. BUILD GRAPH
graph = StateGraph(State)

graph.add_node("uppercase", uppercase_node)
graph.add_node("count_words", word_count_node)
graph.add_node("output", output_node)


# 6. EDGES (flow)
graph.add_edge(START, "uppercase")
graph.add_edge("uppercase", "count_words")
graph.add_edge("count_words", "output")
graph.add_edge("output", END)


# 7. COMPILE
app = graph.compile()


# 8. RUN
result = app.invoke({
    "text": "langgraph makes workflows powerful",
    "word_count": 0
})