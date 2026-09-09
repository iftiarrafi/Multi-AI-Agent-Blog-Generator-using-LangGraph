from langgraph.graph import StateGraph , START , END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Literal
from langgraph.types import interrupt , Command

from state import BlogState
from agents import get_llm , research_agent , writer_agent , editor_agent

MAX_REVISION = 10

## Defining the NODES

def researcher_node(state: BlogState):

    llm = get_llm()

    feedback = state.research_feedback
    previous_research = state.research

    research_data = research_agent(
        llm=llm,
        topic=state.topic,
        audience=state.audience,
        previous_research=previous_research,
        feedback=feedback
    )

    state.research = research_data
    state.research_feedback = ""

    return state


def human_review_research_node(state: BlogState):
    """Pause and ask Human to approve the research or send the feedback"""
    decision = interrupt({
        "stage" : "researcher_review",
        "research" : state.research,
        "instructions" : (
            "Reply with 'approve' to continue to writing",
            "Or describe what to change to send it back to the researcher."
        )
    })
    
    if isinstance(decision , dict) :
        action = decision.get("action" , "approve")
        feedback = decision.get("feedback" , "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approve" , "approved" , "ok" , "fine" , "process" , ""] else "revise"
        feedback = "" if action == "approve" else text
    
    state.research_feedback = feedback
    return state


def writer_node(state: BlogState):
    """
    Writer Agent creates the first draft or revises the previous draft.
    """
    print("\n========== WRITER NODE ==========")
    print("Feedback:")
    print(state.draft_feedback)

    print("\nPrevious draft:")
    print(state.draft)

    print("\nDraft length:")
    print(len(state.draft) if state.draft else 0)

    llm = get_llm()

    feedback = state.draft_feedback
    previous_draft = state.draft

    draft_data = writer_agent(
        llm=llm,
        topic=state.topic,
        audience=state.audience,
        research=state.research,
        previous_draft=previous_draft,
        feedback=feedback
    )

    state.draft = draft_data
    state.draft_feedback = ""

    return state


def human_review_draft_node(state: BlogState):
    """Pause and ask Human to approve the draft or send the feedback"""
    decision = interrupt({
        "stage" : "draft_review",
        "draft" : state.draft,
        "instructions" : (
            "Reply with 'approve' to continue to editor",
            "Or describe what to change to send it back to the writer."
        )
    })
    
    if isinstance(decision , dict) :
        action = decision.get("action" , "approve")
        feedback = decision.get("feedback" , "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approve" , "approved" , "ok" , "fine" , "process" , ""] else "revise"
        feedback = "" if action == "approve" else text
    
    state.draft_feedback = feedback
    if feedback:
        state.revision_count += 1
    return state

    
def editor_node(state:BlogState):
    llm = get_llm()
    final = editor_agent(
        llm=llm , topic=state.topic , draft=state.draft
    )
    state.final_blog = final
    return state


### Conditional edges

# Router after researcher feedback

def router_after_research_review(state: BlogState) -> Literal["researcher" , "writer"]:
    if state.research_feedback:
        return "researcher_edge"
    else:
        return "writer_edge"

# Router after draft feedback

def router_after_draft_review(state: BlogState) -> Literal["writer" , "editor"]:
    if state.draft_feedback and state.revision_count < MAX_REVISION:
        return "writer_edge"
    else:
        return "editor_edge"


# Building the Graph

def build_blog_graph():
    builder = StateGraph(BlogState)
    
    ## Adding Nodes
    builder.add_node("researcher" , researcher_node)
    builder.add_node("researcher_review" , human_review_research_node)
    builder.add_node("writer" , writer_node)
    builder.add_node("writer_review" , human_review_draft_node)
    builder.add_node("editor" , editor_node)
    ## Adding Edges
    builder.add_edge(START , "researcher")
    builder.add_edge("researcher" , "researcher_review")
    builder.add_conditional_edges(
        "researcher_review",
        router_after_research_review,
        {
            "researcher_edge" : "researcher",
            "writer_edge" : "writer"
        }
    )
    builder.add_edge("writer" , "writer_review")
    
    builder.add_conditional_edges(
        "writer_review" ,
        router_after_draft_review,
        {
            "writer_edge":"writer",
            "editor_edge":"editor",
        }
    )
    builder.add_edge("editor" , END)
    
    GRAPH = builder.compile(checkpointer=InMemorySaver())
    return GRAPH
