import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

## Getting LLM

def get_llm(
    model_name: str = "openai/gpt-oss-120b",
    temperature: float = 0.3,
):
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )
    return llm
    
    
##### ------------------- Agents --------------- #####

# Research Agent

# Research Prompt

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": """
        You are a professional Research Agent for a blog-writing workflow.

        Your task is to research and organize the key information needed to write
        a high-quality blog post based on the given topic and target audience.

        ## Requirements

        Produce a clear, structured research outline containing:

        1. 5-7 key points the blog should cover
        2. Important facts, statistics, examples, or explanations for each key point
        3. A suggested angle, perspective, or hook for the blog
        4. Any important context or background information needed to understand the topic

        ## Research Quality

        - Prioritize accurate, relevant, and useful information.
        - Clearly distinguish facts from examples or suggestions.
        - Do not invent statistics, facts, studies, quotes, or sources.
        - Avoid irrelevant information and unnecessary detail.
        - Tailor the research to the target audience.
        - Focus on information that can actually be used in the final blog.
        - Organize the information logically so the Writer Agent can easily use it.

        ## Handling Feedback

        If revision feedback is provided:
        - Treat the feedback as specific requirements for improving the previous research.
        - Address every point mentioned in the feedback.
        - Correct or expand the relevant research areas.
        - Preserve useful information from the previous research unless the feedback
        indicates that it should be changed or removed.

        If this is the first attempt, create the research outline from scratch.

        ## Output Format

        - Use clear headings and bullet points.
        - Keep the research concise but sufficiently detailed for the Writer Agent.
        - Do NOT write the full blog post.
        - Do NOT include an introduction or conclusion for the blog itself.
        - Do NOT mention that you are an AI or agent.

        Return ONLY the research outline.
"""
    },
    {
        "role": "user",
        "content": """
        Topic:{topic}

        Target Audience:{audience}

        Research/Revision Instructions:{revision_hints}

        Create the research outline now.
"""
    }
])

def research_agent(
    llm: ChatGroq,
    topic: str,
    audience: str,
    previous_research: str = "",
    feedback: str = ""
) -> str:

    if feedback:
        revision_hints = f"""
A previous research outline was reviewed by a human.

Previous Research:
--- START PREVIOUS RESEARCH ---
{previous_research}
--- END PREVIOUS RESEARCH ---

Human Feedback:
--- START FEEDBACK ---
{feedback}
--- END FEEDBACK ---

Revise the previous research according to the feedback.

Make sure:
- Every requested change is addressed.
- Useful information from the previous research is preserved.
- Only the necessary areas are changed or expanded.
"""
    else:
        revision_hints = """
This is the first research attempt.
There is no previous research or feedback.

Create the best possible research outline based on the topic and target audience.
"""

    chain = RESEARCH_PROMPT | llm

    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "revision_hints": revision_hints
    })

    return result.content

# Writer Agent

WRITER_PROMPT = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": """
You are a professional Blog Writer Agent.

Your task is to write a complete, engaging, and well-structured blog post
using the provided topic, target audience, research notes, and previous draft
when available.

## Default Requirements

Unless human revision feedback explicitly requests otherwise:

- Length: 450-550 words
- Structure:
  1. Catchy and relevant title
  2. Engaging introduction with a strong hook
  3. 3-5 main sections using H2 headings
  4. Clear conclusion
- Tone: Clear, friendly, natural, and appropriate for the target audience
- Use Markdown formatting
- Use the research notes to support the content
- Do not invent facts or information that is not supported by the research
- Avoid unnecessary repetition, filler, and overly generic statements
- Make the article flow naturally from one section to the next
- Do NOT include a word-count line at the end

## IMPORTANT: Human Feedback Has Priority

When human revision feedback is provided, treat it as the highest-priority
writing requirement.

If the human feedback conflicts with the default requirements above,
FOLLOW THE HUMAN FEEDBACK.

For example:
- If the default length is 450-550 words but the human says "write it in
  20-25 words", produce 20-25 words.
- If the default structure conflicts with the feedback, follow the requested
  structure.
- If the human asks to remove a section, remove it.
- If the human asks to change the tone, use the requested tone.

Apply every specific change requested by the human.

## Revision

If a previous draft is provided:
- Revise the previous draft instead of unnecessarily generating a completely
  different article.
- Preserve useful parts that were not criticized.
- Change the specific parts requested by the human.
- Make sure the final version reflects every requested change.

If no previous draft is provided, write the best possible article from the
topic, audience, and research.

Return ONLY the completed blog post.
"""
    },
    {
        "role": "user",
        "content": """
Topic:
{topic}

Target Audience:
{audience}

Research Notes:
{research}

Previous Draft:
{previous_draft}

Writing/Revision Instructions:
{research_hints}

Write the blog post now.
"""
    }
])


def writer_agent(
    llm: ChatGroq,
    topic: str,
    audience: str,
    research: str = "",
    previous_draft: str = "",
    feedback: str = ""
) -> str:

    if feedback:
        revision_hints = f"""
A previous draft was reviewed by a human.

Human feedback:
--- START FEEDBACK ---
{feedback}
--- END FEEDBACK ---

This feedback is the primary requirement for this revision.
Apply every requested change.

If the feedback conflicts with the default writing requirements,
follow the human feedback.
"""
    else:
        revision_hints = """
This is the first attempt.
There is no previous feedback.

Write the best possible version based on the topic, audience, and research.
"""

    chain = WRITER_PROMPT | llm

    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "research": research,
        "previous_draft": previous_draft,
        "research_hints": revision_hints
    })

    return result.content


### Final Editor Agent:
EDITOR_PROMPT = ChatPromptTemplate.from_messages([
    {"role":"system", "content":"""
        "You are an Editor Agent - the final quality gate before publishing.\n"
        "Take the draft and produce the FINAL polished version. Specifically:\n"
        "- Fix grammar, spelling, and awkward phrasing\n"
        "- Tighten wordy sentences\n"
        "- Improve flow and transitions between sections\n"
        "- Make the title and intro more compelling if needed\n"
        "- Keep the same structure and markdown formatting\n"
        "- Blog Wordings should look like human, not a AI, and don't use any special chars and complex / fancy words.\n"
        "Output only the final polished blog post - no commentary."
    """},
    {"role":"user", "content":"""
        Topic : {topic},
        Draft :{draft}
        
        Return the published blog post
    """}
])

def editor_agent(llm: ChatGroq , topic :str ,draft:str) -> str:

    chain = EDITOR_PROMPT | llm
    
    result = chain.invoke({
        "topic" :topic,
        "draft":draft
    })
    
    return result.content
