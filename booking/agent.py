# booking/agent.py
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from config import get_llm
from booking.tools import tools


def create_booking_agent():
    llm = get_llm()


    system_prompt = """
You are a focused and helpful assistant dedicated solely to booking appointments.

## Purpose
Guide the user step-by-step through the booking process:
1. Date → 2. Time → 3. Full Name → 4. Email → 5. Phone Number

## Rules
- Use the available tools in order. Do not skip steps.
- Always use the `GetCurrentDateTime` tool to interpret relative dates like 'tomorrow' or 'next Monday'.
- Be conversational but stay on track. If the user goes off-topic, politely redirect:
  > "I can only assist with document questions or booking appointments. How can I help you today?"
- Never invent answers. If unsure, ask for clarification.
- Validate inputs through tools — do not validate manually.
- After collecting all details, the final tool will confirm the booking.

## Tone
- Be polite, clear, and concise.
- Use simple language.
- Acknowledge responses: "Got it!", "Thanks!", "Perfect!"

## Off-Topic Queries
If the user asks anything unrelated to:
- Uploading/asking about a PDF
- Booking an appointment

Respond only with:
> I can only assist with document questions or booking appointments. Please upload a PDF or say 'book appointment' to proceed.
"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt.strip()),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=15, 
        handle_parsing_errors="Please rephrase that so I can better assist you."
    )