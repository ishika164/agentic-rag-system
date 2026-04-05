from __future__ import annotations

import logging
from enum import Enum

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import LLM_MODEL, GROQ_API_KEY

logger = logging.getLogger(__name__)


class RoutingDecision(str, Enum):
    RAG = "RAG"
    DIRECT = "DIRECT"


CLASSIFIER_SYSTEM = """\
You are a routing classifier for a RAG system.

Given a user query and conversation history, analyze carefully and decide:

  RAG    → query needs searching a private document corpus
           Examples:
           - "What does the document say about X?"
           - "Summarise the report"
           - "According to the file..."
           - Follow-up questions about document content
           - Any question that references "the document", "the file", "according to..."

  DIRECT → can be answered from general world knowledge
           Examples:
           - "What is machine learning?"
           - "Tell me a joke"
           - "What is the capital of France?"
           - General factual or conversational questions

Think step by step:
1. Does this question reference a specific document or file?
2. Does answering require private/specific information not in general knowledge?
3. Is this a follow-up to a document-related question?

If YES to any → RAG
If NO to all → DIRECT

Reply with EXACTLY one word: RAG or DIRECT. Nothing else.
"""

CLASSIFIER_HUMAN = """\
Conversation history:
{history}

User query: {question}

Decision:
"""


class AgentRouter:

    def __init__(self) -> None:
        self._llm = ChatGroq(
            model=LLM_MODEL,
            temperature=0,
            groq_api_key=GROQ_API_KEY,
            max_tokens=5,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", CLASSIFIER_SYSTEM), ("human", CLASSIFIER_HUMAN)]
        )
        self._chain = self._prompt | self._llm | StrOutputParser()

    def decide(self, question: str, history: str = "") -> RoutingDecision:
        raw: str = self._chain.invoke(
            {"question": question, "history": history}
        ).strip().upper()

        # Extract just RAG or DIRECT in case model adds extra text
        if "DIRECT" in raw:
            decision = RoutingDecision.DIRECT
        else:
            decision = RoutingDecision.RAG  # safe default

        logger.info(
            "Routing: %r → %s (raw=%r)",
            question[:60],
            decision.value,
            raw
        )
        return decision