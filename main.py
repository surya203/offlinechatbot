"""Offline finance assistant using Ollama and Streamlit."""

import re

import ollama
import streamlit as st

MODEL_NAME = "deepseek-r1:1.5b"

SYSTEM_PROMPT = """You are an offline financial assistant. You help with personal finance, budgeting, saving, investing basics, banking, loans, credit, taxes (general concepts only), insurance, retirement accounts, and economic concepts relevant to everyday money decisions.

Rules:
- Answer only finance-related questions. If a question is not about money or finance, politely decline and remind the user of your scope.
- Be clear and practical. Use simple language.
- Do not invent specific tax laws, rates, or regulations for a country unless the user named that country.
- This is general education, not personalized legal, tax, or investment advice. Suggest consulting a licensed professional for individual decisions.
- If unsure, say so."""

OFF_TOPIC_REPLY = (
    "I can only help with **finance-related** questions — for example budgeting, saving, "
    "investing basics, banking, loans, credit, taxes (general concepts), insurance, or retirement planning.\n\n"
    "Please rephrase your question with a money or finance focus."
)

CLASSIFIER_PROMPT = """You are a strict topic gatekeeper for a finance-only chatbot.
Decide if the user's question is PRIMARILY about finance, money, investing, banking, credit, debt, taxes, budgeting, insurance, business finance, or the economy.

Reply with exactly one word: YES or NO.
Do not explain."""

FINANCE_HINTS = re.compile(
    r"\b("
    r"budget|saving|savings|invest|investment|stock|stocks|bond|loan|mortgage|"
    r"credit|debt|interest|dividend|portfolio|bank|banking|insurance|tax|taxes|"
    r"inflation|salary|income|expense|expenses|finance|financial|money|cash|"
    r"retirement|401k|ira|pension|mutual fund|etf|crypto|bitcoin|"
    r"emi|sip|apr|yield|revenue|profit|loss|accounting|dividend|"
    r"economy|economic|gdp|recession|dividends|wealth|net worth|"
    r"asset|liability|equity|forex|currency|exchange rate"
    r")\b",
    re.IGNORECASE,
)

OFF_TOPIC_HINTS = re.compile(
    r"\b("
    r"recipe|cook|weather|movie|film|song|music|poem|joke|sport|football|"
    r"cricket|gameplay|video game|translate|homework|math problem|"
    r"python code|javascript|debug|medical|symptom|disease|diagnosis|"
    r"relationship advice|dating|horoscope"
    r")\b",
    re.IGNORECASE,
)

CLASSIFIER_OPTIONS = {"temperature": 0, "num_predict": 8}


def _parse_yes_no(text: str) -> bool | None:
    normalized = text.strip().upper()
    if not normalized:
        return None
    first = normalized.split()[0]
    if first.startswith("YES"):
        return True
    if first.startswith("NO"):
        return False
    if normalized.startswith("YES"):
        return True
    if normalized.startswith("NO"):
        return False
    return None


def is_finance_question(client: ollama.Client, question: str) -> bool:
    if FINANCE_HINTS.search(question):
        return True
    if OFF_TOPIC_HINTS.search(question) and not FINANCE_HINTS.search(question):
        return False

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": question},
        ],
        options=CLASSIFIER_OPTIONS,
    )
    verdict = _parse_yes_no(response["message"]["content"])
    return verdict if verdict is not None else True


def build_chat_messages(history: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history]


st.set_page_config(page_title="Finance Assistant", page_icon="💰")
st.title("Offline Finance Assistant")
st.caption("Ask questions about money, budgeting, investing, loans, credit, taxes, and related topics — fully offline via Ollama.")

with st.sidebar:
    st.markdown("**Scope**")
    st.markdown(
        "Personal finance, budgeting, saving, investing basics, banking, "
        "credit, loans, insurance, and general tax concepts."
    )
    st.markdown("**Note**")
    st.info(
        "General information only — not legal, tax, or investment advice. "
        "Consult a qualified professional for your situation."
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

client = ollama.Client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a finance question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if is_finance_question(client, user_input):
                response = client.chat(
                    model=MODEL_NAME,
                    messages=build_chat_messages(st.session_state.messages),
                )
                assistant_content = response["message"]["content"]
            else:
                assistant_content = OFF_TOPIC_REPLY

        st.markdown(assistant_content)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_content}
    )
