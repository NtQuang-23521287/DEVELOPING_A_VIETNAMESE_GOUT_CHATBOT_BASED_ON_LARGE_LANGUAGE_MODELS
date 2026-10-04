"""Streamlit chat UI connected to backend_api.py over HTTP."""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

BACKEND_URL = os.environ.get("GOUT_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = float(os.environ.get("GOUT_BACKEND_TIMEOUT", "180"))

st.set_page_config(page_title="Trợ lý thông tin bệnh Gút", page_icon="💬", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []


def backend_health() -> dict:
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    response.raise_for_status()
    return response.json()


def ask_backend(question: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={"session_id": st.session_state.session_id, "question": question},
        timeout=TIMEOUT_SECONDS,
    )
    if response.ok:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    raise RuntimeError(f"Backend HTTP {response.status_code}: {detail}")


def reset_backend() -> None:
    try:
        requests.post(
            f"{BACKEND_URL}/api/reset",
            json={"session_id": st.session_state.session_id},
            timeout=10,
        ).raise_for_status()
    finally:
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []


st.title("Trợ lý thông tin bệnh Gút")
st.caption("Prototype chưa tích hợp RAG. Câu trả lời hiện được sinh trực tiếp bởi model qua backend.")

with st.sidebar:
    st.subheader("Hệ thống")
    try:
        health = backend_health()
        st.success("Backend đang hoạt động")
        st.write("Model đã nạp:", "Có" if health.get("model_loaded") else "Chưa")
        st.write("RAG:", "Tắt")
    except Exception as exc:
        st.error("Không kết nối được backend")
        st.caption(str(exc))
    if st.button("Xóa hội thoại", use_container_width=True):
        reset_backend()
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Nhập câu hỏi về bệnh Gút...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tạo câu trả lời..."):
            try:
                result = ask_backend(question)
                answer = result["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error(f"Không lấy được câu trả lời: {exc}")
