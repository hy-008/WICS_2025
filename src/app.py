# app.py
import streamlit as st
from backend import answer_with_rag

st.set_page_config(page_title="MATH 200 Tutor – Prof. Quack", page_icon="🦆")

with st.sidebar:
    st.markdown("### 🦆 Prof. Quack")
    st.write("Your MATH 200 tutor powered by the CLP-3 textbook and Gemini 1.5.")

st.title("📘 MATH 200 – Prof. Quack Tutor")
st.write("Ask questions about gradients, double integrals, Jacobians, etc. based on the CLP-3 textbook.")

question = st.text_input("Ask Prof. Quack a question:")

if st.button("Ask 🦆") and question.strip():
    with st.spinner("Prof. Quack is thinking..."):
        answer = answer_with_rag(question)

    st.markdown("### 🦆 Prof. Quack says:")
    st.write(answer)