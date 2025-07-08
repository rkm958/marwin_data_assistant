# app_m.py – Streamlit UI for RAG-powered assistant Marwin
"""
This app allows users to ask questions about business metadata using a chat-like interface.
It embeds the query, searches using FAISS index, generates an LLM response, and logs the interaction.
Users can also provide feedback (like/dislike + optional comments) on each answer.
"""

import streamlit as st
from llm_answer_context import get_llm_answer
from chat_memory import get_recent_memory, append_to_memory, update_feedback_to_memory
from load_vector_store import load_index
import uuid

# Load vector index and metadata
df_metadata, index = load_index("vector_index.faiss")

# Streamlit config
st.set_page_config(page_title="Ask Marwin – Data Assistant", layout="wide")
st.title("🧠 Hi..! I am Marwin: Your Personal Data Explorer..")

# Custom CSS to fix input box at bottom and reverse message order
st.markdown(
     """
    <style>
        body, .stApp {background-color: #0e0e11;}
   
        #chat-scroll {height: 78vh; overflow-y: auto; padding: 1rem; margin-top: 0rem;}
        .bubble-user   {width:70%; float:right; background:#616362; padding:10px; border-radius:10px; margin:5px 0;}
        .bubble-bot    {width:100%; float:left; padding:10px; border-radius:10px; margin:5px 0;}
        .feedback-buttons {display: flex; justify-content: flex-start; gap: 0.5rem; margin-top: 0.1rem; margin-bottom: 0.1rem;}
    </style>
    """,
    unsafe_allow_html=True)

# ────────────────────────── Chat window ──────────────────────────
with st.container():
    st.markdown('<div class="message-container">', unsafe_allow_html=True)

    # show past memory (oldest at top, newest at bottom)
    for turn in get_recent_memory(50):
        st.markdown(f"<div class='bubble-user'>{turn['query']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bubble-bot'>{turn['answer']}</div>", unsafe_allow_html=True)

        st.markdown('<div class="feedback-buttons">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([0.05, 0.05, 0.05, 0.2])
        query_key = turn['id']

       
        #Placeholder buttons
        with col1:
            st.button("Raise a Data Issue", key=f"data_issue_{query_key}")
        
        with col2:
            st.button("Track Your Data Issue", key=f"track_data_issue_{query_key}")

        with col3:
            if st.button("👍", key=f"like_{query_key}"):
                update_feedback_to_memory(turn["query"], turn["answer"], "like")
                st.toast("Thanks for your feedback!", icon="👍")

        with col4:
            dislike_clicked_key = f"dislike_clicked_{query_key}"
            comment_key = f"comment_input_{query_key}"
            comment_submitted_key = f"comment_submitted_{query_key}"

            # 👎 Dislike button toggles comment box visibility
            if st.button("👎", key=f"dislike_{query_key}"):
                st.session_state[dislike_clicked_key] = True
                st.session_state[comment_submitted_key] = False  # Reset submission status

            # Show input box only if 👎 was clicked
            if st.session_state.get(dislike_clicked_key):
                comment = st.text_input("What could be better?", key=comment_key)
                # If comment was typed and Enter is hit, save it
                if comment:
                    update_feedback_to_memory(turn["query"], turn["answer"], "dislike", comment)
                    st.toast("Feedback noted!", icon="👍")
                    st.session_state[comment_submitted_key] = True
                    st.session_state[dislike_clicked_key] = False  # Optionally reset to allow again
        
        st.divider()

    st.markdown('</div>', unsafe_allow_html=True)
#--------------------------------------------------------------

# User query box fixed at bottom
user_query = st.chat_input("How can I help you with Your Data Query Today..?")

if user_query:
    # Generate LLM answer with context
    # Recent turns for context
    recent_mem = get_recent_memory(6)
    
    # Generate answer (also returns top_matches for memory logging)
    answer, top_matches = get_llm_answer(user_query, df_metadata, index, memory=recent_mem)
    append_to_memory(user_query, answer, top_matches=top_matches)
    st.rerun()   # Rerun so new message appears

