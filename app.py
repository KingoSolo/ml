import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize the chatbot model (DialoGPT for now)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Question Answering pipeline (using BERT-based model)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Summarization pipeline
summarizer = pipeline("summarization", model="t5-small")

# Helper functions
def chat_with_bot(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def summarize_text(text):
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Streamlit UI
st.title("AI Study Assistant")

# Chatbot interaction
st.subheader("1. Chat with Study Assistant")
user_query = st.text_input("Ask me anything:")
if user_query:
    bot_response = chat_with_bot(user_query)
    st.write(f"**Bot:** {bot_response}")

# Question answering
st.subheader("2. Ask Specific Questions")
context = st.text_area("Paste your study material or notes here:")
question = st.text_input("What do you want to know?")
if st.button("Get Answer"):
    if context and question:
        answer = answer_question(question, context)
        st.write(f"**Answer:** {answer}")
    else:
        st.write("Please provide both context and a question.")

# Summarization
st.subheader("3. Summarize Notes")
notes = st.text_area("Paste your notes here for summarization:")
if st.button("Summarize"):
    if notes:
        summary = summarize_text(notes)
        st.write(f"**Summary:** {summary}")
    else:
        st.write("Please provide notes to summarize.")
