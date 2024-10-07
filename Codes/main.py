import streamlit as st
import spacy
from graphviz import Digraph

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text)
    steps = []
    for sentence in doc.sents:
        steps.append(sentence.text)
    return steps

def identify_flowchart_elements(steps):
    flowchart = []
    for step in steps:
        if "if" in step.lower():
            flowchart.append(("Condition", step))
        elif "then" in step.lower() or "else" in step.lower():
            flowchart.append(("Decision", step))
        else:
            flowchart.append(("Action", step))
    return flowchart

def generate_flowchart(elements):
    dot = Digraph()
    for i, (type, text) in enumerate(elements):
        if type == "Condition":
            dot.node(f'node_{i}', text, shape="diamond")
        else:
            dot.node(f'node_{i}', text, shape="box")
        if i > 0:
            dot.edge(f'node_{i-1}', f'node_{i}')
    return dot

st.title("Flowchart Generator from Text")

# Input from user
user_input = st.text_area("Enter your instructions:")

if st.button("Generate Flowchart"):
    steps = process_text(user_input)
    flowchart_elements = identify_flowchart_elements(steps)
    flowchart = generate_flowchart(flowchart_elements)
    st.graphviz_chart(flowchart.source)

