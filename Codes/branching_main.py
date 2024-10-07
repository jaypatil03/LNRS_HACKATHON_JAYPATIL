import streamlit as st
from graphviz import Digraph
from diagrams import Diagram, Cluster
from diagrams.aws.network import ELB
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
import plotly.express as px
import pandas as pd

# Load libraries for NLP
import spacy
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# Utility functions
def process_text(text):
    return [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]

def identify_flowchart_elements(steps):
    flowchart = []
    for step in steps:
        lower_step = step.lower()
        if "if" in lower_step or "when" in lower_step:
            flowchart.append(("Condition", step))
        elif "else" in lower_step or "otherwise" in lower_step:
            flowchart.append(("Else", step))
        else:
            flowchart.append(("Action", step))
    return flowchart

def generate_flowchart(elements):
    dot = Digraph(format="png")
    condition_node = None
    last_node = None
    for i, (type_, text) in enumerate(elements):
        node_id = f'node_{i}'
        if type_ == "Condition":
            dot.node(node_id, text, shape="diamond", style="filled", color="lightblue")
            condition_node = node_id
        elif type_ == "Else":
            dot.node(node_id, text, shape="box", style="filled", color="lightgray")
            if condition_node:
                dot.edge(condition_node, node_id, label="No")
        else:
            dot.node(node_id, text, shape="box", style="filled", color="lightgreen")
            if condition_node:
                dot.edge(condition_node, node_id, label="Yes")
            if last_node and not condition_node:
                dot.edge(last_node, node_id)
        last_node = node_id
    return dot

# Streamlit UI
st.title("Diagram Generator from Text")

# Sidebar for different diagrams
st.sidebar.title("Select Diagram Type")
diagram_type = st.sidebar.selectbox("Choose the type of diagram:", 
                                    ("Flowchart", "Schematic Architecture", "Gantt Chart", "UML Diagram"))

user_input = st.text_area("Enter your instructions or data:", height=150)

# Flowchart Generation
if diagram_type == "Flowchart":
    if st.button("Generate Flowchart"):
        if user_input.strip():
            steps = process_text(user_input)
            if steps:
                flowchart_elements = identify_flowchart_elements(steps)
                flowchart = generate_flowchart(flowchart_elements)
                st.graphviz_chart(flowchart.source)
            else:
                st.error("No valid steps detected. Please try again.")
        else:
            st.error("Please enter some instructions to generate a flowchart.")

# Schematic Architecture Diagram
if diagram_type == "Schematic Architecture":
    if st.button("Generate Architecture Diagram"):
        if user_input.strip():
            with Diagram("3-Tier Architecture", show=False):
                with Cluster("Web Tier"):
                    lb = ELB("Load Balancer")
                with Cluster("Application Tier"):
                    app1 = EC2("App Server 1")
                    app2 = EC2("App Server 2")
                with Cluster("Database Tier"):
                    db = RDS("Database")
                lb >> [app1, app2] >> db
            st.image("3-tier_architecture.png")
        else:
            st.error("Please provide text instructions to generate the architecture.")

# Gantt Chart Generation
if diagram_type == "Gantt Chart":
    if st.button("Generate Gantt Chart"):
        if user_input.strip():
            # Example data to generate the Gantt chart
            data = {
                'Task': ["Project Planning", "Design Phase", "Development Phase", "Testing Phase", "Launch"],
                'Start': ["2024-10-01", "2024-10-06", "2024-10-11", "2024-10-21", "2024-10-26"],
                'Finish': ["2024-10-05", "2024-10-10", "2024-10-20", "2024-10-25", "2024-10-30"]
            }
            df = pd.DataFrame(data)
            fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", title="Project Timeline")
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig)
        else:
            st.error("Please provide project phases and timelines to generate a Gantt chart.")

# UML Diagram Generation
if diagram_type == "UML Diagram":
    if st.button("Generate UML Diagram"):
        if user_input.strip():
            uml_dot = Digraph(format="png")
            uml_dot.node("User", "User", shape="ellipse")
            uml_dot.node("Order", "Order", shape="ellipse")
            uml_dot.node("Product", "Product", shape="ellipse")
            uml_dot.edge("User", "Order", label="places")
            uml_dot.edge("Order", "Product", label="contains")
            st.graphviz_chart(uml_dot.source)
        else:
            st.error("Please provide the class definitions to generate a UML diagram.")
