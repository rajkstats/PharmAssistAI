---
title: PharmAssistAI
image: pharmassist.jpg
emoji: 😻
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
license: openrail
---

# PharmAssistAI: Your Advanced Pharma Research Assistant

PharmAssistAI revolutionizes how pharmacy professionals and students approach learning and research related to FDA-approved drugs. By integrating modern information retrieval technologies with Large Language Models (LLMs), PharmAssistAI optimizes the research and learning workflow, making it less time-consuming and more efficient.

## Core Features

- **Comprehensive Data Access**: Directly tap into the FDA drug labels dataset, with plans to incorporate the FDA adverse reactions dataset for a fuller data spectrum.
- **Dynamic Retrieval**: Utilize the Retrieval-Augmented Generation (RAG) technique for dynamic, real-time data retrieval.
- **Intelligent Summaries**: Leverage LLMs to generate insightful summaries and contextual answers.
- **Interactive Learning**: Engage with AI-generated related questions to deepen understanding and knowledge retention.
- **Research Linkage**: Automatically fetch and link relevant academic papers from PubMed, enhancing the depth of available information and supporting academic research.

## Monitoring and Evaluation

- **Real-Time Feedback with LangSmith**: Use LangSmith to incorporate real-time feedback and custom evaluations. This system ensures that the AI's responses are not only accurate but also contextually aware and user-focused.
- **Custom Evaluators for Enhanced Accuracy**: Deploy custom evaluators like PharmAssistEvaluator to ensure responses meet high standards of relevance, safety, and perception as human-generated versus AI-generated.

## How It Works

1. **Query Input**: Pharmacists type in their questions directly.
2. **Data Retrieval**: Relevant data is fetched from comprehensive datasets, including automated searches of PubMed for related academic papers.
3. **Data Presentation**: Data is displayed in an easily digestible format.
4. **Summary Generation**: Summaries of the data are created using GenAI
5. **Question Suggestion**: Suggest related questions to encourage further exploration.

## Architecture

![RAG Architecture](https://i.imgur.com/QPNipiI.png)


## Hugging Face App Demo

Experience our app [live](https://huggingface.co/spaces/rajkstats/PharmAssistAI) on Hugging Face:

**Home Screen** 

![Home Screen](https://i.imgur.com/SCasi55.png)

**Demo Screen** 

![Demo Screen](https://i.imgur.com/5GUOYHk.png)

## LangSmith Performance Insights

Explore the effectiveness and interaction tracking of LangSmith in PharmAssistAI through these detailed screenshots:

**Overview of Real-Time Evaluations** 

![Real-Time Evaluations](https://i.imgur.com/H7wkAnl.png)

**Detailed Feedback Example** 

![Feedback Example](https://i.imgur.com/xhxelcx.png)

**Interaction Metrics Dashboard**

![Metrics Dashboard](https://i.imgur.com/H9Q8OKj.png)


## Development Roadmap

- Integrate and index the complete FDA Drug Labeling and Adverse Events datasets.
- Refine the user interface for enhanced interaction and accessibility.
- Develop AI-driven educational tools like flashcards and study guides for mechanism of action.
- Enhance the retrieval system to include more open-source and advanced embedding models for better precision and efficiency.

## Quick Start Guide

Simply enter your question about any FDA-approved drug in our chat interface, and PharmAssistAI will provide you with detailed information, summaries, and follow-up questions to help expand your research and understanding.

## Feedback and Contributions

We value your input and invite you to help us enhance PharmAssistAI:

- 🐛 [Report an issue](https://github.com/rajkstats/pharmassistai/issues) on GitHub for technical issues or feature suggestions.
- 📧 Contact us at [raj.k.stats@gmail.com](mailto:raj.k.stats@gmail.com) for direct support or inquiries.