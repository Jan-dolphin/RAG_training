# Robust RAG Training Project

This project is a comprehensive guide to building and evaluating Retrieval-Augmented Generation (RAG) systems. It is designed to be run as a series of educational notebooks, moving from basic concepts to advanced retrieval strategies and robust evaluation.

## Project Structure

- `data/`: Contains synthetic data for various domains (Technical Docs, Legal, Support Logs, Finance).
- `notebooks/`: The core learning modules.
    - `01_Ingestion_and_Basic_RAG.ipynb`: Basics of loading, splitting, embedding, and retrieving.
    - `02_Advanced_Retrieval.ipynb`: Advanced techniques like MultiQuery, ParentDocument, and Re-ranking.
    - `03_Evaluation_RAGAS.ipynb`: How to scientifically measure RAG performance using synthetic test sets and metrics.
    - `04_End_to_End_and_Monitoring.ipynb`: Putting it all together.

## Setup in Deepnote

1. **Environment**: Ensure you are using a Python 3.9+ environment.
2. **Install Dependencies**:
   Open a terminal in Deepnote or run a cell with:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Variables**:
   You will need an OpenAI API Key.
   - Create a `.env` file or set the environment variable `OPENAI_API_KEY` in Deepnote's integration settings.

## Data
The `data/` directory is pre-populated with synthetic examples to verify the pipelines without needing external datasets.
