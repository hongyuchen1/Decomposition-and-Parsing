# Decomposition-and-Parsing

## Overview
This repository contains the code and annotated dataset for the project titled "Clinical Trial Eligibility Criteria Decomposition and Parsing with Large Language Models". The project introduces a novel Decomposition and Parsing (DP) workflow designed to systematically break down complex clinical trial eligibility criteria into structured study traits. The workflow leverages large language models (LLMs), including GPT-4o and Llama3.3, to enhance automation in decomposition and parsing eligibility criteria.
## Key Features
- DP Workflow:
  - Breaks down complex eligibility criteria into discrete study traits.
  - Utilizes Disjunctive Normal Form (DNF) to represent logical relationships.
  - Parses traits into structured components (main entity, modifiers, constraints, and negation).
- LLM Integration:
  - Leverages GPT-4o and Llama3.3 for decomposition and parsing tasks for demonstration but could support other LLMs via vLLM.
  - Provides a Robust evaluation metrics to improve the performance of the LLMs.
- Annotated Dataset:
  - Alzheimer's disease (AD) trial criteria dataset from ClinicalTrials.gov.
  - Includes structured annotations for study traits, logical relationships, and structured components. 
## Installation
To install the required dependencies, run:
```bash
# Clone the repository
git clone https://github.com/hongyuchen1/Decomposition-and-Parsing.git
cd Decomposition-and-Parsing

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows use 'env\Scripts\activate'

# Install dependencies
pip install -r requirements.txt
```
## Usage
To use GPT, you need to create a `.env` file in the root directory with the following content:
```bash
API_KEY=<YOUR_OPENAI_APIKEY>
```
To use Llama3.3 or other open-source LLMs, you need to host a OpenAI Compatible Server with [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

Detailed instructions on how to use the codebase can be found in the [tutorial](https://github.com/hongyuchen1/Decomposition-and-Parsing/blob/main/tutorial) directory. The notebooks provide a step-by-step guide on

# Organization
- Health Outcomes & Biomedical Informatics, College of Medicine, University of Florida, Florida, USA
- Section of Biomedical Informatics and Data Science, School of Medicine, Yale University, New Haven, USA
- Biostatistics and Health Data Science, School of Medicine, Indiana University, Indiana, USA
# Contact
- If you have any questions, please raise an issue in the GitHub