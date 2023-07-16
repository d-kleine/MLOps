# Project: A Dynamic Risk Assessment System

This repository contains the code and resources for a dynamic risk assessment system. The system involves creating, deploying, and monitoring an ML model that predicts attrition risk for a company's clients. The goal is to provide accurate risk assessments to help prevent client attrition and minimize revenue loss.

## Background

Imagine you are the Chief Data Scientist at a large company with 10,000 corporate clients. The company is concerned about attrition risk, which refers to the risk of clients exiting their contracts and reducing revenue. To address this issue, you need to create, deploy, and monitor an ML model that estimates the attrition risk for each client.

The model should accurately predict the risk, enabling the client managers to focus on high-risk clients and prevent attrition. Additionally, the model should be regularly monitored and updated to ensure accuracy and adaptability to dynamic business conditions.

## Project Structure

The project is structured into several steps, each focusing on a specific aspect of the ML workflow:

1. **Data Ingestion**: Ingest new data, compile training data, and save it to persistent storage.
2. **Training, Scoring, and Deploying**: Train an ML model to predict attrition risk, score the model, and deploy it.
3. **Diagnostics**: Perform diagnostic tests on the model and data, including summary statistics, performance timing, and dependency checks.
4. **Reporting**: Generate reports, plots, and metrics related to the model's performance.
5. **Process Automation**: Automate the entire ML pipeline using scripts and a cron job.

## Repository Structure

The repository structure is organized as follows:
```
- data/
  - practicedata/      # Practice data for testing and development
  - sourcedata/        # Production data for the final model
- models/              # Directory to store trained models and scores
- ingestion.py         # Script for data ingestion
- training.py          # Script for model training
- scoring.py           # Script for model scoring
- deployment.py        # Script for model deployment
- diagnostics.py       # Script for model and data diagnostics
- reporting.py         # Script for generating plots and reports
- app.py               # API setup for accessing model diagnostics and results
- apicalls.py          # Script to call API endpoints and generate combined outputs
- fullprocess.py       # Script for process automation, including checking for new data, model drift, re-training, and reporting
- cronjob.txt          # Cron job schedule for running the full process script
- config.json          # Configuration file for specifying input/output paths and other settings
- README.md            # Repository readme file (you're reading it now)
```