# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a credit card fraud detection project focused on building machine learning models to identify fraudulent transactions. The project contains transaction datasets for training and testing fraud detection algorithms.

## Dataset Structure

The project uses credit card transaction datasets located in `archive (1)/`:
- `fraudTrain.csv` - Training dataset with transaction features and fraud labels
- `fraudTest.csv` - Test dataset for model evaluation

Dataset features include:
- Transaction details: amount, merchant, category, timestamp
- Customer info: name, demographics, location
- Geographic data: latitude/longitude for both customer and merchant
- Target variable: `is_fraud` (0 = legitimate, 1 = fraudulent)

## Development Environment

Since this appears to be a data science project, typical commands will likely include:

### Python/Jupyter Environment
```bash
# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt

# Launch Jupyter notebook for data exploration
jupyter notebook

# Run Python scripts
python main.py
python train_model.py
```

### Common ML/Data Science Libraries Expected
- pandas for data manipulation
- numpy for numerical operations
- scikit-learn for machine learning models
- matplotlib/seaborn for visualization
- jupyter for interactive development

## Project Architecture

This is primarily a data science/machine learning project structure:
- Raw data in CSV format containing transaction records
- Expected development in Python with Jupyter notebooks or scripts
- Typical ML pipeline: data preprocessing → feature engineering → model training → evaluation
- Focus on binary classification (fraud vs legitimate transactions)

## Data Considerations

- Dataset contains sensitive-looking transaction data (credit card numbers, personal info)
- All data appears to be synthetic/simulated for educational purposes
- When working with this data, treat it as if it were real sensitive financial data
- Implement proper data handling and privacy considerations in any code

## Plan and Review

### Before Starting Work
- Always enter plan mode to make a comprehensive plan
- After creating the plan, write it to `.claude/tasks/TASK_NAME.md`
- The plan should be a detailed implementation plan with reasoning behind each step and tasks broken down
- If the task requires external knowledge or packages, research to get latest information (use Task tool for research)
- Don't over-plan - always think MVP first
- Once you write the plan, ask for review and approval before proceeding

### While Implementing
- Update the plan as you work and make progress
- After completing tasks in the plan, update with detailed descriptions of changes made
- This ensures following tasks can be easily handed over to other engineers
- Keep documentation current with implementation progress

## AI Prompts Directory

The `ai-prompts/` directory contains project-related prompt files that may provide additional context for development tasks.