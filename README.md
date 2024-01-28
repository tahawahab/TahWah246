
# Project Documentation

## Overview

This project focuses on developing and deploying a machine learning model aimed at classifying German text phrases into predefined categories. The core objective is to train a model capable of categorizing German search queries (short phrases) using a dataset provided for this purpose. The deployment of the final model is facilitated through a REST API, encapsulated within a Docker container to streamline the process.

This repository encompasses a comprehensive workflow that includes exploratory data analysis, data preprocessing, model selection, and deployment. The repository is structured to facilitate understanding and interaction with the machine learning model and its deployment.

## Repository Structure

- **Exploratory Data Analysis & Model Selection:** A Jupyter notebook is included, featuring detailed commentary and justifications for each step undertaken during the exploratory data analysis and data preprocessing phases. It also elaborates on the choice of the machine learning algorithm, providing insight into the decision-making process.

- **Model Training:** The `model_training.py` file contains the implementation for training the optimal model, as identified through the exploratory data analysis and model selection process.

- **Application Deployment:** The `main.py` file serves to load the trained model and initiate the FastAPI application. This setup ensures seamless integration of the model within a web service framework.

- **API Testing:** The `test_api.py` file comprises various test scenarios for the API, designed to validate its functionality comprehensively. These tests are conducted using pytest, facilitating thorough evaluation.

- **Docker Deployment:** A Dockerfile is provided to facilitate the application's containerization, simplifying the deployment process. Users can build and run the Docker container, thereby hosting the application on port 8000 for easy access and testing.

## Running the Application

To deploy the application:

1. **Containerization:** Use the Dockerfile to containerize the application, a process that involves building a Docker image that encapsulates the application for deployment.

2. **Running the Container:** Following the image build, run the container to make the application available on port 8000, ready for interaction and testing.

3. **Testing the API:** With the application operational, the API endpoints can be tested using `curl` commands or any HTTP client of choice. Predefined test cases in the `test_api.py` file facilitate the API's functionality validation.

## Testing

The application has been rigorously tested across all foundational test cases to ensure its reliability and functionality. These tests aim to cover a broad spectrum of scenarios, affirming the API endpoints' correct behavior.

---
This documentation provides a comprehensive guide to understanding the project's objectives, deploying the application, and conducting tests efficiently.
