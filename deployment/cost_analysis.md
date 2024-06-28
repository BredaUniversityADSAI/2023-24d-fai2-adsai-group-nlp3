# Cost Analysis Report for Cloud Deployment of Machine Learning Model 
---

### Introduction

In this report, we conduct a thorough cost analysis for the deployment of a Machine Learning (ML) endpoint, specifically designed for real-time emotion prediction from text data. The analysis focuses on understanding the computational requirements and proposing a pricing strategy to ensure profitability. Key findings suggest that with efficient use of computational resources, the deployment can be cost-effective while providing significant value to customers.

Additionally, we perform a detailed training cost analysis to estimate the expenses involved in training the model. By leveraging an Nvidia A100 GPU on Azure, we calculate the costs based on different dataset sizes and training durations. This comprehensive analysis provides insights into both deployment and training expenses, enabling informed decisions for budget planning and resource allocation in machine learning projects.

## Project Overview

### Model and Environment Setup

- **Model:** Utilizes a RoBERTa-based model specifically trained for emotion detection in textual data.
- **Environment:** Deployed within an Azure Machine Learning workspace (`NLP3`) on Kubernetes compute (`adsai1`).

**Model Deployment:**
   - **Strategy:** Utilizes blue-green deployment (`blue-deployment-test`) for seamless updates and zero downtime.
   - **Resources:** Configured with `6000m` CPU and `20Gi` memory to handle prediction requests efficiently.
   - **Location:** Deployed in `westeurope` for optimal performance and compliance.


## Methodology

- **Average Response Time**: Each query takes approximately 0.01 seconds.
- **Query Volume**: 100 queries were processed, totaling 1 second.
- **Computational Power**: The ML model can handle 360,000 queries per hour.
- **Cost Analysis**: Calculating the cost per hour of computational resources.

---

## Inference Cost Analysis

To determine the feasibility of the deployment, we need to ascertain the cost of computational resources per hour, denoted as $C_{hour}$. This cost encompasses various factors such as cloud provider charges, instance type, and operational expenses.

### Hour Cost Calculation:

Since the model can handle 360,000 queries per hour and assuming continuous operation for 1 hour:

- **Compute Cost per Hour**: $0.0846

### Monthly Cost Calculation:

To find the monthly cost, multiply the hourly cost by the number of hours in a month:

- **Hours in a month** (assuming 30 days and 24 hours per day): 
30 × 24 = 720 hours

- **Monthly cost**:
Monthly Cost = $0.0846 × 720 = **$60.91**



## Comparison with Other Deployment
Sentiment Analysis API using AWS Lambda and API Gateway

#### Architecture Overview

- **Model:** Uses a BERT-based model for sentiment analysis of text data.
- **Deployment:** Serverless architecture on AWS Lambda for compute and API Gateway for endpoint management.
- **Scalability:** Automatically scales with incoming requests, ideal for variable workloads.

#### Cost Breakdown

- **AWS Lambda:**
  - **Invocation Costs:** $0.20 per 1 million requests (First 1 million requests are free).
  - **Compute Time:** $0.00001667 per GB-second of compute time.

- **API Gateway:**
  - **Request Costs:** $3.50 per million requests (First 1 million requests are free).
  - **Data Transfer:** $0.09 per GB out (free tier includes 1 GB/month).

- **Storage (S3 for model artifacts):**
  - **Storage Costs:** $0.023 per GB per month.

- **Estimated Total Monthly Cost:** Approximately $10 - $50 per month, depending on traffic volume and data transfer.


## Pricing Strategy

To ensure profitability while offering value to customers, it is essential to develop a comprehensive pricing strategy that aligns with both usage patterns and operational costs. By implementing a flexible and scalable pricing model, you can cater to different customer segments and usage scenarios. Below are some proposed pricing strategies:

1. **Pay-Per-Use Model:** Implement a pricing strategy that charges customers based on the number of predictions or queries made to the ML model. This approach aligns costs directly with usage, ensuring that expenses scale with customer demand.
   
2. **Tiered Pricing:** Introduce tiered pricing plans based on usage levels. For instance, offer different pricing tiers with varying limits on the number of predictions per month. This can appeal to different customer segments while ensuring predictable revenue.

3. **Subscription Plans:** Offer subscription-based pricing models where customers pay a fixed monthly fee for a certain number of predictions. This provides customers with cost predictability and ensures a steady revenue stream for your service.
   
### Conclusion

The estimated monthly cost for deploying your RoBERTa-based model on Azure Machine Learning using Kubernetes (`adsai1`), based on the provided hourly compute cost of $0.0846, would be approximately **$60.91**. This cost includes the compute resources needed to handle the model's capabilities and operational expenses associated with maintaining the deployment.

This estimation provides a basis for understanding the recurring expenses involved in running the ML endpoint on Azure. It's essential to monitor usage patterns and optimize resource utilization to manage costs effectively over time. Adjustments in pricing strategy, such as implementing pay-per-use models or scaling resources based on demand, can further optimize costs and ensure profitability.
   
## Model Training Cost Analysis

In the realm of machine learning, training models efficiently and cost-effectively is crucial for both academic research and commercial applications. This section provides a comprehensive cost analysis for training a machine learning model on Azure, utilizing an Nvidia A100 GPU. By estimating the time required to train the model and calculating the associated costs, we aim to offer insights into budget planning and resource allocation for machine learning projects. The analysis covers different dataset sizes to illustrate how costs scale with the volume of data, ensuring that stakeholders can make informed decisions to optimize their machine learning workflows.

### Prediction Model Time Estimate

- **Data Size:** 100 rows
- **Training Epochs:** 5
- **Training Time:** 20 minutes for 5 epochs

Using this information, we can scale the training time for larger datasets and multiple epochs as needed.

### GPU Machine Cost on Azure

To determine the cost of training the model on Azure, we need to select an appropriate GPU-enabled virtual machine (VM). For this analysis, we will consider the Nvidia A100 GPU, which is known for its high performance in deep learning tasks.

#### Azure Nvidia A100 VM Specifications

- **GPU:** 1 Nvidia A100
- **Cost:** Approximately $1.29 per hour

### Training Cost Calculation

Given the training duration of 20 minutes for 100 rows and 5 epochs, we can extrapolate this to estimate costs for longer training sessions.

1. **Training Duration per 100 Rows for 5 Epochs:**
   - **Minutes:** 20
   - **Hours:** 20 / 60 = 0.333 hours

2. **Cost per 20 Minutes (for 100 Rows):**
   - **VM Cost per Hour:** $1.29
   - **Cost for 0.333 Hours:** $1.29 * 0.333 ≈ $0.43

3. **Scaling for Larger Data:**
   - To generalize, if the training time scales linearly, the cost for larger datasets can be estimated by multiplying the cost for 100 rows by the desired data size factor.

### Example Cost for Larger Dataset

Assuming a dataset of 1,000 rows (10 times the original size):

- **Training Time:** 20 minutes * 10 = 200 minutes (3.33 hours)
- **Cost for 3.33 Hours:** $1.29 * 3.33 ≈ $4.30

Similarly, for a dataset of 10,000 rows (100 times the original size):

- **Training Time:** 20 minutes * 100 = 2,000 minutes (33.33 hours)
- **Cost for 33.33 Hours:** $1.29 * 33.33 ≈ $43.00
  
### Conclusion

The cost of training a model on Azure using an Nvidia A100 GPU can vary significantly based on the dataset size and the number of epochs. For a small dataset of 100 rows and 5 epochs, the training cost is approximately $0.43. This cost scales linearly with larger datasets and more training epochs, reaching around $4.30 for 1,000 rows and $43.00 for 10,000 rows.

By understanding these costs, you can make informed decisions about resource allocation and budget planning for your machine learning projects on Azure. Adjusting the dataset size, number of epochs, and optimizing the training process can help manage costs effectively while ensuring robust model performance.



## References
Spot.io. (n.d.). The complete guide to Azure pricing. Retrieved June 28, 2024, from https://spot.io/resources/azure-pricing/the-complete-guide/

Google Cloud. (n.d.). Demystifying cloud pricing: A comprehensive guide for businesses. Retrieved June 28, 2024, from https://cloud.google.com/blog/topics/cost-management/demystifying-cloud-pricing-a-comprehensive-guide-for-businesses

GetDeploying. (2024). Cloud GPU Costs. GetDeploying.com. Retrieved June 28, 2024, from https://getdeploying.com/reference/cloud-gpu
