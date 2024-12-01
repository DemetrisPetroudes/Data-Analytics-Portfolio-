# Bayesian Analysis of Limited Dependent Variable Models

This repository contains the implementation and analysis of two limited dependent variable models—Probit and Logit—using Bayesian techniques. 
The objective is to program posterior simulation methods from scratch and apply them to real-world datasets.

---

## Project Overview

The assignment focuses on:
- Implementing a **Gibbs sampler** for the Probit model.
- Implementing the **Metropolis-Hastings algorithm** for the Logit model.
- Applying these models to real-world datasets:
  - **Probit Model:** Mortgage application data (`loanapp.csv`).
  - **Logit Model:** Female labor market participation data (`mroz.csv`).
- Analyzing the posterior distributions and commenting on coefficients, convergence, and results.

---

## Models

### 1. Probit Model

The Probit model is useful for binary outcome variables.

### 2. Logit Model

The Logit model is also used for binary outcomes but assumes a logistic distribution for the errors in the latent variable equation.

Since the posterior cannot be simplified for direct sampling, a **Metropolis-Hastings algorithm** is used with a proposal distribution:

## Implementation Steps

### Probit Model
1. Define the latent variable structure and priors.
2. Implement the **Gibbs sampler**:
   - Sample y* (latent variable) from its truncated normal distribution.
   - Sample β (the coefficients) from its conditional posterior.
3. Apply the sampler to the dataset `loanapp.csv`.
4. Analyze posterior distributions and comment on results.

### Logit Model
1. Define the logistic likelihood and priors.
2. Implement the **Metropolis-Hastings algorithm**:
   - Calculate the MLE (Maximum Likelihood Estimate), denoted as β̂ ("beta hat").
   - Compute the Hessian matrix, H(β̂).
   - Generate proposals using the proposal distribution g(β), and accept/reject based on the Metropolis-Hastings criterion.
3. Apply the algorithm to the dataset `mroz.csv`.
4. Analyze posterior distributions and comment on results.


---

## Datasets
- **`loanapp.csv`**: Contains data on mortgage applications. Outcome variable: `approve`. Covariates include housing expenditures, loan amount, unemployment rate, number of dependents, and more.
- **`mroz.csv`**: Contains data on female labor market participation. Outcome variable: `inlf`. Covariates include education level, experience, age, and number of children.



5. Tune \( \tau \) to achieve an optimal acceptance rate.

6. Apply the Metropolis-Hastings algorithm to the dataset `mroz.csv`.

7. Analyze convergence and posterior estimates of the parameters.

