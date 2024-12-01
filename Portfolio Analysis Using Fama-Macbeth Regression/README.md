# Portfolio Analysis Using Fama-Macbeth Regression

This project is a comprehensive analysis of U.S. stock returns based on the Fama-Macbeth (1973) methodology. 
The study utilizes a dataset of monthly stock returns for 438 U.S. companies, spanning the period from January 1980 to December 2015. 
The objective is to explore the validity of the Capital Asset Pricing Model (CAPM) by forming portfolios, estimating parameters, and testing hypotheses
over multiple time periods.

## Project Overview

### Dataset
The dataset, `USstocks_balanced.csv`, includes:
- **permno**: Unique identifier for each company.
- **year, month**: Time indicators for each observation.
- **ri**: Monthly stock returns (in %).
- **rm**: Monthly returns of the S&P 500 index (proxy for market returns).
- **rf**: Monthly returns of 3-month U.S. Treasury Bills (proxy for risk-free returns).

### Methodology
1. **Portfolio Formation**: Stocks were divided into 20 portfolios based on specific criteria, ensuring balanced representation and meaningful analysis.
2. **Time Periods**: Analysis was conducted over six distinct time intervals, each with:
   - A portfolio formation period.
   - An estimation period.
   - A testing period.
3. **Regression Analysis**: The Fama-Macbeth two-step regression method was applied:
   - Step 1: Estimation of portfolio-specific betas.
   - Step 2: Cross-sectional regressions of portfolio returns on betas to estimate risk premiums.
4. **Model Evaluation**: Hypothesis testing was performed to assess CAPM's validity. Potential biases (e.g., test statistics, errors) were identified
                         and corrected where possible.
### Portfolio Formation, Estimation, and Testing Periods

The analysis follows the Fama-Macbeth (1973) procedure, which requires dividing the dataset into distinct time intervals for portfolio formation, parameter estimation, and hypothesis testing. These steps are structured as follows:

1. **Portfolio Formation Periods**: 
   - Stocks are grouped into 20 portfolios based on their characteristics during the designated formation periods.
   - Six formation periods are defined:
     - 1980–1986
     - 1984–1990
     - 1988–1994
     - 1992–1998
     - 1996–2002
     - 2000–2006

2. **Initial Estimation Periods**:
   - Portfolio-specific betas and other parameters are estimated using stock returns from these periods. 
   - The estimation windows are as follows:
     - 1987–1991
     - 1991–1995
     - 1995–1999
     - 1999–2003
     - 2003–2007
     - 2007–2011

3. **Testing Periods**:
   - Hypotheses are tested and results are validated during these periods.
   - The testing periods are:
     - 1992–1995
     - 1996–1999
     - 2000–2003
     - 2004–2007
     - 2008–2011
     - 2012–2015

### Methodology Note:
Each phase of the process is sequential and non-overlapping within a single analysis window but overlaps across different periods. 
This ensures robustness in portfolio evaluation and allows for analysis across varying economic conditions.

### Note
The included script demonstrates the analysis for the first time period (1992–1995). The process is identical for the remaining periods. 
You can adjust the time frames directly in the script or use the automated version to run the analysis for all periods sequentially.

### Key Findings
- Regression results were summarized and tabulated, replicating the key insights from Fama and Macbeth (1973).
- Evidence for or against CAPM was evaluated based on statistical tests and economic reasoning.

## Features
- **Data Analysis**: Comprehensive exploration of stock returns and portfolio performance.
- **Replicable Workflow**: The entire analysis is coded in R, ensuring reproducibility.
- **Statistical Insights**: In-depth examination of CAPM's assumptions and predictions.

## File Structure
- `USstocks_balanced.csv`: Is the dataset used
-  period 1.R: Includes the R scripts for data cleaning, portfolio formation, regression analysis, and visualization.
- `results.dox`: Summarized tables and findings from the analysis.
- `README.md`: Overview of the project.


