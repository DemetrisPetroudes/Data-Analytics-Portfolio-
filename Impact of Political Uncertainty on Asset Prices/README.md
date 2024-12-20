# The Impact of Political Uncertainty on Asset Prices (2014-2024)

## Project Overview
This project investigates the impact of political uncertainty on stock market dynamics across 14 European countries over a decade (2014-2024). Utilizing an event study approach and volatility modeling, it examines how major political events influence investor behavior, market returns, and volatility. The findings offer valuable insights for policymakers, investors, and analysts in managing risks associated with political instability.

## Key Objectives
- Analyze the effects of political events on stock returns and volatility around different time windows.
- Explore how political uncertainty impacts market dynamics using ARX-GARCH models.
- Provide actionable insights into investor behavior and market resilience during times of political uncertainty.

## Methodology
- **Data Collection**: Daily stock price data for 14 European stock markets were sourced from Yahoo Finance and Google Finance. Political events included key geopolitical shifts such as Brexit, the Russian annexation of Crimea, and the COVID-19 pandemic.
- **Models Used**:
  - **ARX-GARCH(1,1)**: Combines autoregressive modeling with volatility estimation to capture market reactions and conditional variance.
  - **Event Study**: Examined stock returns and volatility in three-, seven-, and 14-day windows surrounding major political events.
  - **Tools**: Analysis performed using Python with the `arch` library for GARCH modeling.

## Main Findings
1. **Increased Volatility**: Political uncertainty consistently led to heightened market volatility, reflecting investor risk aversion and uncertainty.
2. **Sectoral and Regional Variation**: The impact of political events varied significantly across sectors and countries, influenced by the nature of the event and economic context.
3. **Short-term Declines**: Stock returns often experienced short-term declines during political crises, with recovery patterns varying by event type.
4. **Event Highlights**:
   - Brexit caused significant volatility across European markets, particularly in countries with strong trade ties to the UK.
   - COVID-19 lockdowns led to sharp declines in stock prices, followed by recovery upon vaccine approvals.
   - The Russian invasion of Ukraine had a profound effect on energy-dependent markets.

## Repository Structure
- `code/`: Contains Python scripts for ARX-GARCH modeling and event analysis.
- `data/`: Includes anonymized sample data for replicating results.
- `README.md`: Project documentation.

---

This work was completed as part of the MSc in Data Analytics for Economics and Finance at the University of Glasgow.
