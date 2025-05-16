# Asset-Dynamics

This project implements and analyzes various methodologies for modeling the movements of financial asset prices. This repository implements multiple models of asset dynamics in Python and applies them to real financial market data for analysis.

## Table of Contents

1. [Introduction](#introduction)
2. [Model Overview](#model-overview)
   - [Additive Model](#additive-model)
   - [Multiplicative Model](#multiplicative-model)
   - [Random Walk](#random-walk)
   - [Wiener Process](#wiener-process)
   - [Geometric Wiener Process](#geometric-wiener-process)
   - [Geometric Brownian Motion (GBM)](#geometric-brownian-motion-gbm)
3. [Applications](#applications)
   - [Option Market Applications](#option-market-applications)
4. [Installation and Usage](#installation-and-usage)
5. [Dependencies](#dependencies)

## Introduction

Asset Dynamics is a field of study that focuses on mathematical methodologies for modeling changes in asset prices in financial markets. This project implements various asset pricing models in Python, applies them to real data, and analyzes their predictive performance and characteristics.

## Model Overview

### Additive Model

The additive model assumes that asset price changes occur by adding a fixed increment to the previous price.

```
S(t+1) = S(t) + μ + σε
```

Where:
- S(t): asset price at time t
- μ: mean change (drift)
- σ: volatility (standard deviation)
- ε: random variable following the standard normal distribution

Although simple, this model allows asset prices to become negative, which is unrealistic.

### Multiplicative Model

The multiplicative model assumes that asset price changes occur in proportion to the current price.

```
S(t+1) = S(t) × (1 + μ + σε)
```

Where:
- S(t): asset price at time t
- μ: average return
- σ: return volatility
- ε: random variable following the standard normal distribution

This model is more realistic than the additive model because asset prices always remain positive.

### Random Walk

The random walk model assumes that the change in the asset price at the next step is determined by a random value independent of the current price.

```
S(t+1) = S(t) + ε
```

Where:
- S(t): asset price at time t
- ε: independent and identically distributed (i.i.d.) random variable

This model is closely related to the Efficient Market Hypothesis (EMH), which assumes that market prices instantly reflect all publicly available information.

### Wiener Process

The Wiener process (also known as Brownian motion) is a generalization of the random walk in continuous time. It has the following properties:

1. W(0) = 0  
2. W(t) has independent increments  
3. W(t+s) - W(t) follows a normal distribution with mean 0 and variance s  
4. W(t) has continuous paths

Mathematically:
```
dW(t) = ε√dt
```

Where:
- dW(t): increment of the Wiener process
- ε: random variable following the standard normal distribution
- dt: infinitesimal time increment

The Wiener process is a basic building block for modeling randomness in finance.

### Geometric Wiener Process

The geometric Wiener process is based on the Wiener process but assumes that asset prices follow a log-normal distribution.

```
dS(t) = μS(t)dt + σS(t)dW(t)
```

Where:
- S(t): asset price at time t
- μ: expected return (drift)
- σ: volatility
- dW(t): increment of the Wiener process

### Geometric Brownian Motion (GBM)

Geometric Brownian Motion is one of the most widely used continuous-time models in finance. 
It forms the basis of the Black-Scholes option pricing model and is expressed as:

```
dS(t) = μS(t)dt + σS(t)dW(t)
```

Solution:

```
S(t) = S(0)exp((μ - σ²/2)t + σW(t))
```

Where:
- S(t): asset price at time t
- S(0): initial asset price
- μ: expected return
- σ: volatility
- W(t): Wiener process

GBM ensures that asset prices remain positive and that log-returns are normally distributed.

## Applications

### Option Market Applications

This project applies the implemented asset price models to option pricing. In particular, using the Geometric Brownian Motion (GBM), the following steps are carried out:

1. Simulate asset price paths under multiple scenarios  
2. Simulate option exercise at maturity  
3. Analyze the distribution of option returns  
4. Evaluate various performance metrics (MDD, CVaR, ratios, etc.)

This provides a framework for evaluating the risk and return of option investments.

## Installation and Usage

1. Clone the repository:
   ```
   git clone https://github.com/HAWONNA/Asset-Dynamics.git
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run Jupyter Notebook or Python script:
   ```
   jupyter notebook asset-dynamics.ipynb
   ```
   or
   ```
   python asset_dynamics_main.py
   ```

## Dependencies

- pandas
- numpy
- matplotlib
- scipy
- statsmodels
- seaborn
- pandas-datareader
