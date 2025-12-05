# Electricity Analysis Dashboard – StateBook (2023)


An interactive and data-driven dashboard built using **Shiny for Python** to analyze the U.S. electricity landscape across **Operable**, **Proposed**, and **Retired** power plants.

![Dashboard Preview](dashboard_images/Combined_Analysis.png)

---

## 📌 Overview

This dashboard provides insights on:

- Power plant distribution across states  
- Technology-wise capacity trends  
- Retirements and new additions  
- Plant generation patterns  
- Suitability scoring for future energy projects  
- State-level maps, treemaps, sunburst charts & Sankey diagrams  

The tool helps energy stakeholders and analysts understand the capacity mix, growth patterns, and strategic opportunities across the United States.

---

## 🧩 My Contributions

In this multi-member capstone, **my primary work included**:

- Designing and building the **modular Shiny for Python UI**
- Implementing **multi-state dropdown filters** and dynamic sheet switching  
- Integrating **Plotly charts** for capacity, status, and distribution comparisons  
- Cleaning, merging, and preparing the **Operable, Proposed & Retired datasets**  
- Creating combined analysis logic and capacity calculations  
- Deploying the dashboard on **AWS EC2** with a reproducible environment  
- Creating visualizations for technology distribution, capacity trends & recommendation scoring  

---

## 📊 Key Dashboard Features

### **Operable Plants**
- Total utilities, plants, and generators  
- Capacity by technology (Nameplate, Summer, Winter)  
- Sector vs Status grouped bars  
- Uprates, derates, repowers  
- Technology distribution dot plot  
- Technology-year summary table  

### **Proposed Plants**
- Summary metrics  
- Sector vs status visual  
- Effective vs current year scatter  
- Planned retirement bubble chart  

### **Retired Plants**
- Summary metrics  
- Capacity + generator retirements over time  
- Sector vs status visual  

### **Combined Analysis**
- U.S. map of all plants  
- Capacity treemaps  
- Technology distribution  
- Net generation map  
- Fuel-type Sankey diagram  
- Recommendation scoring modal  
- Summary tables for energy mix  

---

## 🧮 Recommendation Formula

Used to evaluate whether a state is suitable for future power project investments:

```
Score = 0.5 × (Proposed Plants / (Retired Plants + 1)) 
      + 0.5 × (Proposed Capacity / (Retired Capacity + 1))
```

Categories:

| Score | Recommendation |
|-------|----------------|
| ≥ 2.0 | Highly Favorable |
| 1.0 – 1.99 | Favorable |
| 0.5 – 0.99 | Neutral |
| < 0.5 | Unfavorable |

---


## 🗂 File Structure

```
/app
   main.py

/data
   operable_sample.csv

/dashboard_images
   Operable.png
   Proposed.png
   Retired.png
   Combined_Analysis.png

README.md
```


---

## ▶️ Running the Dashboard (Local)

### 1️⃣ Install dependencies

Run the following command:

```
pip install pandas numpy plotly shiny dash openpyxl
```

### 2️⃣ Start the dashboard

```
shiny run --port 8000 app/main.py
```

The dashboard will open in your browser at:

http://localhost:8000


---

## 📘 Dataset

The full cleaned datasets used in the capstone are large and therefore **not included** in this repository.  
A **20-row sample** (`operable_sample.csv`) is provided to illustrate structure and preprocessing steps.

---

## 📌 Disclaimer

This project was originally developed as a **team capstone** at George Mason University.  
This repository reflects **my personal contributions**, cleaned structure, and documentation for portfolio purposes.  

---

## 💬 Contact

**Email:** sriyareddy696@gmail.com  
Feel free to reach out for any questions or feedback.

