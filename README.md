# 🛒 Online Retail Analysis & Customer Segmentation

## 📘 Overview
This project performs **data cleaning, customer segmentation (K-Means Clustering)**, and **Market Basket Analysis (Apriori Algorithm)** on the **Online Retail Dataset** to uncover key customer patterns and frequently bought-together products.

---

## ⚙️ Workflow

### 1. Data Cleaning
- Removed missing values and negative quantities/prices  
- Converted `InvoiceDate` to datetime  
- Extracted `Date`, `Hour`, and `DayOfWeek` for time-based insights  
- Created `Total_Price = Quantity * UnitPrice`

### 2. Exploratory Insights
- **Daily Transactions:** Unique invoices per day  
- **Country Analysis:** Top countries by order volume and revenue  
- **Top Products:** Identified top 10 items by quantity and total revenue  
- **Revenue Trend:** Visualized daily revenue patterns  
- **Sales Heatmap:** Revenue by hour and day of week  

### 3. Customer Segmentation (K-Means)
- Aggregated customer-level data:  
  `TotalSpend`, `AvgSpend`, `NumOrders`, `TotalQty`, `AvgPrice`, `Country`
- Standardized data with **StandardScaler**  
- Used **Elbow Method** to find optimal clusters  
- Segmented customers into **4 clusters** based on spending and behavior  

### 4. Market Basket Analysis (Apriori)
- Generated frequent itemsets and association rules per cluster  
- Extracted strong rules using **lift**, **confidence**, and **support**  


---

## 🧰 Tools & Libraries
Python • Pandas • NumPy • Seaborn • Scikit-learn • Datashader • Mlxtend (Apriori)

---

## 📊 Output
- `customer_df` → Segmented customer dataset  
- `MBA.csv` → Association rules for product combinations  

---

## 🚀 Key Insights
- Top-selling items and high-revenue products  
- Customer clusters reveal spending and quantity behavior  
- Association rules identify frequently co-purchased products  

---

## 🏁 Final Command
```python
df.to_csv('MBA.csv', index=False, columns=['antecedents', 'consequents'])

