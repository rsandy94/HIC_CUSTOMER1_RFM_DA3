# HIC_CUSTOMER1_RFM_DA3

### Task 1: Data Preparation

- **Description**: The first step involves cleaning and preparing the data by handling missing values, outliers, and ensuring correct formats. Clean data is critical to ensure accurate analysis.
  
- **Code Instructions**:
  1. Load the dataset and check for missing values.
  2. Fill or remove missing data and handle outliers.
  3. Ensure columns are in the correct format (e.g., dates as `datetime`).

```python
import pandas as pd
import numpy as np

# Load the data (assuming it is in CSV format)
df = pd.read_csv('rfm_data.csv')

# Check for missing values
print(df.isnull().sum())

# Fill missing monetary values with median
df['Total Spend (Monetary Value)'].fillna(df['Total Spend (Monetary Value)'].median(), inplace=True)

# Convert dates to datetime format
df['Last Purchase Date'] = pd.to_datetime(df['Last Purchase Date'])
df['Account Created Date'] = pd.to_datetime(df['Account Created Date'])

# Handle outliers (cap large values in Total Spend, optional)
df['Total Spend (Monetary Value)'] = np.where(df['Total Spend (Monetary Value)'] > 5000, 5000, df['Total Spend (Monetary Value)'])

# Check the cleaned data
df.head()
```

---

### Task 2: Calculating Recency

- **Description**: Recency is calculated by measuring how long it has been since the customer’s last purchase. Customers who purchased recently are more likely to engage again.
  
- **Code Instructions**:
  1. Create a reference date (e.g., today's date).
  2. Calculate the time difference between the reference date and the `Last Purchase Date`.
  3. Store the result in a new column called `Recency` (in days).

```python
# Set reference date (today's date)
reference_date = pd.Timestamp('2024-10-07')

# Calculate recency (difference in days between the reference date and Last Purchase Date)
df['Recency'] = (reference_date - df['Last Purchase Date']).dt.days

# Check the updated data with Recency
df[['Customer ID', 'Last Purchase Date', 'Recency']].head()
```

---

### Task 3: Calculating Frequency

- **Description**: Frequency refers to how often a customer has made purchases. Customers who purchase more often tend to be more loyal.
  
- **Code Instructions**:
  1. Use the `Total Number of Purchases` column to calculate frequency.
  2. No transformation needed, just store it in the `Frequency` column.
  3. Ensure the column is clean and ready for further steps.

```python
# Frequency is already in the dataset, simply ensure it's clean
df['Frequency'] = df['Total Number of Purchases']

# Check Frequency column
df[['Customer ID', 'Frequency']].head()
```

---

### Task 4: Calculating Monetary Value

- **Description**: Monetary value measures how much a customer has spent. This helps identify the high-spending customers.
  
- **Code Instructions**:
  1. Use the `Total Spend (Monetary Value)` column.
  2. Ensure there are no missing or extreme outlier values.
  3. Store the result in a new column called `Monetary`.

```python
# Ensure no missing or outlier values in Monetary
df['Monetary'] = df['Total Spend (Monetary Value)']

# Check the Monetary column
df[['Customer ID', 'Monetary']].head()
```

---

### Task 5: Assigning RFM Scores

- **Description**: Assign scores for Recency, Frequency, and Monetary based on quartiles. This categorizes customers for segmentation.
  
- **Code Instructions**:
  1. Use quantiles to rank customers into tiers (1 to 4) for each R, F, and M.
  2. Assign scores based on the quartiles.
  3. Combine the scores into a single `RFM_Score`.

```python
# Assign scores based on quantiles (1 to 4) for each R, F, and M
df['R_Score'] = pd.qcut(df['Recency'], 4, labels=[4, 3, 2, 1])
df['F_Score'] = pd.qcut(df['Frequency'], 4, labels=[1, 2, 3, 4])
df['M_Score'] = pd.qcut(df['Monetary'], 4, labels=[1, 2, 3, 4])

# Create a combined RFM Score
df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)

# Check RFM scores
df[['Customer ID', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score']].head()
```

---

### Task 6: Segmenting Customers

- **Description**: Segment customers into groups like best customers, high spenders, and churned customers based on their RFM scores.
  
- **Code Instructions**:
  1. Define rules for each customer segment based on RFM scores.
  2. Assign customers to relevant segments.
  3. Store the segment labels in a new column `Customer_Segment`.

```python
# Segment customers based on RFM Score
#Can we add more segments like worst customer,  low spending customer? Couldnt see a segment for frequency. A segment for frequenct spenders and non frequent spenders?
def segment_customers(df):
    if df['RFM_Score'] == '444': #it needs to be 444 for the best customer? A best customer is someone who purchase frequently, more recent and high monetary value
        return 'Best Customer'
     elif df['RFM_Score'][1] == '4':  # For loyal, we can use frequency to determine loyalty
        return 'Loyal Customer' 
    elif df['RFM_Score'][0] == '1': # Churned customer- Recency in first quadrant
        return 'Churned Customer'
    elif df['RFM_Score'][2] == '4':
        return 'High-Spending Customer'
    else:
        return 'Other'

# Apply the segmentation function
df['Customer_Segment'] = df.apply(segment_customers, axis=1)

# Check segmented data
df[['Customer ID', 'RFM_Score', 'Customer_Segment']].head()
```

---

### Task 7: Counting Customer Segments

- **Description**: Counting the number of customers in each segment helps assess customer distribution and guides marketing strategies.
  
- **Code Instructions**:
  1. Group by the `Customer_Segment` column.
  2. Count the number of customers in each segment.
  3. Print the final counts for analysis.

```python
# Count the number of customers in each segment
segment_counts = df['Customer_Segment'].value_counts()

# Display the count of each customer segment
print(segment_counts)
```

---

### Outcome

By completing these tasks, the learner will be able to:
- Calculate Recency, Frequency, and Monetary value for each customer.
- Assign RFM scores and segment customers.
- Identify and count the number of best customers, high-spending customers, loyal customers, and churned customers for effective targeted marketing strategies.

### Additional Tasks to Create the Table in the Image:

### Task 8: Group Customers into Clusters

- **Description**: Based on RFM scores or segments, group customers into different tiers or clusters, such as Top Tier, High Tier, Mid High Tier, Mid Tier, and Low Tier. This step is crucial to differentiate customer behavior and categorize them into meaningful groups for further analysis.
  
- **Code Instructions**:
  1. Group customers into clusters based on their `RFM_Score` or segmentation rules.
  2. Create a new column `Cluster Name` that assigns each customer to a cluster.

```python
# Define clusters based on customer segment or RFM scores
def assign_cluster(df):
    # Top Tier: High Recency, High Frequency, High Monetary
    if df['RFM_Score'] == '444':
        return 'Top Tier Customers'
    # High Tier: Moderate to High Frequency, Moderate to High Monetary, High Recency
    elif df['RFM_Score'].startswith('4'):
        return 'High Tier Customers'
    # Mid High Tier: Moderate Recency
    elif df['RFM_Score'][0] == '3':
        return 'Mid High Tier Customers'
    # Mid Tier: At risk customers with moderate recency
    elif df['RFM_Score'][0] == '2':
        return 'Mid Tier Customers'
    # Low Tier: Churned or low-value customers
    else:
        return 'Low Tier Customers'

# Apply the function to assign clusters
df['Cluster Name'] = df.apply(assign_cluster, axis=1)

# Check assigned clusters
df[['Customer ID', 'Cluster Name', 'RFM_Score']].head()
```

---

### Task 9: Calculate Cluster Size

- **Description**: For each cluster, calculate the total number of customers. Cluster size helps understand how large each group is in comparison to others.
  
- **Code Instructions**:
  1. Group by `Cluster Name`.
  2. Count the number of customers in each cluster.

```python
# Calculate the size of each cluster
cluster_size = df.groupby('Cluster Name')['Customer ID'].count().reset_index()
cluster_size.columns = ['Cluster Name', 'Cluster Size']

# Check the cluster size
print(cluster_size)
```

---

### Task 10: Calculate Total Order Value per Cluster

- **Description**: Calculate the total monetary value (order value) for each customer cluster. This helps in understanding the revenue contribution of each group.
  
- **Code Instructions**:
  1. Group by `Cluster Name`.
  2. Sum the `Monetary` column to calculate total order value for each cluster.

```python
# Calculate the total order value (monetary value) per cluster
total_order_value = df.groupby('Cluster Name')['Monetary'].sum().reset_index()
total_order_value.columns = ['Cluster Name', 'Total Order Value']

# Check the total order value per cluster
print(total_order_value)
```

---

### Task 11: Calculate the Number of Order Days per Cluster

- **Description**: For each cluster, calculate the average number of order days. This gives insights into how frequently customers in each cluster place orders.
  
- **Code Instructions**:
  1. Group by `Cluster Name`.
  2. Calculate the average number of order days using the `Frequency` column.

```python
# Calculate the number of order days per cluster (average frequency)
order_days = df.groupby('Cluster Name')['Frequency'].mean().reset_index()
order_days.columns = ['Cluster Name', 'Number of Order Days']

# Check the average number of order days per cluster
print(order_days)
```

---

### Task 12: Calculate Days Since Last Order (Recency) per Cluster

- **Description**: Calculate the average recency for each cluster. This step helps analyze how long ago customers in each cluster made their last purchase.
  
- **Code Instructions**:
  1. Group by `Cluster Name`.
  2. Calculate the average recency using the `Recency` column.

```python
# Calculate days since the last order (average recency) per cluster
days_since = df.groupby('Cluster Name')['Recency'].mean().reset_index()
days_since.columns = ['Cluster Name', 'Days Since']

# Check the average recency per cluster
print(days_since)
```

---

### Task 13: Combine Results into Final Table

- **Description**: Combine the calculated metrics (cluster size, total order value, number of order days, and days since) into a final table. This is the step where all insights are consolidated.
  
- **Code Instructions**:
  1. Merge the previous results (cluster size, total order value, number of order days, days since) into one final DataFrame.
  2. Display the final table.

```python
# Merge all the calculated metrics into a final DataFrame
final_table = cluster_size.merge(total_order_value, on='Cluster Name')
final_table = final_table.merge(order_days, on='Cluster Name')
final_table = final_table.merge(days_since, on='Cluster Name')

# Display the final table
print(final_table)
```

---

### Task 14: Visualize the Table (Optional)

- **Description**: Create a visual representation of the table for better understanding and presentation. This can help in making the data more comprehensible.
  
- **Code Instructions**:
  1. Use a library like `matplotlib` or `seaborn` to visualize the table.
  2. Create a bar chart or heatmap to visualize the cluster size, order value, and other metrics.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize cluster size vs total order value
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster Name', y='Total Order Value', data=final_table)
plt.title('Total Order Value per Cluster')
plt.xticks(rotation=45)
plt.show()
```

By completing these tasks, you will be able to generate a table similar to the one in the image, showing the cluster size, total order value, number of order days, and days since last purchase for each customer cluster.
