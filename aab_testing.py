# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

hypothesis = pd.read_csv('/datasets/hypotheses_us.csv', sep=';')
orders = pd.read_csv('/datasets/orders_us.csv')
visits = pd.read_csv('/datasets/visits_us.csv')

"""Data Preprocessing"""

# Verify that the data type is correct for each dataframe

hypothesis.info()

"""There are no null values, and all columns have the correct data type."""

orders.info()

orders['date'] = pd.to_datetime(orders['date'])

"""There are no null values, the data type of column ['date'] was changed from object to datetime64."""

visits.info()

visits['date'] = pd.to_datetime(visits['date'])

"""There are no null values, the data type of column ['date'] was changed from object to datetime64."""

orders.groupby('group').agg({
    'visitorId':'count'
})

"""There are more transactions that entered group B (640), compared to those that entered group A (570). This is a difference of 14.9%. You have to verify how many unique users there are per group to understand this difference."""

orders.groupby('group').agg({
    'visitorId':'nunique'
})

"""There are 503 unique users from group A and 586 from group B. These are 83 more unique users in group B, to avoid bias when analyzing the data, it is important to do a random sampling of 503 users for group B"""

users_group_b = orders[orders['group'] == 'B']['visitorId'].unique()
np.random.seed(42)
sampled_users_b = np.random.choice(users_group_b, size=503, replace=False)

orders_b = orders[
    (orders['group'] == 'A') |
    ((orders['group'] == 'B') & (orders['visitorId'].isin(sampled_users_b)))
].copy()

orders_b

"""503 random users were extracted from group B and these were used to create a new DataFrame called **orders_b**, this in order to have a DataFrame with all the original data but with the correct size for group B"""

orders_b.groupby('group')['visitorId'].nunique()

"""Now we have the same number of unique users for each group.

# Priortthize hypotheses
"""

# Apply framework ICE
# ICE = (impact * confidence) / effort
# RICE = (reach * impact * confidence) / effort

ice_score = (hypothesis['Impact'] * hypothesis['Confidence']) / hypothesis['Effort']
rice_score = (hypothesis['Reach'] * hypothesis['Impact'] * hypothesis['Confidence'] / hypothesis['Effort'])

print(ice_score.sort_values(ascending=False))
print('')
print(rice_score.sort_values(ascending=False))

hypothesis['ICE_Score'] = ice_score
hypothesis['RICE_Score'] = rice_score

# Show how scores changes
sns.lineplot(data=hypothesis, x=hypothesis.index, y='ICE_Score', label='ICE Score', color='orange')
sns.lineplot(data=hypothesis, x=hypothesis.index, y='RICE_Score', label='RICE Score', color='purple')
plt.title('ICE and RICE Score Variation for each Hypothesis')
plt.xlabel('Hypothesis')
plt.ylabel('Score')
plt.grid(linestyle='--')
plt.savefig("figures/icerice_score.png", dpi=150)
plt.close()

"""If we take into account the scope (RICE) of the hypotheses. 
The priority hypotheses would be hypothesis 7 and hypothesis 2. If we did not take into account the scope (ICE), 
the best hypotheses would be 8 and 0. I consider that the best metric to prioritize hypotheses is the RICE score.
It is very important to take into account the scope of each of the hypotheses to avoid losing valuable resources in a change that
will only affect a very low percentage of users.

# A/B test analysis #

### Cumulative income by group
"""

# New df that sums revenue by date and group
daily_revenue = orders_b.groupby(['date', 'group'])['revenue'].sum().reset_index()

# New column with the cumulative sum of revenue
daily_revenue['cumulative_revenue'] = daily_revenue.groupby('group')['revenue'].cumsum()

daily_revenue.head()

palette = {'A':'red', 'B':'blue'}
sns.lineplot(data=daily_revenue, x='date', y='cumulative_revenue', hue='group', palette=palette)

plt.title('Cumulative Income Over Time by Group')
plt.xlabel('Date')
plt.ylabel('Cumulative Income')
plt.legend(title='Group')
plt.xticks(rotation=45)
plt.grid(linestyle='--')
plt.savefig("figures/cumulative_income_by_group.png", dpi=150)
plt.close()

"""When comparing the accumulated income by group, it is evident that group B had more accumulated income than group A.
But it is necessary to continue investigating to avoid reaching hasty conclusions.

### Cumulative average order size by group
"""

# New df that sums revenue and count transactions
daily_totals = orders_b.groupby(['date', 'group']).agg({
    'revenue': 'sum',
    'transactionId': 'count'
}).reset_index()

# New column with accumulated revenue
daily_totals['cumulative_revenue'] = daily_totals.groupby('group')['revenue'].cumsum()

# New column with accumulated orders
daily_totals['cumulative_orders'] = daily_totals.groupby('group')['transactionId'].cumsum()

# New column with average size of order.
## The values ​​are the quotient between accumulated profits and accumulated orders
daily_totals['cumulative_avg_order_size'] = daily_totals['cumulative_revenue'] / daily_totals['cumulative_orders']

# Renaming columns to avoid confussion

daily_totals.rename(columns={'transactionId': 'orders'}, inplace=True)

print('Top 5: Tamaño promedio acumulado por grupo')
print()
print(daily_totals[daily_totals['group'] == 'A'][['date', 'group', 'cumulative_avg_order_size']].sort_values(by='cumulative_avg_order_size', ascending=False).head())
print()
print(daily_totals[daily_totals['group'] == 'B'][['date', 'group', 'cumulative_avg_order_size']].sort_values(by='cumulative_avg_order_size', ascending=False).head())

# Graphic Representation
sns.lineplot(data=daily_totals, x='date', y='cumulative_avg_order_size', hue='group', palette=palette)
plt.title('Comparison of Average Cumulative Purchase Size by Group')
plt.ylabel('Price per Purchase Unity ($)')
plt.xlabel('Date')
plt.legend(title='Group')
plt.xticks(rotation=45)
plt.grid(linestyle='--')
plt.savefig("figures/avg_purchase_size_by_group.png", dpi=150)
plt.close()

"""The average cumulative purchase size for group B is noticeably higher than for group A. For much of the experiment it remained above. 
Group B had a peak of 171.68 per order unit on August 19. For group A, the best value was 118.22 per order unit, on August 13.
It is clear that on average each order for group B has a higher value than for group A. This may be because the products purchased by customers 
in group B are more expensive or they buy in greater quantities.

### Relative difference in cumulative average order size for Group B compared to Group A
"""

# Pivot table with the average cumulative size data for each group on each date
pivot_data = daily_totals.pivot(index='date', columns='group', values='cumulative_avg_order_size')

# Calculate the relative difference (B vs A)
pivot_data['relative_difference'] = ((pivot_data['B'] - pivot_data['A']) / pivot_data['A']) * 100

# Reset the index to be able to graph
pivot_data = pivot_data.reset_index()
pivot_data.head()

plt.figure(figsize=(12, 6))
sns.lineplot(data=pivot_data, x='date', y='relative_difference', color='green', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.title('Relative Difference in Average Cumulative Purchase Size (Group B vs Group A)')
plt.xlabel('Date')
plt.ylabel('Relative Difference (%)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.savefig("figures/rltv_difference_avg_cumulative_purhcase_size.png", dpi=150)
plt.close()

"""Although at the beginning group B starts a little weaker and has some intermediate ups and downs, 
in the second half of the experiment it surpasses group A in average order size (Up to 50% - 60% larger than A).
 At the end of the month, it remains constant above 30%.

### Conversion rate of each group as the relationship between orders and the number of visits each day"""

# New column in the 'visits' df that calculates accumulated visits
visits['cumulativeVisits'] = visits.groupby('group')['visits'].cumsum()

# New dataframe with accumulated data on the number of orders and visits
cumulativeData = daily_totals[['date', 'group', 'cumulative_orders']].merge(visits[['date', 'group', 'cumulativeVisits']], on=['date', 'group'], how='inner')
cumulativeData.rename(columns={'cumulative_orders':'cumulativeOrders'}, inplace=True)

# Conversion rate calculation
cumulativeData['conversionRate'] = cumulativeData['cumulativeOrders'] / cumulativeData['cumulativeVisits']

print('Top 5 conversion rate per group')
print()
print(cumulativeData[cumulativeData['group'] == 'A'][['date', 'group', 'conversionRate']].sort_values(by='conversionRate', ascending=False).head())
print()
print(cumulativeData[cumulativeData['group'] == 'B'][['date', 'group', 'conversionRate']].sort_values(by='conversionRate', ascending=False).head())

# Graphical Representation
sns.lineplot(data=cumulativeData, x='date', y='conversionRate', hue='group', palette=palette)
plt.title('Difference between Conversion Rates per Group')
plt.xlabel('Date')
plt.ylabel('Converstion Rate')
plt.legend(title='Group')
plt.xticks(rotation=45)
plt.grid(linestyle='--')
plt.savefig("figures/conversion_rates_per_group.png", dpi=150)
plt.close()

"""Group A began the experiment with a conversion rate of 3.3% but over time it dropped to 2.9%. Its peak was 3.6% on August 3. 
A big peak but with a quite notable decrease. On the other hand, group B began the experiment with a conversion rate of 2.
5% and over time this rate increased to 2.9%, the peak of group B was a conversion rate of 3.1% on August 10.

At the beginning and at the end of the test, group A surpassed group B, but when reaching the middle of the experiment, for at least 10 days, group B remained constantly above group A

### Scatter chart of number of orders per user
"""

# Create a df per group containing the number of orders per user
ordersByUsersA = (orders_b[orders_b['group'] == 'A']
                 .groupby('visitorId', as_index=False)
                 .agg({'transactionId':'nunique'})
                 )
ordersByUsersA.columns = ['visitorId', 'orders']

ordersByUsersB = (orders_b[orders_b['group'] == 'B']
                 .groupby('visitorId', as_index=False)
                 .agg({'transactionId':'nunique'})
                 )
ordersByUsersB.columns = ['visitorId', 'orders']


print('Top 5 users with more orders: Group A')
print(ordersByUsersA.sort_values(by='orders', ascending=False).head())
print()
print('Top 5 users with more orders: Group B')
print(ordersByUsersB.sort_values(by='orders', ascending=False).head())

# Concatenated table to be able to graph correctly
ordersByUsers = pd.concat([
    ordersByUsersA.assign(group='A'),
    ordersByUsersB.assign(group='B')
])
ordersByUsers = ordersByUsers.reset_index(drop=True)

# Graphical Representation
sns.scatterplot(data=ordersByUsers, x=ordersByUsers.index, y='orders', hue='group', palette=palette)
plt.title('Dispersion of the Number of Orders per User')
plt.xlabel('User (index)')
plt.ylabel('Number of Orders')
plt.grid(linestyle='--')
plt.savefig("figures/dispersion_number_orders_per_user.png", dpi=150)
plt.close()

"""Customers from both groups behave similarly, the vast majority of customers buy only once. There are a few customers who buy between 2 and 3 times. 
The distribution is similar but A has a slightly wider range (up to 7). These anomalies must be eliminated to avoid a distorted result.

### 95th and 99th percentiles of the number of orders per user When does data become an anomaly?
"""

print('Percentiles 95 y 99:')
print(np.percentile(ordersByUsers['orders'], [95, 99]))

"""According to the percentiles obtained (P95 = 2, P99 = 4) only 5% of users make 2 or more orders and only 1% of users make 4 or more orders. 
Therefore, we can consider the data of those users who make 2 or more orders as anomalous, since they belong to the 5% with the highest activity (outliers)

## Order Price Scatter Chart
"""

revenuePerUserA = (orders_b[orders_b['group'] == 'A']
                 .groupby('transactionId', as_index=False)
                 .agg({'revenue':'sum'})
                 )
revenuePerUserA.columns = ['transactionId', 'revenue']

revenuePerUserB = (orders_b[orders_b['group'] == 'B']
                 .groupby('transactionId', as_index=False)
                 .agg({'revenue':'sum'})
                 )
revenuePerUserB.columns = ['transactionId', 'revenue']

print('Top 5 orders with the highest price: Group A')
print(revenuePerUserA.sort_values(by='revenue', ascending=False).head())
print()
print('Top 5 orders with the highest price: Group B')
print(revenuePerUserB.sort_values(by='revenue', ascending=False).head())

# Concatenated table to be able to graph correctly
revenuePerUsers = pd.concat([
    revenuePerUserA.assign(group='A'),
    revenuePerUserB.assign(group='B')
])
revenuePerUsers = revenuePerUsers.reset_index(drop=True)

# Graphical Representation
sns.scatterplot(data=revenuePerUsers, x=revenuePerUsers.index, y='revenue', hue='group', palette=palette)
plt.title('Order Price Dispersion')
plt.xlabel('Order (index)')
plt.ylabel('Price ($)')
plt.legend(title='Group')
plt.grid(linestyle='--')
plt.savefig("figures/order_price_dispersion.png", dpi=150)
plt.close()

"""The clients of both groups behave in a similar way but in group B there is a single anomaly, an order is close to $20,000. 
This is definitely part of 1% of the clients.

## 95th and 99th percentiles of the price per order When does data become an anomaly?"""

print('95th and 99th percentiles:')
print(np.percentile(revenuePerUsers['revenue'], [95, 99]))

"""According to the percentiles obtained (P95 = 432 P99 = 905) only 5% of the orders exceed 432 and only 1% of the orders exceed 905. 
Therefore, we can consider the data of those orders that exceed 432 as anomalous, since they belong to the 5% with the highest activity (outliers).

## Statistical significance of the difference in conversion between the groups using **raw data**"""

# Total visits by group
visitorsA = visits[visits['group'] == 'A']['visits'].sum()
visitorsB = visits[visits['group'] == 'B']['visits'].sum()

print(f'Total visits group A: {visitorsA}')
print(f'Total visits group B: {visitorsB}')

# Create the samples for testing
sampleA = pd.concat([
    ordersByUsersA['orders'],
    pd.Series(
        0,
        index=np.arange(visitorsA - len(ordersByUsersA)),
        name='orders'
    )
]).reset_index(drop=True)

sampleB = pd.concat([
    ordersByUsersB['orders'],
    pd.Series(
        0,
        index=np.arange(visitorsB - len(ordersByUsersB)),
        name='orders'
    )
]).reset_index(drop=True)

# Mann-Whitney U Test
print(f'p_value: {p_value_mannwhitneyu(sampleA, sampleB):.4f}')

print(f"Conversion A: {mean(sampleA):.4f}")
print(f"Conversion B: {mean(sampleB):.4f}")
print(f'Relative difference: {relative_diff(mean(sampleA), mean(sampleB)):.4f}')

"""Although the conversion of group B was 2.2% lower than that of group A, the p value obtained (0.8777) is much higher than the significance level of 0.05.
Therefore, **the null hypothesis is not rejected**, which indicates that there is no significant statistical evidence to affirm that the conversion differs between both groups.
The difference observed between the groups is so small and so inconsistent that **it is very likely due to chance.**

## Statistical significance of the difference in average order size between the groups using the **raw data**"""

# Mann-Whitney U Test

revenueA = orders_b[orders_b['group'] == 'A']['revenue']
revenueB = orders_b[orders_b['group'] == 'B']['revenue']

# Creation of function that extracts the p_value of the Mann-Whitney U test
def p_value_mannwhitneyu(control_group, treatment_group):
    return st.mannwhitneyu(control_group, treatment_group)[1]

print(f'p_value: {p_value_mannwhitneyu(revenueA, revenueB):.4f}')

# Creation of function that calculates the average
## (this to avoid creating multiple variables that have the same name)

def mean(group):
    return(group.mean())

# Creation function that calculates the relative difference of average values
def relative_diff(control_group, treatment_group):
    return(treatment_group / control_group) - 1

print(f'Average size A: {mean(revenueA):.4f}')
print(f'Average size B: {mean(revenueB):.4f}')
print(f'Diferencia relativa (B vs A): {relative_diff(mean(meanA), mean(meanB)):.4f}')

"""The p value obtained (0.72) is much higher than the significance level α = 0.05, so **the null hypothesis cannot be rejected.**
This indicates that there is no statistical evidence to claim that the average order size differs between groups A and B when using the raw data.
Although group B shows a 30% larger average order size, **this difference is not statistically significant and may be due to chance.**

## Statistical significance of the difference in conversion between the groups using the **filtered data**
"""

# Filter buyers from group A
ordersByUsersA_filtered = ordersByUsersA[
    ~ordersByUsersA['visitorId'].isin(abnormalUsers)
]

# Filter buyers from group B
ordersByUsersB_filtered = ordersByUsersB[
    ~ordersByUsersB['visitorId'].isin(abnormalUsers)

]

# Filtered sample adding zeros for users without purchases
sampleAFiltered = pd.concat([
    ordersByUsersA_filtered['orders'],
    pd.Series(
        0,
        index=np.arange(visitorsA - len(ordersByUsersA_filtered)),
        name='orders'
    )
], axis=0)

sampleBFiltered = pd.concat([
    ordersByUsersB_filtered['orders'],
    pd.Series(
        0,
        index=np.arange(visitorsB - len(ordersByUsersB_filtered)),
        name='orders'
    )
], axis=0)

print(f"Filered p-value: {p_value_mannwhitneyu(sampleAFiltered, sampleBFiltered):.5f}")

print(f"Filtered Conversion A: {mean(sampleAFiltered):.4f}")
print(f"Filtered Conversion B: {mean(sampleBFiltered):.4f}")
print(f'Relative Difference {relative_diff(mean(sampleAFiltered), mean(sampleBFiltered)):.4f}')

"""After eliminating abnormal users (those with ≥3 orders or orders greater than $433), the conversion rate was recalculated for groups A and B 
and the MannWhitney U test was applied.

The filtered results show that:

Conversion A: 2.68%

Conversion B: 2.71%

Relative difference B vs A: +0.82%

p-value: 0.96391

Since the p-value is much higher than the significance level (α = 0.05), **the null hypothesis is not rejected.**
This means that there is no statistical evidence that the conversion differs between the groups,
even after eliminating anomalies.

The observed difference is extremely small and **is almost certainly due to chance and not the effect of variant B.**

## Statistical significance of the difference in average order size between the groups using the **filtered data**
"""

# Definition of abnormal data
## Transactions of more than $433 and more than 3 orders per user

# Create a series containing the abnormal data
usersWithExpensiveOrders = revenuePerUsers[revenuePerUsers['revenue'] >=  433]['transactionId']
usersWithManyOrders = ordersByUsers[ordersByUsers['orders'] >= 3]['visitorId']

abnormalUsers = abnormalUsers = pd.concat(
    [usersWithManyOrders, usersWithExpensiveOrders],
    axis=0
).drop_duplicates().sort_values()

print('Abnormal data users:', abnormalUsers.count())

# Create a series that does not contain the abnormal data
revenueA_filtered = orders_b[
    (orders_b['group'] == 'A') &
    (~orders_b['transactionId'].isin(abnormalUsers))
]['revenue']

revenueB_filtered = orders_b[
    (orders_b['group'] == 'B') &
    (~orders_b['transactionId'].isin(abnormalUsers))
]['revenue']

# P value
print(f'p_value: {p_value_mannwhitneyu(revenueA_filtered, revenueB_filtered):.4f}')

print(f"Average Size A: {mean(revenueA_filtered):.4f}")
print(f"Average Size B: {mean(revenueB_filtered):.4f}")
print(f'Relative Difference: {relative_diff(mean(revenueA_filtered), mean(revenueB_filtered)):.4f}')
"""After eliminating anomalous users (those with orders of $433 or more and with 3 or more orders), the average order size was recalculated for both groups and the Mann–Whitney U test was applied.

• Average order size (A): 83.27

• Average order size (B): 80.78

• Relative difference (B vs A): −2.99%

• p-value: 0.8708
The p-value is considerably greater than the significance threshold α = 0.05, so **the null hypothesis is not rejected.**
This indicates that, even after filtering out anomalies, **there is no statistical evidence to show that the average order size differs between groups A and B.**
**The observed difference (3%) may be completely due to chance and does not represent a real effect of the experiment.**

### Based on the statistical results (p-values consistently greater than 0.05 for both conversion and average order size,
with and without anomaly filtering) and in the absence of significant trends in the graphs, it is concluded that there is no statistical evidence that variant B produces a different effect than group A.
### Therefore, the recommendation is to stop the test and conclude that there is no difference between the groups.
Variant B does not offer improvements or disadvantages compared to the current version.
"""