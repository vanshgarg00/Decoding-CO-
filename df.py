import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import missingno as msno
from scipy import stats
from scipy.stats import chi2_contingency

# Load the data
df = pd.read_csv('C:/Users/Vansh Garg/Downloads/Co2EmminsionData.csv')

# Data cleaning
numeric_cols = ['population', 'gdp', 'co2', 'co2_per_capita', 'coal_co2', 'gas_co2', 'oil_co2', 
                'cement_co2', 'flaring_co2', 'methane', 'nitrous_oxide']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

emission_cols = ['co2', 'coal_co2', 'gas_co2', 'oil_co2', 'cement_co2', 'flaring_co2']
df[emission_cols] = df[emission_cols].fillna(0)

# Create a decade column
df['decade'] = (df['year'] // 10) * 10
recent_df = df[df['year'] >= 1950].copy()
recent_df['co2_per_gdp'] = recent_df['co2'] / recent_df['gdp']

print("\nDescriptive statistics for CO2 emissions (1950-2021):")
print(recent_df['co2'].describe())

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Global CO2 Emissions Over Time
plt.figure(figsize=(14, 7))
global_co2 = df.groupby('year')['co2'].sum().reset_index()
sns.lineplot(data=global_co2, x='year', y='co2')
plt.title('Global CO2 Emissions Over Time', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('CO2 Emissions (million tonnes)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axvline(2008, linestyle='--', color='red')
plt.text(2008, global_co2['co2'].max() * 0.7, '2008 Financial Crisis', rotation=90, color='red')
plt.axvline(2020, linestyle='--', color='blue')
plt.text(2020, global_co2['co2'].max() * 0.5, 'COVID-19 Dip', rotation=90, color='blue')
plt.show()

# 2. CO2 Emissions by Country (Top 10)
plt.figure(figsize=(14, 7))
top_countries = recent_df.groupby('country')['co2'].sum().nlargest(10).reset_index()
sns.barplot(data=top_countries, x='co2', y='country', hue='country', palette='viridis', legend=False)
plt.title('Top 10 Countries by Total CO2 Emissions (1950-2021)', fontsize=16)
plt.xlabel('Total CO2 Emissions (million tonnes)', fontsize=14)
plt.ylabel('Country', fontsize=14)
plt.show()

# Line plot for top 10 countries
top_countries_list = top_countries['country'].tolist()
top_df = recent_df[recent_df['country'].isin(top_countries_list)]
plt.figure(figsize=(14, 7))
sns.lineplot(data=top_df, x='year', y='co2', hue='country', palette='tab10')
plt.title('CO2 Emissions Over Time for Top 10 Countries', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('CO2 Emissions (million tonnes)', fontsize=14)
plt.show()

# 3. CO2 Emissions per Capita Over Time
plt.figure(figsize=(14, 7))
sns.lineplot(data=recent_df, x='year', y='co2_per_capita', hue='country', 
             estimator='mean', errorbar=None, palette='tab20', legend=False)
plt.title('Average CO2 Emissions per Capita Over Time (1950-2021)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('CO2 per Capita (tonnes)', fontsize=14)
plt.show()

# 4. CO2 Sources Breakdown (Pie Chart)
plt.figure(figsize=(10, 10))
sources = recent_df[['coal_co2', 'oil_co2', 'gas_co2', 'cement_co2', 'flaring_co2']].sum()
sources = sources[sources > 0]
plt.pie(sources, labels=sources.index, autopct='%1.1f%%', startangle=90, 
        colors=sns.color_palette('pastel'))
plt.title('Global CO2 Emissions by Source (1950-2021)', fontsize=16)
plt.show()

# 5. CO2 vs GDP (Scatter Plot)
plt.figure(figsize=(14, 7))
sns.scatterplot(data=recent_df, x='gdp', y='co2', hue='country', alpha=0.6, legend=False)
plt.xscale('log')
plt.yscale('log')
plt.title('CO2 Emissions vs GDP (Log Scale)', fontsize=16)
plt.xlabel('GDP (log scale)', fontsize=14)
plt.ylabel('CO2 Emissions (log scale)', fontsize=14)
plt.show()

# 6. Heatmap of Correlation Matrix
plt.figure(figsize=(12, 10))
corr_matrix = recent_df[['co2', 'co2_per_capita', 'gdp', 'population', 'coal_co2', 'oil_co2', 'gas_co2']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of CO2 Related Metrics', fontsize=16)
plt.show()

# 7. CO2 Emissions by Decade (Box Plot)
plt.figure(figsize=(14, 7))
sns.boxplot(data=recent_df, x='decade', y='co2', showfliers=False)
plt.title('Distribution of CO2 Emissions by Decade', fontsize=16)
plt.xlabel('Decade', fontsize=14)
plt.ylabel('CO2 Emissions (million tonnes)', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# 8. CO2 Emissions Growth Rate
plt.figure(figsize=(14, 7))
growth = df.groupby('year')['co2'].sum().pct_change() * 100
growth.plot()
plt.title('Global CO2 Emissions Growth Rate (%)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Annual Growth Rate (%)', fontsize=14)
plt.axhline(0, color='red', linestyle='--')
plt.grid(True, alpha=0.3)
plt.show()

# 9. Stacked Area Chart of CO2 Sources Over Time
plt.figure(figsize=(14, 7))
sources_over_time = df.groupby('year')[['coal_co2', 'oil_co2', 'gas_co2', 'cement_co2']].sum()
sources_over_time.plot(kind='area', stacked=True, alpha=0.7)
plt.title('Global CO2 Emissions by Source Over Time', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('CO2 Emissions (million tonnes)', fontsize=14)
plt.legend(title='Source')
plt.show()

# 10. Population vs CO2 Emissions (Bubble Chart)
plt.figure(figsize=(14, 10))
sample_countries = recent_df[recent_df['year'] == 2020].nlargest(20, 'co2')
sample_countries['bubble_size'] = sample_countries['co2_per_capita'].fillna(0) + 0.1
sample_countries['safe_gdp'] = sample_countries['gdp'].fillna(1)
plt.scatter(x=sample_countries['population'], y=sample_countries['co2'], 
            s=sample_countries['bubble_size']*50, alpha=0.6,
            c=np.log(sample_countries['safe_gdp']), cmap='viridis')
for i, row in sample_countries.iterrows():
    plt.text(row['population'], row['co2'], row['country'], fontsize=9)
plt.colorbar(label='Log GDP')
plt.xscale('log')
plt.title('Population vs CO2 Emissions (Bubble Size = CO2 per Capita)', fontsize=16)
plt.xlabel('Population (log scale)', fontsize=14)
plt.ylabel('Total CO2 Emissions (million tonnes)', fontsize=14)
plt.show()

# ===========================
# ✨ STATISTICAL ANALYSIS ✨
# ===========================

# Pearson Correlation Test: CO2 vs GDP
co2_gdp_data = recent_df[['co2', 'gdp']].dropna()
co2_gdp_corr, co2_gdp_p = stats.pearsonr(co2_gdp_data['co2'], co2_gdp_data['gdp'])

print("\n--- Pearson Correlation ---")
print(f"Correlation Coefficient: {co2_gdp_corr:.4f}")
print(f"P-value: {co2_gdp_p:.4e}")
if co2_gdp_p < 0.05:
    print("=> Statistically significant correlation between CO2 and GDP.")
else:
    print("=> No significant correlation found.")

# T-Test Example: Compare CO2 per capita between two decades
decade_1980s = recent_df[recent_df['decade'] == 1980]['co2_per_capita'].dropna()
decade_2010s = recent_df[recent_df['decade'] == 2010]['co2_per_capita'].dropna()
ttest_stat, ttest_p = stats.ttest_ind(decade_1980s, decade_2010s)

print("\n--- Independent T-Test: 1980s vs 2010s CO2 per Capita ---")
print(f"T-statistic: {ttest_stat:.4f}")
print(f"P-value: {ttest_p:.4e}")
if ttest_p < 0.05:
    print("=> Significant difference in CO2 per capita between decades.")
else:
    print("=> No significant difference in CO2 per capita between decades.")

# Chi-Squared Test Example (Categorical dummy test)
recent_df['high_emitter'] = (recent_df['co2'] > recent_df['co2'].median()).astype(int)
recent_df['high_gdp'] = (recent_df['gdp'] > recent_df['gdp'].median()).astype(int)
contingency = pd.crosstab(recent_df['high_emitter'], recent_df['high_gdp'])
chi2, chi_p, _, _ = chi2_contingency(contingency)

print("\n--- Chi-Squared Test: High CO2 vs High GDP ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {chi_p:.4e}")
if chi_p < 0.05:
    print("=> Significant association between CO2 level and GDP level.")
else:
    print("=> No significant association found.")
