# Fortnite_Data_Analysis
Foad_Jawabreh_12113427
Game Analytics Report: Player Behavior Analysis Using Python
Author: Fouad
________________________________________
Executive Summary
This report presents a data-driven analysis of player behavior in a gaming dataset inspired by Fortnite and similar titles. By leveraging Python tools such as pandas, seaborn, matplotlib, scikit-learn, and SciPy, we explored patterns in gameplay metrics, performed clustering and regression modeling, and produced visualizations to inform game design decisions. Key findings include observable player types based on eliminations and movement, a modest correlation between hits and distance traveled, and regression-based predictions of damage taken. These insights guide recommendations for player feedback systems and gameplay balancing.
________________________________________
Introduction
Problem Statement
Understanding how players behave in a game is critical for designing balanced, enjoyable gameplay. This project analyzes session-based gameplay data to identify patterns, cluster similar sessions, and build predictive models.
Project Goals
•	Clean and analyze raw gameplay data
•	Visualize key behavioral trends
•	Cluster sessions to identify player types
•	Predict gameplay outcomes using machine learning
•	Recommend data-informed design changes
________________________________________
Methodology
Game Design Choices and Rationale
The dataset reflects a game similar to Fortnite, capturing metrics such as eliminations, hits, distance traveled, materials used, and damage taken. These features were chosen for their relevance to player activity, decision-making, and performance.
Data Collection Approach
The dataset is assumed to have been collected using in-game telemetry tools, tracking each session with metrics stored in a CSV file. Features represent a mix of player actions (hits, materials used), outcomes (eliminations, damage), and context (distance traveled).
Analytics Dashboard Implementation
We used the following Python libraries:
•	pandas: Load and manipulate data tables
•	numpy: Numerical functions and array operations
•	seaborn + matplotlib: Graphical visualization
•	scikit-learn: Machine learning (KMeans, regression)
•	scipy: Statistical functions (z-score)
Code to load data:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('fortnite.csv')
df.head()
Machine Learning Algorithm Selection and Implementation
We applied:
•	KMeans Clustering: Group similar sessions by eliminations, hits, and distance
•	Linear Regression: Predict damage taken using hits, materials used, and movement
Key implementation snippet:
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

features = df[['Eliminations', 'Hits', 'Distance Traveled']]
model = KMeans(n_clusters=3)
df['Cluster'] = model.fit_predict(features)
________________________________________
Results and Analysis
Key Patterns in Player Behavior
•	Players with high eliminations often showed high mobility and high hit counts.
•	A scatter plot of Hits vs Distance Traveled showed moderate correlation — players who moved more tended to engage more.
Machine Learning Insights and Accuracy
•	Clustering revealed 3 behavioral groups: aggressive movers, passive campers, and balanced players.
•	Regression model predicting damage taken achieved reasonable accuracy, showing that players who moved more and used more materials tended to take more damage.
Visualizations with Explanations
•	Histogram of Eliminations: Revealed most players get between 5–15 kills.
sns.histplot(df['Eliminations'], bins=10)
•	Scatter Plot (Hits vs Distance Traveled): Displayed engagement patterns.
sns.scatterplot(x='Hits', y='Distance Traveled', data=df)
•	Cluster Plot (Eliminations vs Hits):
sns.scatterplot(x='Eliminations', y='Hits', hue='Cluster', data=df)
•	Regression Modeling:
X = df[['Hits', 'Materials Used', 'Distance Traveled']]
y = df['Damage Taken']
model = LinearRegression()
model.fit(X, y)
Design Recommendations
•	Introduce feedback systems for high-damage low-elimination players.
•	Consider rebalancing material costs or effectiveness.
•	Create achievements tied to player movement + engagements.
________________________________________
Technical Challenges
•	Some columns like Accuracy were inconsistent or redundant.
•	Choosing meaningful feature pairs (e.g., Hits vs Distance) required trial and error.
•	Normalizing data for clustering was essential to avoid bias toward large-scale values.
________________________________________
Future Improvements
•	Collect per-minute statistics for finer analysis.
•	Add metadata such as match type or map region.
•	Train classification models (e.g., Random Forest) for more robust prediction.
•	Add UI-based dashboard with interactivity (e.g., using Streamlit).
________________________________________
Conclusion
This project demonstrated the power of Python-based data analysis in uncovering actionable player insights. By combining visualizations, statistical analysis, clustering, and prediction, we generated a foundational analytics dashboard and several design recommendations. This method can easily be extended to live-service games with evolving telemetry.
________________________________________
References
•	pandas documentation
•	seaborn visualization guide
•	scikit-learn user guide
•	Fortnite gameplay concepts

