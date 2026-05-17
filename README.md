# Credit Card Fraud Detection: Real-World Imbalance Handling

## What is this project?
I built this project to understand how Machine Learning behaves when data is extremely skewed. It wasn't just about writing code; the real goal was to figure out how to catch ~490 fraud cases hidden among 2.8 lakh normal transactions without crashing the system or missing the critical ones.

## The Realization: Accuracy is a Trap
Initially, seeing 99.8% accuracy felt like a win. But I quickly realized that accuracy is the biggest trap in imbalanced datasets. If a model predicts "Normal" for every single transaction, it still gets 99.8% accuracy but fails to catch a single fraud.

Because of this, I ignored accuracy and focused entirely on **Recall**. For me, a false alarm (False Positive) is manageable, but a missed fraud (False Negative) is a failure.

## Visualization Strategy: The Log Scale Logic
When I first plotted the class distribution, the fraud bar was practically invisible because the mountain of normal transactions was too high. To solve this, I used a **Logarithmic Scale** (`plt.yscale('log')`). This simple shift made the minority class visible and helped me visually grasp the scale of the imbalance I was dealing with.

## The 0.35 Threshold: A Manual Engineering Decision
Standard models use a 0.5 probability threshold. However, in fraud detection, missing a fraud is much costlier for a bank than a slight delay due to a false alarm. I manually tuned the classification threshold to **0.35**. This decision significantly boosted my model’s Recall, ensuring that it flags more potential frauds that a standard model would have missed.

## Overcoming Hardware Constraints
Training a Random Forest model on 300,000 rows with hyperparameter tuning on an 8GB RAM system was a major hurdle. I faced multiple session crashes due to memory limits. This taught me to be efficient with data types, scaling, and memory management, ensuring this pipeline can run smoothly on standard consumer-grade laptops.

## Tech Stack
* **Language:** Python (Pandas, NumPy)
* **Machine Learning:** Scikit-Learn (Random Forest, Preprocessing)
* **Visualization:** Matplotlib, Seaborn
* **Model Export:** Joblib

## About Me
I am a BCA student currently mastering the core logic behind AI and Machine Learning. I don't believe in "Black Box" solutions; I want to understand the "Why" behind every algorithm. My focus is on building efficient, data-driven solutions that solve real problems under practical hardware constraints.

## Dataset Source
The data used in this project can be found here: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
