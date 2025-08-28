
# 🏎 F1 Grand Prix Predictor

This **Streamlit application** predicts Formula 1 Grand Prix race performance based on qualifying session data. It leverages **FastF1** for real-world race data, machine learning for modeling, and an interactive UI for quick experimentation.

##  Features

* **Data Integration**: Fetches and processes **historical F1 race data** directly from FastF1.
* **Predictive Modeling**: Trains a **linear regression model** on qualifying lap times to predict race results.
* **Simulation Mode**: Allows simulation of future qualifying data to forecast upcoming race outcomes.
* **Driver Ranking**: Provides predicted **finishing positions** and ranks drivers for a selected race.
* **Model Evaluation**: Includes performance metrics such as **Mean Absolute Error (MAE)** for transparency and improvement.
* **Interactive Dashboard**: Built with **Streamlit**, offering an intuitive interface to explore predictions without coding.

##  Tech Stack

* **Python** for data handling and modeling
* **FastF1** for motorsport data retrieval
* **Scikit-learn** for machine learning (Linear Regression)
* **Streamlit** for the web interface

## How to Run

1. Clone the repository:

   ```bash
   git clone <your-repo-link>
   cd f1-grand-prix-predictor
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```


##  Future Improvements

* Incorporating weather, pit stop strategies, and tire choices for more accurate predictions.
* Trying more advanced models (Random Forest, XGBoost, Neural Networks).
* Adding **visualizations** (lap time trends, performance comparisons).

