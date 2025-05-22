# 🎬 Movie Recommendation System

A movie recommendation system built using **Collaborative Filtering** and **Content-Based Filtering**, powered by **Flask** for a simple web interface. The system uses the popular [MovieLens dataset](https://grouplens.org/datasets/movielens/) to provide personalized movie suggestions.

---

##  Features

-  Recommend movies based on user input
-  Collaborative filtering using matrix factorization (SVD)
-  Content-based filtering using movie metadata
-  Web interface built with Flask
-  Model saved and loaded using `joblib`

---

##  Project Structure

movie-recommendation-system/


                            │ ├── main.py # Flask app

├── data/


                           │ ├── movies_metadata.csv # Movie metadata
                           │ └── credits.csv # User ratings

├── models/


                           │ └── recommendation_model.pkl # Trained model

├── requirements.txt


├── README.md



---

##  How It Works

- **Collaborative Filtering**: Recommends movies based on similar users' preferences using SVD (Singular Value Decomposition).
- **Content-Based Filtering**: Uses movie genres and descriptions to recommend similar movies.
- Combines both for a **hybrid recommendation** experience.

---

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system

2. **Install dependencies**

bash

pip install -r requirements.txt

3. **Download dataset**

Get the MovieLens dataset from here.

Place movies_metadata.csv and credits.csv inside the data/ folder.

4. **Run the Flask app**

python app/main.py

5. Visit http://127.0.0.1:5000 in your browser.

**Example**
Input:

"The Matrix"

Output Recommendations:

Inception

Interstellar

Minority Report

The Terminator

Blade Runner


**Future Improvements**
Add user login and profile-based recommendations

Integrate deep learning-based recommendation (e.g., using embeddings)

Deploy on Heroku or Render

**License**
This project is licensed under the MIT License.

Acknowledgements
MovieLens Dataset

Scikit-learn

Surprise library for recommender systems




