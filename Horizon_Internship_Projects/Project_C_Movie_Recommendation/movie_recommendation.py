import pandas as pd

df = pd.read_csv("movies.csv")

def recommend_movies(genre, min_rating=7.5, limit=5):
    genre = genre.strip().lower()

    result = df[
        df["Genre"].str.lower().str.contains(genre, na=False)
        & (df["Rating"] >= min_rating)
    ].sort_values("Rating", ascending=False).head(limit)

    return result[["Title", "Genre", "Year", "Rating"]]

print("Available genres: Action, Drama, Comedy, Romance, Sci-Fi, Animation, Thriller, Horror")
user_genre = input("Enter a genre: ")

recommendations = recommend_movies(user_genre)

if recommendations.empty:
    print("\nNo matching movies found.")
else:
    print("\nRecommended Movies:")
    print(recommendations.to_string(index=False))
