import pandas as pd
import os

def load_and_clean_data(file_path):
    print("Loading dataset...")

    df = pd.read_csv(file_path)

    df = df.dropna(subset=['title', 'synopsis'])
    
    df['rich_text'] = "Movie Title: " + df['title'] + "\nPlot: " + df['synopsis']

    texts = df['rich_text'].tolist()
    metadatas = [{"title": title} for title in df['title'].tolist()]
    
    print(f"Successfully loaded and cleaned {len(texts)} movies!")
    return texts, metadatas

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "data", "movie_synopsis.csv")
    
    movie_texts, movie_metadata = load_and_clean_data(file_path)
    
    print("\n--- Sample Output ---")
    print(movie_texts[0])