# 📚 Semantic Book Recommender

This project is a **semantic book recommendation system** powered by **OpenAI embeddings** and **emotion analysis**. It allows users to search for books based on descriptions, categories, and emotional tones using a sleek **Gradio dashboard**.

## 🚀 Features

- **Semantic Search**: Find books based on the meaning of your query.
- **Category Filtering**: Narrow down results by genre or category.
- **Emotion-based Recommendations**: Get book suggestions aligned with specific emotional tones (e.g., Happy, Sad, Suspenseful).
- **Interactive Dashboard**: User-friendly interface built with **Gradio** for seamless interaction.

---

## 🛠️ Project Structure

```
.
├── books_cleaned.csv            # Original dataset with book information
├── books_with_categories.csv    # Dataset after category classification
├── books_with_emotions.csv      # Dataset after emotion analysis
├── tagged_description.txt       # Text file of book descriptions for vector search
├── vector-search.py             # Script for creating vector embeddings and search functionality
├── text-classification.py       # Script for adding categories to books
├── sentiment-analysis.py        # Script for analyzing emotional tones in book descriptions
├── dashboard.py                 # Gradio dashboard for user interaction
└── README.md                    # Project documentation
```

---

## 📦 Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-repo/semantic-book-recommender.git
   cd semantic-book-recommender
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key:**

   - Create a `.env` file in the root directory and add your OpenAI API key:
   
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

---

## 🔄 Data Processing Workflow

### 1. **Vector Search (Embedding Book Descriptions)**

   **File:** `vector-search.py`

   - Loads book descriptions from `tagged_description.txt`.
   - Splits the descriptions into smaller chunks.
   - Creates vector embeddings using **OpenAI** models.
   - Stores vectors in a **Chroma** vector database for semantic searching.

   **Run:**
   ```bash
   python vector-search.py
   ```

---

### 2. **Text Classification (Categorizing Books)**

   **File:** `text-classification.py`

   - Uses NLP models to classify books into categories (e.g., Fiction, Nonfiction).
   - Saves the updated dataset as `books_with_categories.csv`.

   **Run:**
   ```bash
   python text-classification.py
   ```

---

### 3. **Sentiment Analysis (Emotion Scoring for Books)**

   **File:** `sentiment-analysis.py`

   - Analyzes book descriptions to detect emotions like **joy**, **fear**, **sadness**, etc.
   - Stores the maximum detected emotion scores for each book in `books_with_emotions.csv`.

   **Run:**
   ```bash
   python sentiment-analysis.py
   ```

---

## 🎨 Launch the Gradio Dashboard

**File:** `dashboard.py`

- Provides an interactive interface to search for books based on descriptions, categories, and emotional tones.
- Displays book recommendations with thumbnails and short descriptions.

**Run:**
```bash
python dashboard.py
```

---

## 📋 Usage Instructions

1. **Enter a Description:**
   - Type a brief description of the kind of book you're looking for.
   - Example: *"A book about adventure and self-discovery."*

2. **Select a Category (Optional):**
   - Choose a specific category like *Fiction*, *Biography*, etc., or leave it as **All**.

3. **Choose an Emotional Tone (Optional):**
   - Pick an emotional tone such as *Happy*, *Sad*, *Suspenseful*, etc., to refine your recommendations.

4. **View Recommendations:**
   - Click **Find recommendations** to see book suggestions with covers and descriptions.

---

## 🧩 Dependencies

- **LangChain**: For vector embeddings and document processing.
- **OpenAI**: To generate semantic embeddings.
- **Chroma**: Vector store for similarity search.
- **Gradio**: To build the interactive dashboard.
- **Pandas & NumPy**: For data manipulation.
- **dotenv**: To manage environment variables.