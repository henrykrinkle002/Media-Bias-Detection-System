# NEXIS DATABASE

## Usage

1. **Clone the repo**
2. **Add your Nexis Uni login credentials** to a `.env` file:

   ```
   Email=your_username
   PASSWORD=your_password
   ```
3. **Install dependencies** from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```
4. **Run** the scraper:

   ```bash
   python selena.py
   ```
5. **Output**:
   Article metadata is saved to a CSV file (`nexis_articles2.csv`) for downstream use in bias detection tasks.

---

## Features

* **Automated Login**
  Credentials are securely loaded from a `.env` file using `dotenv`. Requires `Email` and `PASSWORD` keys.

* **Query Batching**
  Dynamically generates logical AND queries by combining subsets of `event_keywords` and `actor_keywords`:

  ```
  ("War crimes" OR "IDPs") AND ("ENDF" OR "Abiy Ahmed")
  ```

* **Fully Automated Interaction**
  Handles JavaScript-heavy elements. Includes:

  * Page load spinners
  * Safe click logic with retries
  * Pagination navigation
  * Dynamic content handling

* **Rich Metadata Extraction**
  Extracted fields include:

  * `title`
  * `author`
  * `source`
  * `publication date`
  * `article URL`
  * `search query used`

* **Incremental CSV Storage**
  Scraped articles are written incrementally to `nexis_articles2.csv`.

* **Batch Logging**
  Logs per-batch statistics to `batch_report.log`, including:

  * Search query used
  * Number of articles collected
  * Warnings for empty or low-yield queries
