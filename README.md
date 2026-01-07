# Pipeline to identify travellers' needs from review data

## Introduction 

This repository contains the work carried out by Albin Morisseau and Emma Lhuillery as part of the Large Project in AI at the University of Klagenfurt between October 2025 and February 2026 under the supervision of Dr Markus Zanker

The aim of this project is to study the behaviors and the needs of travellers with specific needs (disabilities, pets, young children) who require special arrangements and conditions.

The following elements are available in this repository:

1. **End to end pipeline to extract reviews related to travellers' needs**

Will be detailled soon...

2. **Tools for topic modeling**

Will be detailled soon...

## Hardware

We ran the code using this laptop hardware:
- RTX4070 GPU / 8 Go RAM
- Intel Core I9 CPU / 32 Go RAM

Please modify the NUM_THREADS environment variable in the .env file to match your CPU's capabilities to avoid damaging your hardware.

NB: We strongly recommend using hardware or cloud services to significantly speed up calculation times.

## Project Structure
```
review-pipeline/
├── data/ # Input data, configuration files (categories, exclusions)
├── logs/ # Pipeline logs
├── models/bert_finetuned/ # Fine-tuned BERT model and tokenizer for validation
├── scripts/ # Standalone scripts (scraping, model dowloading)
├── src/ # Core pipeline modules 
├── tests/ # Unit and integration tests
├── .env # Environment variables
├── main.py # Main entry point for the end-to-end pipeline
├── README.md
└── requirements.txt
```

## Setup of the project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AlbinMorisseau/review-pipeline.git](https://github.com/AlbinMorisseau/review-pipeline.git)
    cd review-pipeline
    ```
2.  **Create and activate a virtual envrionment:**
    ```bash
    python -m venv venv
    # For Windows
    .\venv\Scripts\activate
    # For MacOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependancies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download fine tuned BERT model for validation:**
    ```bash
    python -m scripts/download_model.py
    ```
5.  **(Optional) Launch tests:**
    ```bash
    pytest tests/
    ```

## Potential datasets use

This project has been tested on a wide range of publicly available review datasets covering hotels, restaurants, airlines, activities, and social media.  
Below is the complete list of datasets:

- **Booking.com Accommodation Reviews**  
  https://huggingface.co/datasets/Booking-com/accommodation-reviews/tree/main

- **Yelp Open Dataset**  
  https://business.yelp.com/data/resources/open-dataset/

- **Hotel Reviews 1 / 2 / 3 (Datafiniti)**  
  https://www.kaggle.com/datasets/datafiniti/hotel-reviews

- **TripAdvisor Hotel Reviews**  
  https://www.kaggle.com/datasets/joebeachcapital/hotel-reviews?select=reviews.csv

- **Twitter Reviews Dataset**  
  https://www.kaggle.com/datasets/goyaladi/twitter-dataset?select=twitter_dataset.csv

- **Airline Reviews (Dataset 1)**  
  https://www.kaggle.com/datasets/chaudharyanshul/airline-reviews

- **Airline Reviews (Dataset 2)**  
  https://www.kaggle.com/datasets/sujalsuthar/airlines-reviews

- **European Hotel Reviews (515k)**  
  https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe

- **Restaurant Reviews (Dataset 1)**  
  https://www.kaggle.com/datasets/d4rklucif3r/restaurant-reviews

- **Restaurant Reviews (Dataset 2)**  
  https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews

- **Activities Reviews**  
  https://www.kaggle.com/datasets/johnwdata/reviewsactivities

- **European Restaurant Reviews**  
  https://www.kaggle.com/datasets/gorororororo23/european-restaurant-reviews


- **AccessibleGo**
    We also provide scripts that enable ethical scraping of the community section of the accessiblego website, allowing qualitative reviews to be obtained on the needs of people with any kind of disability.

    ```bash
    python -m scripts.scrapping_accessiblego
    ```

    **WARNING:** Ensure that the robots.txt file has not been modified in the meantime and that scraping this site is still permitted. Please follow best practices by clearly identifying your intention and not overloading the site's internal APIs.