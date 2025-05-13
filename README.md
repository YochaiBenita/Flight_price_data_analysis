# Flight Data Analysis Dashboard

## Project Overview
This project is a comprehensive flight data analysis system that includes a data analysis component and an interactive dashboard built with Streamlit. The project processes and analyzes flight itineraries data, providing insights and visualizations through an interactive web interface.

## Authors
- Yochai Benita
- Benjamin Rosin

## Features
- Interactive dashboard for flight data visualization
- Data analysis and processing capabilities
- Geographic visualization of airport locations
- Database integration with DuckDB and SQLite
- Custom data analysis scripts

## Project Structure
```
├── dashboard.py          # Main dashboard application
├── dataAnalise.py        # Data analysis scripts
├── database.duckdb       # DuckDB database file
├── database.sqlite       # SQLite database file
├── requirements.txt      # Project dependencies
├── airportsLocation.csv  # Airport location data
└── itineraries_500.csv   # Sample flight itineraries data
```

## Requirements
The project requires the following Python packages:
- streamlit
- duckdb
- pandas
- matplotlib
- seaborn
- pydeck
- numpy

## Installation
1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the dashboard:
```bash
streamlit run ./dashboard.py
```

## Data Files
- `itineraries_500.csv`: Sample dataset containing 500 rows of flight itineraries
- `airportsLocation.csv`: Contains geographical coordinates (latitude and longitude) for airports in the dataset
- `database.sqlite`: Processed data after running queries
- `database.duckdb`: Main database file containing the complete dataset

## Documentation
Detailed project documentation is available in the following files:
- Project report (דוח פרויקט.pdf)
- Screenshots and additional documentation are included in the repository

## License
This project is for educational purposes as part of the Big Data course.

---
*Note: This project contains both English and Hebrew files. Some documentation files are in Hebrew.* 

<!--
מגישים:
יוחאי בניטה - 322636036
בנימין רוסין - 211426598

פירוט קבצים: 
README.txt
itineraries_500.csv - קובץ 500 שורות ראשונות בדאטה, בתור דוגמא למבנה הדאטה
airportsLocation.csv - קובץ המכיל מיקום (קווי אורך ורוחב) של כל נמל תעופה בדאטה, בתור דוגמא למבנה הדאטה
dashboard.py - קובץ קוד של הדשבורד
dataAnalise.py - קובץ קוד של חקר המידע
database.sqlite - קובץ הדאטה לאחר הרצת השאילתות
requirements.txt - קובץ דרישות מערכת
דוח פרויקט.pdf - קובץ תיעוד של מהלך הפרוייקט

הרצה של הדשבורד: streamlit run ./dashboard.py
