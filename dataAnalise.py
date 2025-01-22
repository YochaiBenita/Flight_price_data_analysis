import csv
import os


import duckdb
from pandas.io.common import file_exists

# creating small csv example
if not file_exists("itineraries_500.csv"):
    with (
        open("itineraries.csv", "r", newline="") as infile,
        open("itineraries_500.csv", "w", newline="") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write header row
        header = next(reader)
        writer.writerow(header)

        # Write specified number of data rows
        for i, row in enumerate(reader):
            if i >= 500:
                break
            writer.writerow(row)

# connecting to duckDB, creating table if needed
if not file_exists("database.duckdb"):
    duckdb_conn = duckdb.connect("database.duckdb")
    duckdb_conn.execute(
        "CREATE TABLE main AS SELECT * FROM read_csv_auto('itineraries.csv')"
    )
    print("duckDB file created")
else:
    duckdb_conn = duckdb.connect("database.duckdb")
    print("duckDB file connected")

# remove old tables
duckdb_conn.execute("DROP TABLE IF EXISTS sample")
duckdb_conn.execute("DROP TABLE IF EXISTS query1")
duckdb_conn.execute("DROP TABLE IF EXISTS query2")
duckdb_conn.execute("DROP TABLE IF EXISTS query3")
duckdb_conn.execute("DROP TABLE IF EXISTS query4")

# creating sample data
duckdb_conn.execute("CREATE TABLE sample AS SELECT * FROM main USING SAMPLE 500;")

# query 1
duckdb_conn.execute("""
    CREATE TABLE query1 AS
    WITH time_ranges AS (
        SELECT *,
               (flightDate - searchDate) as days_until_flight
        FROM main
    )
    SELECT 
        days_until_flight as days_before_flight,
        ROUND(AVG(totalFare), 2) as avg_fare,
        COUNT(*) as number_of_searches,
        ROUND(MIN(totalFare), 2) as min_fare,
        ROUND(MAX(totalFare), 2) as max_fare
    FROM time_ranges
    GROUP BY days_until_flight
    ORDER BY days_until_flight
""")

# query 2
duckdb_conn.execute("""
    CREATE TABLE query2 AS
    WITH flight_categories AS (
        SELECT *,
               CASE 
                   WHEN totalTravelDistance <= 500 THEN 1
                   WHEN totalTravelDistance <= 1000 THEN 2
                   WHEN totalTravelDistance <= 1500 THEN 3
                   WHEN totalTravelDistance <= 2000 THEN 4
                   ELSE 5
               END as distance_category
        FROM main
        WHERE totalTravelDistance IS NOT NULL
    )
    SELECT 
        CASE distance_category
            WHEN 1 THEN 'Short (0-500 miles)'
            WHEN 2 THEN 'Medium (501-1000 miles)'
            WHEN 3 THEN 'Long (1001-1500 miles)'
            WHEN 4 THEN 'Very Long (1501-2000 miles)'
            ELSE 'Ultra Long (2000+ miles)'
        END as flight_distance,
        isNonStop,
        ROUND(AVG(totalFare), 2) as avg_fare,
        ROUND(AVG(travelDuration), 0) as avg_duration_minutes,
        COUNT(*) as number_of_flights
    FROM flight_categories
    GROUP BY distance_category, isNonStop
    ORDER BY distance_category, isNonStop
""")

# query 3
duckdb_conn.execute("""
    CREATE TABLE query2 AS
    WITH flight_times AS (
        SELECT *,
               CAST(SUBSTR(segmentsDepartureTimeRaw, 12, 2) AS INTEGER) as departure_hour
        FROM main
        WHERE segmentsDepartureTimeRaw IS NOT NULL
    )
    SELECT 
        departure_hour,
        ROUND(AVG(totalFare), 2) as avg_fare,
        ROUND(AVG(seatsRemaining), 1) as avg_seats_remaining,
        COUNT(*) as number_of_flights,
        ROUND(MIN(totalFare), 2) as min_fare,
        ROUND(MAX(totalFare), 2) as max_fare,
        SUM(isNonStop::INTEGER) * 100.0 / COUNT(*) as nonstop_percentage
    FROM flight_times
    GROUP BY departure_hour
    ORDER BY departure_hour
""")

# query 4
duckdb_conn.execute("""
    CREATE TABLE query4 AS
    WITH day_info AS (
        SELECT 
            totalFare,
            TO_CHAR(flightDate, 'Day') as flight_day,
            isNonStop
        FROM main
    )
    SELECT 
        TRIM(flight_day) as flight_day,
        ROUND(AVG(totalFare), 2) as avg_fare,
        ROUND(AVG(CASE WHEN isNonStop THEN totalFare END), 2) as avg_nonstop_fare,
        ROUND(AVG(CASE WHEN NOT isNonStop THEN totalFare END), 2) as avg_connection_fare,
        COUNT(*) as number_of_flights
    FROM day_info
    GROUP BY flight_day
    ORDER BY EXTRACT(DOW FROM DATE_TRUNC('day', MIN(flightDate)))
""")

# query 5
duckdb_conn.execute("""
    CREATE TABLE query5 AS
    WITH price_volatility AS (
        SELECT 
            startingAirport,
            destinationAirport,
            flightDate,
            AVG(totalFare) as avg_fare,
            LAG(AVG(totalFare)) OVER (
                PARTITION BY startingAirport, destinationAirport
                ORDER BY flightDate
            ) as prev_day_fare
        FROM flights
        GROUP BY startingAirport, destinationAirport, flightDate
    )
    SELECT 
        flightDate,
        startingAirport,
        destinationAirport,
        avg_fare,
        prev_day_fare,
        ((avg_fare - prev_day_fare) / prev_day_fare * 100) as daily_change_percent
    FROM price_volatility
    WHERE ((avg_fare - prev_day_fare) / prev_day_fare * 100) > 20
    ORDER BY daily_change_percent DESC;
""")


duckdb_conn.close()
