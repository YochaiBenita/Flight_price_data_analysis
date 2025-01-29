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
duckdb_conn.execute("DROP TABLE IF EXISTS airportsLocation")
duckdb_conn.execute("DROP TABLE IF EXISTS query1")
duckdb_conn.execute("DROP TABLE IF EXISTS query2")
duckdb_conn.execute("DROP TABLE IF EXISTS query3")
duckdb_conn.execute("DROP TABLE IF EXISTS query4")
duckdb_conn.execute("DROP TABLE IF EXISTS query5")

# creating sample data
duckdb_conn.execute("CREATE TABLE sample AS SELECT * FROM main USING SAMPLE 500;")
duckdb_conn.execute("CREATE TABLE airportsLocation AS SELECT * FROM read_csv_auto('airportsLocation.csv')")

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
print("query 1 finished")

# query 2
duckdb_conn.execute("""
CREATE TABLE query2 AS
SELECT 
   CASE 
       WHEN totalTravelDistance < 500 THEN 'Short (<500 miles)'
       WHEN totalTravelDistance < 1000 THEN 'Medium (500-1000 miles)'
       ELSE 'Long (>1000 miles)'
   END as distance_category,
   ROUND(AVG(CASE WHEN isNonStop THEN totalFare END), 2) as direct_avg_fare,
   ROUND(AVG(CASE WHEN NOT isNonStop THEN totalFare END), 2) as connection_avg_fare,
   COUNT(CASE WHEN isNonStop THEN 1 END) as direct_flights_count,
   COUNT(CASE WHEN NOT isNonStop THEN 1 END) as connection_flights_count
FROM main
WHERE totalTravelDistance IS NOT NULL
GROUP BY 1
ORDER BY distance_category;
""")
print("query 2 finished")

# query 3
duckdb_conn.execute("""
    CREATE TABLE query3 AS
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
print("query 3 finished")

duckdb_conn.execute("""
   CREATE TABLE query4 AS
   WITH day_info AS (
       SELECT 
           totalFare,
           STRFTIME(flightDate::DATE, '%A') as flight_day,
           isNonStop
       FROM main
   )
   SELECT 
       flight_day,
       ROUND(AVG(totalFare), 2) as avg_fare,
       ROUND(AVG(CASE WHEN isNonStop THEN totalFare END), 2) as avg_nonstop_fare,
       ROUND(AVG(CASE WHEN NOT isNonStop THEN totalFare END), 2) as avg_connection_fare,
       COUNT(*) as number_of_flights
   FROM day_info
   GROUP BY flight_day
   ORDER BY CASE flight_day
       WHEN 'Sunday' THEN 0
       WHEN 'Monday' THEN 1 
       WHEN 'Tuesday' THEN 2
       WHEN 'Wednesday' THEN 3
       WHEN 'Thursday' THEN 4
       WHEN 'Friday' THEN 5
       WHEN 'Saturday' THEN 6
   END
""")
print("query 4 finished")

# query 5
duckdb_conn.execute("""
    CREATE TABLE query5 AS
    SELECT 
        f.startingAirport,
        f.destinationAirport,
        f.isNonStop,
        COUNT(*) as num_flights,
        ROUND(AVG(totalFare), 2) as avg_fare,
        ROUND(MIN(totalFare), 2) as min_fare,
        ROUND(MAX(totalFare), 2) as max_fare,
        a1.lat as start_lat,
        a1.lon as start_lon,
        a2.lat as dest_lat,
        a2.lon as dest_lon
    FROM main f
    JOIN airportsLocation a1 ON f.startingAirport = a1.airport
    JOIN airportsLocation a2 ON f.destinationAirport = a2.airport
    GROUP BY f.startingAirport, f.destinationAirport, f.isNonStop, 
             a1.lat, a1.lon, a2.lat, a2.lon
    ORDER BY f.startingAirport, f.destinationAirport, f.isNonStop DESC;
""")
print("query 5 finished")

try:
    os.remove("database.sqlite")
    print("old sqlite file deleted")
except:
    print("cannot delete old sqlite file")

duckdb_conn.execute("INSTALL sqlite;")
duckdb_conn.execute("LOAD sqlite;")
duckdb_conn.execute("ATTACH 'database.sqlite' AS sqliteDB (TYPE SQLITE);")

duckdb_conn.execute("CREATE TABLE sqliteDB.sample AS SELECT * FROM sample")
print("Table 'sample' copied successfully")
duckdb_conn.execute("CREATE TABLE sqliteDB.airportsLocation AS SELECT * FROM airportsLocation")
print("Table 'airportsLocation' copied successfully")
for i in range(1, 6):
    duckdb_conn.execute(f"CREATE TABLE sqliteDB.query{i} AS SELECT * FROM query{i}")
    print(f"Table 'query{i}' copied successfully")

# duckdb_conn.execute("EXPORT DATABASE 'sqliteDB' (FORMAT SQLITE);")

duckdb_conn.close()


'''
def first_query():
    st.title("✈️ Flight Price and Search Analysis Dashboard")

    conn = sqlite3.connect("database.sqlite")
    # Load the 'sample' table into a pandas DataFrame
    query = "SELECT * FROM query1"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Price vs Days Before Flight
    st.subheader("🎟️ Ticket Price vs. Days Before Flight")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x='days_before_flight', y='avg_fare', alpha=0.3, ax=ax1)
    ax1.set_title('Ticket Price vs. Days Before Flight')
    ax1.set_xlabel('Days Before Flight')
    ax1.set_ylabel('Average Fare (USD)')
    ax1.invert_xaxis()
    st.pyplot(fig1)

    # Searches Over Time
    st.subheader("📊 Number of Searches Over Time")
    # Aggregate searches per day
    search_counts = df.groupby('days_before_flight')['number_of_searches'].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=search_counts, x='days_before_flight', y='number_of_searches', ax=ax2)
    ax2.set_title('Number of Searches Over Time')
    ax2.set_xlabel('Days Before Flight')
    ax2.set_ylabel('Number of Searches')
    ax2.invert_xaxis()
    st.pyplot(fig2)

    # _________________________________________________________________________________________________________

'''