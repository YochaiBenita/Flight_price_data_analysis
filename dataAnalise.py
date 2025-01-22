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
duckdb_conn.execute()

# query 2
duckdb_conn.execute()

# query 3
duckdb_conn.execute()

# query 4
duckdb_conn.execute()



duckdb_conn.close()
