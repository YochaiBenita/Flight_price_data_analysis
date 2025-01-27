import streamlit as st
import sqlite3
import pandas as pd
from pygments import highlight
import matplotlib.pyplot as plt
import seaborn as sns

sqlite_conn = sqlite3.connect("database.sqlite")

# st.write("# Hi")


def mainPage():
    st.write("# Welcome to Streamlit! 👋")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what Streamlit can do!

        ### Want to learn more?

        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)

        ### See more complex demos

        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )


def data_sample():
    # Load the SQLite database
    try:
        conn = sqlite3.connect("database.sqlite")
        # Load the 'sample' table into a pandas DataFrame
        query = "SELECT * FROM sample"
        df = pd.read_sql_query(query, conn)
        conn.close()

        def highlight_low_seats(s):
            """
            Highlights rows where 'seatsRemaining' is less than 5.
            """
            if s.name == "seatsRemaining":
                return ["background-color: yellow" if v < 3 else "" for v in s]
            return [""] * len(s)

        # Define a function to highlight non-stop flights
        def highlight_non_stop(row):
            """
            Highlights the entire row if the flight is NOT non-stop.
            """
            if row["isNonStop"] == 0:  # Not non-stop
                return ["background-color: lightgrey"] * len(row)
            return [""] * len(row)

        # Define a function to highlight fares (as before)
        def highlight_fares(s):
            if s.name in ["baseFare", "totalFare"]:
                return [
                    "background-color: lightgreen"
                    if v < df[s.name].mean()
                    else "background-color: lightcoral"
                    for v in s
                ]
            return [""] * len(s)

        # Combine the styles
        styled_df = (
            df.style.apply(
                highlight_non_stop, axis=1
            )  # Highlight full rows for non-stop flights
            .apply(highlight_low_seats)  # Color departure airports
            .apply(highlight_fares)  # Highlight fare columns
        )

        # Use Streamlit to display the table
        st.title("Sample Table Viewer")
        st.dataframe(styled_df, use_container_width=True)

        st.sidebar.markdown("""
        :blue-background[Low fares] will appear green.
        
        :blue-background[High fares] will appear red.
        
        :blue-background[Low remaining] seats will appear yellow.
        
        :blue-background[non direct] flights will appear gray.""")

        st.title("About The Data")
        st.markdown("""
        Our database contains records of flight searches over a six-month period in the United States. The data includes information on several airports within the United States. The total weight of the data is close to 31GB, and it contains over 80 million rows.

        Here's a detailed description of the columns:
        
        - :blue-background[legId]: An identifier for the flight.

        - :blue-background[searchDate]: The date (YYYY-MM-DD) on which this entry was taken from Expedia.

        - :blue-background[flightDate]: The date (YYYY-MM-DD) of the flight.

        - :blue-background[startingAirport]: Three-character IATA airport code for the initial location.

        - :blue-background[destinationAirport]: Three-character IATA airport code for the arrival location.

        - :blue-background[fareBasisCode]: The fare basis code.

        - :blue-background[travelDuration]: The travel duration in hours and minutes.

        - :blue-background[elapsedDays]: The number of elapsed days (usually 0).

        - :blue-background[isBasicEconomy]: Boolean for whether the ticket is for basic economy.

        - :blue-background[isRefundable]: Boolean for whether the ticket is refundable.

        - :blue-background[isNonStop]: Boolean for whether the flight is non-stop.

        - :blue-background[baseFare]: The price of the ticket (in USD).

        - :blue-background[totalFare]: The price of the ticket (in USD) including taxes and other fees.

        - :blue-background[seatsRemaining]: Integer for the number of seats remaining.

        - :blue-background[totalTravelDistance]: The total travel distance in miles. This data is sometimes missing.

        - :blue-background[segmentsDepartureTimeEpochSeconds]: String containing the departure time (Unix time) for each leg of the trip. The entries for each of the legs are separated by '||'.

        - :blue-background[segmentsDepartureTimeRaw]: String containing the departure time (ISO 8601 format: YYYY-MM-DDThh:mm:ss.000±[hh]:00) for each leg of the trip. The entries for each of the legs are separated by '||'.

        - :blue-background[segmentsArrivalTimeEpochSeconds]: String containing the arrival time (Unix time) for each leg of the trip. The entries for each of the legs are separated by '||'.
    
        - :blue-background[segmentsArrivalTimeRaw]: String containing the arrival time (ISO 8601 format: YYYY-MM-DDThh:mm:ss.000±[hh]:00) for each leg of the trip. The entries for each of the legs are separated by '||'.

        - :blue-background[segmentsArrivalAirportCode]: String containing the IATA airport code for the arrival location for each leg of the trip. The entries for each of the legs are separated by '||'.

        - :blue-background[segmentsDepartureAirportCode]: String containing the IATA airport code for the departure location for each leg of the trip. The entries for each of the legs are separated by '||'.

        - :blue-background[segmentsAirlineName]: String containing the name of the airline that services each leg of the trip. The entries for each of the legs are separated by '||'.

        - :blue-background[segmentsAirlineCode]: String containing the two-letter airline code that services each leg of the trip. The entries for each of the legs are separated by '||'.

        - :blue-background[segmentsEquipmentDescription]: String containing the type of airplane used for each leg of the trip (e.g. "Airbus A321" or "Boeing 737-800"). The entries for each of the legs are separated by '||'.

        - :blue-background[segmentsDurationInSeconds]: String containing the duration of the flight (in seconds) for each leg of the trip. The entries for each of the legs are separated by '||'.

        - :blue-background[segmentsDistance]: String containing the distance traveled (in miles) for each leg of the trip. The entries for each of the legs are separated by '||'.

        - :blue-background[segmentsCabinCode]: String containing the cabin for each leg of the trip (e.g. "coach"). The entries for each of the legs are separated by '||'.
        """)
    except sqlite3.Error as e:
        st.error(f"An error occurred while connecting to the database: {e}")


def frist_query():
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


def mapping_demo():
    import pandas as pd
    import pydeck as pdk

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
to display geospatial data.
"""
    )

    @st.cache_data
    def from_data_file(filename):
        url = (
            "http://raw.githubusercontent.com/streamlit/"
            "example-data/master/hello/v1/%s" % filename
        )
        return pd.read_json(url)

    try:
        ALL_LAYERS = {
            "Bike Rentals": pdk.Layer(
                "HexagonLayer",
                data=from_data_file("bike_rental_stats.json"),
                get_position=["lon", "lat"],
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
            "Bart Stop Exits": pdk.Layer(
                "ScatterplotLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_color=[200, 30, 0, 160],
                get_radius="[exits]",
                radius_scale=0.05,
            ),
            "Bart Stop Names": pdk.Layer(
                "TextLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_text="name",
                get_color=[0, 0, 0, 200],
                get_size=15,
                get_alignment_baseline="'bottom'",
            ),
            "Outbound Flow": pdk.Layer(
                "ArcLayer",
                data=from_data_file("bart_path_stats.json"),
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color=[200, 30, 0, 160],
                get_target_color=[200, 30, 0, 160],
                auto_highlight=True,
                width_scale=0.0001,
                get_width="outbound",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        }
        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state={
                        "latitude": 37.76,
                        "longitude": -122.4,
                        "zoom": 11,
                        "pitch": 50,
                    },
                    layers=selected_layers,
                )
            )
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )


def plotting_demo():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")
    st.write(
        """
        This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


def data_frame_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

    @st.cache_data
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write("### Gross Agricultural Production ($B)", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )


page_names_to_funcs = {
    "main page": mainPage,
    "data sample": data_sample,
    "first query": frist_query,
    "Plotting Demo": plotting_demo,
    "Mapping Demo": mapping_demo,
    "DataFrame Demo": data_frame_demo,
}

dashboard = st.sidebar.selectbox("pages", page_names_to_funcs.keys())
page_names_to_funcs[dashboard]()
