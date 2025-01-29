import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import numpy as np

#import plotly.express as px
#import plotly.graph_objects as go


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
        #-----------------------------------------------------------------------------------------
        st.title('US Major Airports Visualization')

        conn = sqlite3.connect("database.sqlite")
        # Load the 'sample' table into a pandas DataFrame
        query = "SELECT * FROM airportsLocation"
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Display the data table
        st.subheader('Airports Data Table')
        st.dataframe(df)

        # Create map
        st.subheader('Airports Map')

        # Prepare data for the map
        map_data = df[['lat', 'lon']]

        # Display the map using st.map
        st.map(map_data)

        # Add some statistics
        st.subheader('Quick Statistics')
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Airports", len(df))
            st.metric("Northernmost Airport", df.loc[df['lat'].idxmax(), 'airport'])

        with col2:
            st.metric("Southernmost Airport", df.loc[df['lat'].idxmin(), 'airport'])
            st.metric("Westernmost Airport", df.loc[df['lon'].idxmin(), 'airport'])
        #-----------------------------------------------------------------------------------------

    except sqlite3.Error as e:
        st.error(f"An error occurred while connecting to the database: {e}")

#seaborn and matplot
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
#seaborn and matplot
def second_query():
    st.subheader("✈️ Flight Analysis by Distance Category")

    conn = sqlite3.connect("database.sqlite")
    query = "SELECT * FROM query2"
    query2_data = pd.read_sql_query(query, conn)
    conn.close()

    st.sidebar.subheader("🔍 Filters")
    show_direct = st.sidebar.checkbox('Show Direct Flights', value=True)
    show_connection = st.sidebar.checkbox('Show Connection Flights', value=True)

    st.subheader("🎟️ Average Fares by Distance Category")

    plot_data = pd.melt(query2_data,
                        id_vars=['distance_category'],
                        value_vars=['direct_avg_fare', 'connection_avg_fare'],
                        var_name='flight_type',
                        value_name='average_fare')

    selected_types = []
    if show_direct:
        selected_types.append('direct_avg_fare')
    if show_connection:
        selected_types.append('connection_avg_fare')

    filtered_data = plot_data[plot_data['flight_type'].isin(selected_types)]

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    if len(filtered_data) > 0:
        bars = sns.barplot(data=filtered_data,
                           x='distance_category',
                           y='average_fare',
                           hue='flight_type',
                           palette={'direct_avg_fare': 'skyblue', 'connection_avg_fare': 'lightcoral'},
                           ax=ax1)

        ax1.set_title('Average Fares by Distance Category')
        ax1.set_xlabel('Distance Category')
        ax1.set_ylabel('Average Fare (USD)')
        ax1.tick_params(axis='x', rotation=45)

        legend_handles = [plt.Rectangle((0, 0), 1, 1, color='skyblue'),
                          plt.Rectangle((0, 0), 1, 1, color='lightcoral')]
        labels = ['Direct Flights', 'Connection Flights']

        ax1.get_legend().remove()
        ax1.legend(handles=legend_handles, labels=labels, loc='upper right', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        for p in ax1.patches:
            ax1.annotate(f'${p.get_height():.2f}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom')

    st.pyplot(fig1)

    if show_direct or show_connection:
        st.subheader("📈 Key Insights")
        col1, col2 = st.columns(2)

        if show_direct:
            with col1:
                st.write("Direct Flights:")
                st.write(f"- Highest average fare: ${query2_data['direct_avg_fare'].max():.2f}")
                st.write(f"- Total number of flights: {query2_data['direct_flights_count'].sum():,}")

        if show_connection:
            with col2:
                st.write("Connection Flights:")
                st.write(f"- Highest average fare: ${query2_data['connection_avg_fare'].max():.2f}")
                st.write(f"- Total number of flights: {query2_data['connection_flights_count'].sum():,}")
#matplot
def third_query():
    st.subheader("🕒 Flight Analysis by Departure Hour")

    conn = sqlite3.connect("database.sqlite")
    query = "SELECT * FROM query3"
    query3_data = pd.read_sql_query(query, conn)
    conn.close()

    st.sidebar.subheader("📊 Select Metrics")
    show_avg_fare = st.sidebar.checkbox('Average Fare', value=True)
    show_seats = st.sidebar.checkbox('Average Seats Remaining', value=True)
    show_flights = st.sidebar.checkbox('Number of Flights', value=True)

    st.subheader("📈 Hourly Flight Metrics")

    fig1, ax1 = plt.subplots(figsize=(12, 6))

    if show_avg_fare:
        ax1.plot(query3_data['departure_hour'], query3_data['avg_fare'],
                 marker='o', color='skyblue', label='Average Fare ($)')
        ax1.set_ylabel('Average Fare (USD)', color='skyblue')

    if show_seats:
        if not show_avg_fare:
            ax1.set_ylabel('Average Seats Remaining', color='lightcoral')
        ax2 = ax1.twinx()
        ax2.plot(query3_data['departure_hour'], query3_data['avg_seats_remaining'],
                 marker='s', color='lightcoral', label='Avg Seats Remaining')
        ax2.set_ylabel('Average Seats Remaining', color='lightcoral')

    if show_flights:
        if not show_avg_fare and not show_seats:
            ax1.set_ylabel('Number of Flights', color='green')
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(query3_data['departure_hour'], query3_data['number_of_flights'],
                 marker='^', color='green', label='Number of Flights')
        ax3.set_ylabel('Number of Flights', color='green')

    ax1.set_xlabel('Departure Hour')
    ax1.set_title('Flight Metrics by Departure Hour')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(query3_data['departure_hour'])

    lines = []
    labels = []
    if show_avg_fare:
        lines.append(plt.Line2D([0], [0], color='skyblue', marker='o', label='Average Fare ($)'))
        labels.append('Average Fare ($)')
    if show_seats:
        lines.append(plt.Line2D([0], [0], color='lightcoral', marker='s', label='Avg Seats Remaining'))
        labels.append('Avg Seats Remaining')
    if show_flights:
        lines.append(plt.Line2D([0], [0], color='green', marker='^', label='Number of Flights'))
        labels.append('Number of Flights')

    plt.legend(handles=lines, labels=labels, loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    st.pyplot(fig1)

    if any([show_avg_fare, show_seats, show_flights]):
        st.subheader("📊 Key Statistics")
        col1, col2, col3 = st.columns(3)

        if show_avg_fare:
            with col1:
                st.write("Average Fare:")
                st.write(f"- Peak: ${query3_data['avg_fare'].max():.2f}")
                st.write(f"- Low: ${query3_data['avg_fare'].min():.2f}")

        if show_seats:
            with col2:
                st.write("Seats Remaining:")
                st.write(f"- Highest avg: {query3_data['avg_seats_remaining'].max():.1f}")
                st.write(f"- Lowest avg: {query3_data['avg_seats_remaining'].min():.1f}")

        if show_flights:
            with col3:
                st.write("Flight Volume:")
                st.write(f"- Peak: {query3_data['number_of_flights'].max():,} flights")
                st.write(f"- Low: {query3_data['number_of_flights'].min():,} flights")
#matplot
def fourth_query():
    st.subheader("📅 Flight Analysis by Day of Week")

    conn = sqlite3.connect("database.sqlite")
    query = "SELECT * FROM query4"
    query4_data = pd.read_sql_query(query, conn)
    conn.close()

    st.sidebar.subheader("🔍 Select Fare Types")
    show_avg = st.sidebar.checkbox('Average Fare', value=True)
    show_nonstop = st.sidebar.checkbox('Non-Stop Fare', value=True)
    show_connection = st.sidebar.checkbox('Connection Fare', value=True)

    st.subheader("💰 Fares by Day of Week")

    fig1, ax1 = plt.subplots(figsize=(12, 6))

    x = range(len(query4_data['flight_day']))

    if show_avg:
        ax1.plot(x, query4_data['avg_fare'], marker='o', color='skyblue', label='Average Fare')

    if show_nonstop:
        ax1.plot(x, query4_data['avg_nonstop_fare'], marker='s', color='lightcoral', label='Non-Stop Fare')

    if show_connection:
        ax1.plot(x, query4_data['avg_connection_fare'], marker='^', color='green', label='Connection Fare')

    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Fare (USD)')
    ax1.set_title('Flight Fares by Day of Week')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(query4_data['flight_day'], rotation=45)

    if show_avg or show_nonstop or show_connection:
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))

    for line in ax1.lines:
        for x_val, y_val in zip(x, line.get_ydata()):
            ax1.annotate(f'${y_val:.2f}',
                         (x_val, y_val),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

    plt.tight_layout()
    st.pyplot(fig1)

    if show_avg or show_nonstop or show_connection:
        st.subheader("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)

        if show_avg:
            with col1:
                st.write("Average Fare:")
                st.write(f"- Highest: ${query4_data['avg_fare'].max():.2f}")
                st.write(f"- Lowest: ${query4_data['avg_fare'].min():.2f}")

        if show_nonstop:
            with col2:
                st.write("Non-Stop Fare:")
                st.write(f"- Highest: ${query4_data['avg_nonstop_fare'].max():.2f}")
                st.write(f"- Lowest: ${query4_data['avg_nonstop_fare'].min():.2f}")

        if show_connection:
            with col3:
                st.write("Connection Fare:")
                st.write(f"- Highest: ${query4_data['avg_connection_fare'].max():.2f}")
                st.write(f"- Lowest: ${query4_data['avg_connection_fare'].min():.2f}")

def fifth_query():
    st.subheader("🛫 Flight Routes Analysis")

    # Load and prepare data
    conn = sqlite3.connect("database.sqlite")
    query = "SELECT * FROM query5"
    query5_data = pd.read_sql_query(query, conn)
    conn.close()

    query5_data = query5_data.dropna(subset=['avg_fare', 'num_flights'])

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        flight_type = st.selectbox(
            "Flight Type",
            options=['All Flights', 'Non-Stop Only', 'Connecting Flights Only']
        )

    with col2:
        selected_origins = st.multiselect(
            "Select Origin Airports",
            options=sorted(query5_data['startingAirport'].unique()),
            default=[]
        )

    with col3:
        selected_destinations = st.multiselect(
            "Select Destination Airports",
            options=sorted(query5_data['destinationAirport'].unique()),
            default=[]
        )

    # Filter and aggregate data
    filtered_data = query5_data.copy()

    if flight_type == 'All Flights':
        filtered_data = filtered_data.groupby(
            ['startingAirport', 'destinationAirport', 'start_lat', 'start_lon', 'dest_lat', 'dest_lon']
        ).agg({
            'num_flights': 'sum',
            'avg_fare': lambda x: np.average(x, weights=filtered_data.loc[x.index, 'num_flights']),
            'min_fare': 'min',
            'max_fare': 'max'
        }).reset_index()
    elif flight_type == 'Non-Stop Only':
        filtered_data = filtered_data[filtered_data['isNonStop'] == True]
    elif flight_type == 'Connecting Flights Only':
        filtered_data = filtered_data[filtered_data['isNonStop'] == False]

    if selected_origins:
        filtered_data = filtered_data[filtered_data['startingAirport'].isin(selected_origins)]
    if selected_destinations:
        filtered_data = filtered_data[filtered_data['destinationAirport'].isin(selected_destinations)]

    if len(filtered_data) > 0:
        max_flights = filtered_data['num_flights'].max()
        min_price = filtered_data['avg_fare'].min()
        max_price = filtered_data['avg_fare'].max()
        price_range = max_price - min_price

        # Prepare route data with color and width calculations
        route_data = []
        for _, flight in filtered_data.iterrows():
            try:
                # Normalize price for color
                if price_range > 0:
                    normalized_price = (flight['avg_fare'] - min_price) / price_range
                else:
                    normalized_price = 0.5

                # Calculate RGB color (blue for low prices, red for high prices)
                red = int(255 * normalized_price)
                blue = int(255 * (1 - normalized_price))

                # Calculate line width based on number of flights
                width = 1 + (flight['num_flights'] / max_flights) * 10

                # Format numbers for tooltip
                avg_fare = round(float(flight['avg_fare']), 2)
                min_fare = round(float(flight['min_fare']), 2)
                max_fare = round(float(flight['max_fare']), 2)
                num_flights = int(flight['num_flights'])

                route_data.append({
                    'sourcePosition': [flight['start_lon'], flight['start_lat']],
                    'targetPosition': [flight['dest_lon'], flight['dest_lat']],
                    'color': [red, 0, blue],
                    'width': width,
                    'startingAirport': flight['startingAirport'],
                    'destinationAirport': flight['destinationAirport'],
                    'num_flights': num_flights,
                    'avg_fare': avg_fare,
                    'min_fare': min_fare,
                    'max_fare': max_fare,
                    # Add formatted strings for tooltip
                    'avg_fare_display': f"${avg_fare:.2f}",
                    'price_range_display': f"${min_fare:.2f} - ${max_fare:.2f}"
                })
            except (ValueError, TypeError):
                continue

        # Define layers
        ALL_LAYERS = {
            "Flight Routes": pdk.Layer(
                "ArcLayer",
                data=route_data,
                get_source_position='sourcePosition',
                get_target_position='targetPosition',
                get_width='width',
                get_source_color='color',
                get_target_color='color',
                pickable=True,
            ),
            "Airport Locations": pdk.Layer(
                "ScatterplotLayer",
                data=pd.concat([
                    filtered_data[['start_lat', 'start_lon', 'startingAirport']].rename(
                        columns={'startingAirport': 'name', 'start_lat': 'lat', 'start_lon': 'lon'}),
                    filtered_data[['dest_lat', 'dest_lon', 'destinationAirport']].rename(
                        columns={'destinationAirport': 'name', 'dest_lat': 'lat', 'dest_lon': 'lon'})
                ]).drop_duplicates(),
                get_position=["lon", "lat"],
                get_color=[0, 0, 0, 200],
                get_radius=5000,
                pickable=True,
            ),
            "Airport Names": pdk.Layer(
                "TextLayer",
                data=pd.concat([
                    filtered_data[['start_lat', 'start_lon', 'startingAirport']].rename(
                        columns={'startingAirport': 'name', 'start_lat': 'lat', 'start_lon': 'lon'}),
                    filtered_data[['dest_lat', 'dest_lon', 'destinationAirport']].rename(
                        columns={'destinationAirport': 'name', 'dest_lat': 'lat', 'dest_lon': 'lon'})
                ]).drop_duplicates(),
                get_position=["lon", "lat"],
                get_text="name",
                get_size=15,
                get_color=[0, 0, 0, 200],
                get_alignment_baseline="'bottom'",
            )
        }

        # Layer selection
        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]

        if selected_layers:
            # Create the map
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=pdk.ViewState(
                        latitude=39.8283,
                        longitude=-98.5795,
                        zoom=3,
                        pitch=50,
                    ),
                    layers=selected_layers,
                    tooltip={
                        "html": "<b>{startingAirport} ✈ {destinationAirport}</b><br>"
                                "Flights: {num_flights}<br>"
                                "Average Fare: {avg_fare_display}<br>"
                                "Price Range: {price_range_display}"
                    }
                )
            )

            # Legend
            st.write("### Map Legend")
            col1, col2 = st.columns(2)
            with col1:
                st.write("🔵 Lower fares")
            with col2:
                st.write("🔴 Higher fares")
            st.write("*Line thickness indicates number of flights on the route*")

        else:
            st.error("Please choose at least one layer above.")

        # Statistics
        st.write("### Route Statistics")
        stats_df = filtered_data[['startingAirport', 'destinationAirport',
                                  'num_flights', 'avg_fare',
                                  'min_fare', 'max_fare']]

        st.dataframe(stats_df.style.format({
            'avg_fare': '${:.2f}',
            'min_fare': '${:.2f}',
            'max_fare': '${:.2f}',
            'num_flights': '{:,.0f}'
        }))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Routes", len(filtered_data))
        with col2:
            st.metric("Total Flights", f"{filtered_data['num_flights'].sum():,.0f}")
        with col3:
            st.metric("Average Fare", f"${filtered_data['avg_fare'].mean():.2f}")
    else:
        st.warning("No routes found for the selected filters.")

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
    "first query": first_query,
    "second query": second_query,
    "third query": third_query,
    "fourth query": fourth_query,
    "fifth query": fifth_query,
    "Plotting Demo": plotting_demo,
    "Mapping Demo": mapping_demo,
    "DataFrame Demo": data_frame_demo,
}

dashboard = st.sidebar.selectbox("pages", page_names_to_funcs.keys())
page_names_to_funcs[dashboard]()
