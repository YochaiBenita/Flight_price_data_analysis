import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import numpy as np

sqlite_conn = sqlite3.connect("database.sqlite")

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
    def load_data(table_name):
        conn = sqlite3.connect("database.sqlite")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df

    # Sidebar navigation with links to sections
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    - [Flight Data](#flight-data)
    - [Airport Locations](#airport-locations)
    """)

    # Flight Data Section
    st.markdown("<a name='flight-data'></a>", unsafe_allow_html=True)
    st.title("Sample Table Viewer")
    df = load_data("sample")

    # Data styling
    def highlight_low_seats(s):
        return ["background-color: yellow" if v < 3 else "" for v in s] if s.name == "seatsRemaining" else [""] * len(s)

    def highlight_non_stop(row):
        return ["background-color: lightgrey"] * len(row) if row["isNonStop"] == 0 else [""] * len(row)

    def highlight_fares(s):
        return ["background-color: lightgreen" if v < df[s.name].mean() else "background-color: lightcoral" for v in
                s] if s.name in ["baseFare", "totalFare"] else [""] * len(s)

    styled_df = df.style.apply(highlight_non_stop, axis=1).apply(highlight_low_seats).apply(highlight_fares)
    st.dataframe(styled_df, use_container_width=True)

    # Coloring description
    st.markdown("""
    **Color Coding:**
    - 🟩 Low fares appear **green**.
    - 🟥 High fares appear **red**.
    - 🟨 Low remaining seats appear **yellow**.
    - 🏾 Non-direct flights appear **gray**.
    
    **Column Descriptions:**
    
    - **legId**: An identifier for the flight.
    - **searchDate**: The date (YYYY-MM-DD) when the entry was recorded.
    - **flightDate**: The flight's scheduled departure date.
    - **startingAirport**: The IATA airport code for the departure airport.
    - **destinationAirport**: The IATA airport code for the arrival airport.
    - **fareBasisCode**: The fare basis code assigned to the ticket.
    - **travelDuration**: Total travel duration in hours and minutes.
    - **elapsedDays**: The number of days elapsed (usually 0).
    - **isBasicEconomy**: Whether the ticket is for basic economy (Boolean).
    - **isRefundable**: Whether the ticket is refundable (Boolean).
    - **isNonStop**: Whether the flight is non-stop (Boolean).
    - **baseFare**: The base fare price (in USD).
    - **totalFare**: The total fare price, including taxes and fees (in USD).
    - **seatsRemaining**: The number of seats left for the flight.
    - **totalTravelDistance**: The total travel distance in miles.
    """)


    # Airport Locations Section
    st.markdown("<a name='airport-locations'></a>", unsafe_allow_html=True)
    st.title("US Major Airports Visualization")
    df_airports = load_data("airportsLocation")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Airports Data Table")
        st.dataframe(df_airports)

    with col2:
        st.subheader("Airports Map")
        st.map(df_airports[['lat', 'lon']])


def first_query():
    st.title("✈️ Flight Price and Search Analysis Dashboard")

    conn = sqlite3.connect("database.sqlite")
    query = "SELECT * FROM query1"
    df = pd.read_sql_query(query, conn)
    conn.close()

    st.subheader("🎟️ Price and Search Volume Analysis")

    # Create figure with two Y axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # First Y axis - Average Fare
    color1 = '#1f77b4'  # Blue
    sns.lineplot(data=df, x='days_before_flight', y='avg_fare', ax=ax1, color=color1, label='Average Fare')
    ax1.set_xlabel('Days Before Flight')
    ax1.set_ylabel('Average Fare (USD)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.invert_xaxis()

    # Second Y axis - Number of Searches
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'  # Orange
    search_counts = df.groupby('days_before_flight')['number_of_searches'].sum().reset_index()
    sns.lineplot(data=search_counts, x='days_before_flight', y='number_of_searches', ax=ax2, color=color2,
                 label='Number of Searches')
    ax2.set_ylabel('Number of Searches', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title and layout adjustments
    plt.title('Price and Search Volume vs Days Before Flight')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.get_legend().remove()  # Remove second legend

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    st.pyplot(fig)

    # Add insights section
    st.subheader("📈 Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Price Trends**")
        # Calculate price statistics
        max_price = df['avg_fare'].max()
        min_price = df['avg_fare'].min()
        avg_price = df['avg_fare'].mean()
        price_peak_day = df.loc[df['avg_fare'].idxmax(), 'days_before_flight']
        price_lowest_day = df.loc[df['avg_fare'].idxmin(), 'days_before_flight']

        st.write(f"• Highest average fare: ${max_price:.2f} ({price_peak_day} days before flight)")
        st.write(f"• Lowest average fare: ${min_price:.2f} ({price_lowest_day} days before flight)")
        st.write(f"• Overall average fare: ${avg_price:.2f}")

    with col2:
        st.write("**Search Volume Trends**")
        # Calculate search statistics
        max_searches = search_counts['number_of_searches'].max()
        min_searches = search_counts['number_of_searches'].min()
        avg_searches = search_counts['number_of_searches'].mean()
        search_peak_day = search_counts.loc[search_counts['number_of_searches'].idxmax(), 'days_before_flight']
        search_lowest_day = search_counts.loc[search_counts['number_of_searches'].idxmin(), 'days_before_flight']

        st.write(f"• Peak searches: {int(max_searches):,} ({search_peak_day} days before flight)")
        st.write(f"• Lowest searches: {int(min_searches):,} ({search_lowest_day} days before flight)")
        st.write(f"• Average daily searches: {int(avg_searches):,}")

    st.write("**🔍 Analysis Summary**")
    correlations = np.corrcoef(df['avg_fare'], search_counts['number_of_searches'])[0, 1]

    summary_points = [
        f"• The correlation between price and search volume is {correlations:.2f}",
        f"• Prices tend to {'increase' if correlations > 0 else 'decrease'} as search volume {'increases' if correlations > 0 else 'decreases'}",
        f"• The most active search period is around {search_peak_day} days before flight",
        f"• The best time to book (lowest prices) appears to be {price_lowest_day} days before flight"
    ]

    for point in summary_points:
        st.write(point)

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


page_names_to_funcs = {
    "main page": mainPage,
    "data sample": data_sample,
    "Price analysis and searches": first_query,
    "Analysis by distance category": second_query,
    "Analysis by departure time": third_query,
    "Analysis by day of the week": fourth_query,
    "Flight path analysis": fifth_query,
}

dashboard = st.sidebar.selectbox("pages", page_names_to_funcs.keys())
page_names_to_funcs[dashboard]()
