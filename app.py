import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Ekurhuleni Women's Safety",
    layout="wide"
)

st.title("Ekurhuleni Women's Safety & Violence Hotspot Analysis")

st.markdown("""
This dashboard analyses historical police-recorded violence in
**Ekurhuleni Metropolitan Municipality** to identify geographic
and temporal patterns and areas with comparatively high recorded
violence.

A hotspot is a statistical designation based on historical recorded
crime. It does **not** mean that a community is inherently dangerous.
""")


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

@st.cache_data
def load_data():

    crime = pd.read_csv(
        "ekurhuleni_station_year_analysis.csv"
    )

    stations = pd.read_csv(
        "ekurhuleni_station_summary.csv"
    )

    annual = pd.read_csv(
        "ekurhuleni_annual_violence_trend.csv"
    )

    metrics = pd.read_csv(
        "hotspot_model_test_metrics.csv"
    )

    return crime, stations, annual, metrics


crime, stations, annual, metrics = load_data()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Filters")

station_options = sorted(
    crime["station"].dropna().unique()
)

selected_station = st.sidebar.selectbox(
    "Police Station",
    ["All Stations"] + list(station_options)
)

year_options = sorted(
    crime["year_start"].dropna().astype(int).unique()
)

selected_year = st.sidebar.selectbox(
    "Reporting Year",
    ["All Years"] + list(year_options)
)


crime_categories = {
    "Overall Violence": "violence_count",
    "Murder": "murder",
    "Attempted Murder": "attempted_murder",
    "Rape": "rape",
    "Sexual Assault": "sexual_assault",
    "Common Assault": "common_assault",
    "Assault GBH": "assault_gbh",
    "Kidnapping": "kidnapping"
}

selected_category_name = st.sidebar.selectbox(
    "Crime Category",
    list(crime_categories.keys())
)

selected_category = crime_categories[selected_category_name]


# --------------------------------------------------
# FILTER
# --------------------------------------------------

filtered = crime.copy()

if selected_station != "All Stations":
    filtered = filtered[
        filtered["station"] == selected_station
    ]

if selected_year != "All Years":
    filtered = filtered[
        filtered["year_start"] == selected_year
    ]


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

with st.sidebar.expander("Dataset Information"):

    dataset = st.selectbox(
        "Dataset",
        [
            "SAPS Crime Data",
            "Municipal Population Indicators"
        ]
    )

    if dataset == "SAPS Crime Data":

        st.markdown("""
### SAPS Crime Data

Historical police-recorded crime statistics were used at
police-station/year level.

The dataset contains police station, municipality, reporting year,
coordinates and multiple recorded crime categories.

The downloadable research dataset required registration. It was
selected because it provides consistently structured long-term
station-level SAPS statistics suitable for temporal hotspot analysis.

The analysis does not assume that recorded crime represents all crime.

**Source:** SAPS crime statistics distributed through DataFirst.
""")

    else:

        st.markdown("""
### Municipal Population Indicators

Municipal population information provides demographic context for
Ekurhuleni.

The population dataset was aligned with the crime data at the
**municipality level**, not police-station level.

It is therefore used as contextual information only and is not used
to calculate station-level crime rates.

This prevents an invalid comparison between station crime counts and
municipality-wide population.
""")


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

st.header("1. Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Observations",
    len(filtered)
)

c2.metric(
    "Police Stations",
    filtered["station"].nunique()
)

if len(filtered) > 0:

    c3.metric(
        "Recorded Violence",
        f"{filtered['violence_count'].sum(skipna=True):,.0f}"
    )

    hotspot_records = filtered["hotspot"].fillna(0).sum()

    c4.metric(
        "Hotspot Records",
        f"{hotspot_records:,.0f}"
    )


st.info("""
The violence measure combines murder, attempted murder, rape,
sexual assault, common assault, assault GBH and kidnapping.

The data are not consistently disaggregated by victim gender.
Therefore, this is interpreted as a broader violence indicator
relevant to women's safety rather than a count of offences committed
exclusively against women.
""")


# --------------------------------------------------
# EDA
# --------------------------------------------------

st.header("2. Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs([
    "Temporal Pattern",
    "Crime Categories",
    "Station Comparison"
])


# TEMPORAL

with tab1:

    st.subheader("Historical Violence Trend")

    if selected_station == "All Stations":

        temporal = (
            crime.groupby("year_start")[selected_category]
            .sum(min_count=1)
            .reset_index()
        )

    else:

        temporal = (
            crime[
                crime["station"] == selected_station
            ]
            .groupby("year_start")[selected_category]
            .sum(min_count=1)
            .reset_index()
        )

    temporal = temporal.dropna()

    fig, ax = plt.subplots(figsize=(11,5))

    ax.plot(
        temporal["year_start"],
        temporal[selected_category],
        marker="o"
    )

    ax.set_title(
        f"Historical Trend: {selected_category_name}"
    )

    ax.set_xlabel("Reporting Year")
    ax.set_ylabel("Recorded Count")

    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.caption("""
    Temporal changes show patterns in police-recorded incidents.
    They do not by themselves establish why violence increased
    or decreased.
    """)


# CATEGORIES

with tab2:

    st.subheader("Recorded Violence Categories")

    category_columns = [
        "murder",
        "attempted_murder",
        "rape",
        "sexual_assault",
        "common_assault",
        "assault_gbh",
        "kidnapping"
    ]

    totals = (
        filtered[category_columns]
        .sum()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10,6))

    totals.plot(
        kind="barh",
        ax=ax
    )

    ax.set_title(
        "Recorded Violence by Category"
    )

    ax.set_xlabel(
        "Recorded Cases"
    )

    st.pyplot(fig)


# STATIONS

with tab3:

    st.subheader("Station Comparison")

    station_values = (
        crime.groupby("station")[selected_category]
        .sum(min_count=1)
        .sort_values(ascending=False)
        .head(15)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10,7))

    station_values.plot(
        kind="barh",
        ax=ax
    )

    ax.set_title(
        f"Highest Recorded {selected_category_name}"
    )

    ax.set_xlabel("Recorded Count")

    st.pyplot(fig)


# --------------------------------------------------
# SPATIAL ANALYSIS
# --------------------------------------------------

st.header("3. Spatial Pattern")

latest_year = int(
    crime["year_start"].dropna().max()
)

map_year = (
    latest_year
    if selected_year == "All Years"
    else selected_year
)

map_data = crime[
    crime["year_start"] == map_year
].copy()

map_data = map_data[
    map_data["coordinates_valid"] == True
]

map_data = map_data.dropna(
    subset=[
        "latitude",
        "longitude",
        selected_category
    ]
)


if len(map_data) > 0:

    fig, ax = plt.subplots(figsize=(9,7))

    scatter = ax.scatter(
        map_data["longitude"],
        map_data["latitude"],
        c=map_data[selected_category],
        s=80,
        cmap="YlOrRd",
        alpha=0.8
    )

    plt.colorbar(
        scatter,
        ax=ax,
        label=selected_category_name
    )

    top5 = map_data.nlargest(
        5,
        selected_category
    )

    for _, row in top5.iterrows():

        ax.annotate(
            row["station"],
            (
                row["longitude"],
                row["latitude"]
            ),
            fontsize=8
        )

    ax.set_title(
        f"Spatial Pattern - {map_year}"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    st.pyplot(fig)


st.caption("""
The spatial visualisation uses police-station coordinates.
No station-boundary polygons were available in the final analysis.
""")


# --------------------------------------------------
# HOTSPOT DEFINITION
# --------------------------------------------------

st.header("4. Hotspot Identification")

st.markdown("""
A station-year is classified as a **hotspot** when its recorded
violence count is equal to or greater than the **75th percentile**
of Ekurhuleni station violence counts for the same reporting year.

This identifies approximately the highest quarter of recorded
violence concentrations relative to other stations in that year.

**Hotspot = 1**  
**Not hotspot = 0**
""")


hotspot_counts = (
    crime["hotspot"]
    .dropna()
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(6,4))

hotspot_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Hotspot Class Distribution")
ax.set_xlabel("0 = Not Hotspot | 1 = Hotspot")
ax.set_ylabel("Observations")

plt.xticks(rotation=0)

st.pyplot(fig)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

st.header("5. Classification Model")

st.success(
    "Selected Model: Logistic Regression"
)

st.markdown("""
Logistic Regression was compared with Random Forest using the
validation period.

**Logistic Regression was selected because it achieved the highest
validation F1-score.**

Previous-year crime variables were used as predictors to reduce
target leakage.

The data were split chronologically:

- Training: 2006–2019
- Validation: 2020–2022
- Test: 2023–2025
""")


# --------------------------------------------------
# METRICS
# --------------------------------------------------

row = metrics.iloc[0]

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Precision",
    f"{row['test_precision']:.3f}"
)

m2.metric(
    "Recall",
    f"{row['test_recall']:.3f}"
)

m3.metric(
    "F1 Score",
    f"{row['test_f1']:.3f}"
)

m4.metric(
    "Balanced Accuracy",
    f"{row['test_balanced_accuracy']:.3f}"
)


m5, m6, m7 = st.columns(3)

m5.metric(
    "ROC-AUC",
    f"{row['test_roc_auc']:.3f}"
)

m6.metric(
    "Average Precision",
    f"{row['test_average_precision']:.3f}"
)

m7.metric(
    "MCC",
    f"{row['test_mcc']:.3f}"
)


st.markdown("""
### Model Interpretation

Recall is particularly important in hotspot classification because
a **false negative** means an actual hotspot was not identified.

False positives also matter because they could direct limited
prevention resources toward an area that does not meet the hotspot
threshold.

Therefore, accuracy alone is not sufficient for evaluating this
public-safety classification problem.
""")


# --------------------------------------------------
# THREE HOTSPOTS
# --------------------------------------------------

st.header("6. Three Selected Hotspot Areas")

top_three = stations.sort_values(
    ["hotspot_share", "mean_violence"],
    ascending=False
).head(3)

st.dataframe(
    top_three[
        [
            "station",
            "years_with_violence",
            "mean_violence",
            "hotspot_years",
            "hotspot_share"
        ]
    ],
    use_container_width=True
)


three_names = top_three["station"].tolist()

st.write(
    "**Selected areas:**",
    ", ".join(three_names)
)


three_data = crime[
    crime["station"].isin(three_names)
].copy()

three_trend = (
    three_data
    .groupby(
        ["year_start", "station"]
    )["violence_count"]
    .sum(min_count=1)
    .reset_index()
)


fig, ax = plt.subplots(figsize=(11,6))

sns.lineplot(
    data=three_trend,
    x="year_start",
    y="violence_count",
    hue="station",
    marker="o",
    ax=ax
)

ax.set_title(
    "Historical Comparison of Three Selected Hotspots"
)

ax.set_xlabel("Reporting Year")
ax.set_ylabel("Recorded Violence Count")

st.pyplot(fig)


st.markdown("""
The three areas were selected using **hotspot persistence first**
and **mean recorded violence second**.

In the notebook these were:

- **Tembisa**
- **Ivory Park**
- **Tsakane**

All three were classified as hotspots in every observed year with
complete hotspot information in the station summary.

This comparison identifies persistence and differences in recorded
violence patterns. It does not establish why those patterns occurred.
""")


# --------------------------------------------------
# POPULATION
# --------------------------------------------------

st.header("7. Population Context")

if "total_pop_2022" in crime.columns:

    population_2022 = crime[
        "total_pop_2022"
    ].dropna()

    female_2022 = crime[
        "female_pop_2022"
    ].dropna()

    if len(population_2022) > 0:

        p1, p2 = st.columns(2)

        p1.metric(
            "Ekurhuleni Population 2022",
            f"{population_2022.iloc[0]:,.0f}"
        )

        p2.metric(
            "Female Population 2022",
            f"{female_2022.iloc[0]:,.0f}"
        )


st.warning("""
Municipal population is NOT used as the denominator for individual
police stations because the population dataset is available at
municipality level rather than station/precinct level.

Dividing individual station crime counts by the entire Ekurhuleni
population would create an invalid station-level crime rate.
""")


# --------------------------------------------------
# PREVENTION
# --------------------------------------------------

st.header("8. Prevention-Oriented Insights")

st.markdown("""
The results can support public-safety planning by helping identify
areas with persistent concentrations of recorded violence.

Possible applications include:

- prioritising prevention resources;
- community-safety visibility;
- identifying locations requiring deeper investigation;
- environmental safety interventions;
- monitoring changes in violence patterns over time;
- supporting targeted community safety programmes.

The model is intended as a **decision-support tool**, not an
automatic resource-allocation system.
""")


# --------------------------------------------------
# LIMITATIONS
# --------------------------------------------------

st.header("9. Limitations")

st.markdown("""
- Police statistics represent **recorded crime**, not all crime.
- Some incidents may never be reported.
- The crime dataset is not consistently disaggregated by victim gender.
- Municipal population cannot be used as a station-level denominator.
- Police-station structures and reporting practices may change over time.
- The 75th-percentile hotspot definition is a methodological choice.
- Geographic analysis uses station coordinates rather than boundary polygons.
- Statistical associations do not establish causation.
- The recent cases motivating the study were **not** used as training data.
- Hotspot status does not mean that a community or its residents are
  inherently dangerous.
""")


st.divider()

st.caption(
    "Ekurhuleni Women's Safety Hotspot Analysis | "
    "Machine Learning Examination"
)