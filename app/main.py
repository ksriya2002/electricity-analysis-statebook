import pandas as pd
import numpy as np
import plotly.express as px
from shiny import App, ui, render, reactive
from shiny.ui import tags
import plotly.graph_objects as go

# Load Excel Data - Using a function to centralize data loading
def load_data():
    try:
        dfs = {
            'operable': pd.read_excel('Operable.xlsx'),
            'proposed': pd.read_excel('Proposed.xlsx'),
            'retired': pd.read_excel('Retired and Canceled.xlsx'),
            'cleaned': pd.read_excel('860PlantDataset.xlsx'),
            'netgencleaned': pd.read_excel('923GeneratorDataset.xlsx')
        }
        return dfs
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
# Load all data
data = load_data()

# For backward compatibility and ease of reference
df_operable = data['operable']
df_proposed = data['proposed']
df_retired = data['retired']
df_cleaned = data['cleaned']
df_netgencleaned = data['netgencleaned']

# Create a State Dropdown Widget - Extract once
states = sorted(df_operable['State'].dropna().unique())
states_operable = states  # For backward compatibility

def calculate_metrics(df, selected_states, operable=True):
    """Calculate metrics based on filtered data"""
    # Filter dataframe based on selected state
    filtered_df = filter_by_states(df, selected_states)
    
    # Basic Counts for Utilities and Plant Codes
    total_utilities = filtered_df['Utility ID'].nunique()
    total_plant_codes = filtered_df['Plant Code'].nunique()

    # Total Unique Generators - using more readable approach to ensure compatibility
    unique_combinations = filtered_df.groupby(['Utility ID', 'Plant Code', 'Generator ID']).size()
    total_unique_generators = len(unique_combinations)

    # Format the numbers with commas
    total_utilities_fmt = f"{total_utilities:,}"
    total_plant_codes_fmt = f"{total_plant_codes:,}"
    total_unique_generators_fmt = f"{total_unique_generators:,}"

    if operable:
        # Count occurrences where 'Uprate or Derate Completed During Year' is 'Y'
        total_uprates_derates = (filtered_df['Uprate or Derate Completed During Year'] == 'Y').sum()
        total_uprates_derates_fmt = f"{total_uprates_derates:,}"
        
        # Check for non-empty values (handling both NaN and empty strings)
        total_planned_uprates = filtered_df['Planned Uprate Year'].apply(lambda x: pd.notnull(x) and str(x).strip() != '').sum()
        total_planned_uprates_fmt = f"{total_planned_uprates:,}"
        
        total_planned_derates = filtered_df['Planned Derate Year'].apply(lambda x: pd.notnull(x) and str(x).strip() != '').sum()
        total_planned_derates_fmt = f"{total_planned_derates:,}"
        
        total_planned_repowers = filtered_df['Planned Repower Year'].apply(lambda x: pd.notnull(x) and str(x).strip() != '').sum()
        total_planned_repowers_fmt = f"{total_planned_repowers:,}"
        
        return (total_utilities_fmt, total_plant_codes_fmt, total_unique_generators_fmt, 
                total_uprates_derates_fmt, total_planned_uprates_fmt, 
                total_planned_derates_fmt, total_planned_repowers_fmt)
    else:
        return total_utilities_fmt, total_plant_codes_fmt, total_unique_generators_fmt

def create_card(value, title, color):
    """Create a styled card for displaying metrics"""
    return tags.div(
        tags.div(
            tags.h5(title, class_="card-title"),
            tags.p(str(value), style="font-size: 25px; font-weight: bold; color: black;", class_="card-text"),
            class_="card-body",
        ),
        class_=f"card bg-{color} text-black mb-3 shadow-lg",
        style="width: 215px; height: 100px; border: 1px solid #ced4da; margin-right: 10px;"
    )

# Wrapper for cards with flexbox style
cards_row = tags.div(
    *[create_card(100, f"Card {i+1}", "primary") for i in range(7)],  # Example with 7 cards
    style="display: flex; justify-content: space-between; gap: 20px; align-items: center;"
)

# ------------------------ CLEANING FUNCTIONS ------------------------

def filter_by_states(df, selected_states):
    """Filter the DataFrame based on selected states."""
    if not selected_states:
        return df
    return df[df["State"].isin(selected_states)]

def convert_columns_to_numeric(df, columns, errors='coerce'):
    """Convert the specified columns to numeric, handling errors."""
    df[columns] = df[columns].apply(pd.to_numeric, errors=errors)
    return df

def drop_na_for_columns(df, columns):
    """Drop rows with NaN in the specified columns."""
    return df.dropna(subset=columns)

def clean_capacity_columns(df):
    """Convert capacity columns to numeric and drop NaNs."""
    capacity_columns = ['Nameplate Capacity (MW)', 'Summer Capacity (MW)', 'Winter Capacity (MW)']
    df = convert_columns_to_numeric(df, capacity_columns)
    return drop_na_for_columns(df, capacity_columns)

def get_technology_counts(df, selected_states):
    """Get value counts for technology column."""
    filtered_df = filter_by_states(df, selected_states)
    counts = filtered_df["Technology"].value_counts().reset_index()
    counts.columns = ['Technology', 'Count']
    return counts.sort_values(by='Count', ascending=False)

def get_status_by_sector_counts(df, selected_states):
    """Return generator counts grouped by sector and status after filtering by selected states."""
    return (
        filter_by_states(df, selected_states)
        .groupby(['Sector Name', 'Status'])
        .size()
        .reset_index(name='Count')
    )

# Helper function to clean binary technology columns
def clean_binary_columns(df, selected_states, tech_columns):
    """Clean and prepare the binary tech indicator columns."""
    filtered_df = filter_by_states(df, selected_states)
    df_tech = filtered_df[tech_columns].replace({'Y': 1, 'N': 0, '': 0})
    df_tech_count = df_tech.sum().reset_index()
    df_tech_count.columns = ['Technology', 'Count']
    df_tech_count = df_tech_count[df_tech_count['Count'] > 0]
    df_tech_count['Technology'] = df_tech_count['Technology'].str.replace('?', '', regex=False)
    return df_tech_count

def clean_other_tech_columns(df, selected_states):
    """Clean and prepare the binary tech indicator columns."""
    tech_columns = [
        'Carbon Capture Technology?', 'Fluidized Bed Technology?', 'Pulverized Coal Technology?',
        'Stoker Technology?', 'Other Combustion Technology?', 'Subcritical Technology?',
        'Supercritical Technology?', 'Ultrasupercritical Technology?'
    ]
    return clean_binary_columns(df, selected_states, tech_columns)

def clean_planned_retirement_data(df, selected_states):
    """Filter and clean the retirement data by selected states."""
    filtered_df = filter_by_states(df, selected_states)
    filtered_df = convert_columns_to_numeric(filtered_df, ['Planned Retirement Year', 'Nameplate Capacity (MW)'])
    filtered_df = drop_na_for_columns(filtered_df, ['Planned Retirement Year'])

    if filtered_df.empty:
        return pd.DataFrame()

    return filtered_df.groupby(['Planned Retirement Year', 'Technology']).agg(
        Capacity_MW=('Nameplate Capacity (MW)', 'sum'),
        Count=('Technology', 'count')
    ).reset_index()

def clean_capacity_data(df, selected_states, last_n_years=5):
    """Clean and filter the data based on selected states and the last N years."""
    filtered_df = filter_by_states(df, selected_states)
    filtered_df = convert_columns_to_numeric(filtered_df, ['Operating Year'])
    
    most_recent_year = filtered_df["Operating Year"].max()
    start_year = most_recent_year - last_n_years + 1
    recent_df = filtered_df[(filtered_df["Operating Year"] >= start_year) & filtered_df["Operating Year"].notna()]

    homes_per_mw = 834
    capacity_by_tech = recent_df.groupby("Technology")["Nameplate Capacity (MW)"].sum().reset_index()
    capacity_by_tech.columns = ['Technology', 'Capacity Added (MW)']
    capacity_by_tech['Homes Powered'] = capacity_by_tech['Capacity Added (MW)'] * homes_per_mw
    return capacity_by_tech.sort_values(by='Homes Powered', ascending=False)

def clean_uprates_data(df, selected_states):
    """Clean and filter the data for uprates based on the selected states."""
    filtered_df = filter_by_states(df, selected_states)
    df_uprates = filtered_df[['Planned Net Summer Capacity Uprate (MW)', 
                              'Planned Net Winter Capacity Uprate (MW)', 
                              'Planned Uprate Year', 
                              'Energy Source 1']].dropna(subset=['Planned Uprate Year'])

    uprates = df_uprates[df_uprates[['Planned Net Summer Capacity Uprate (MW)', 
                                       'Planned Net Winter Capacity Uprate (MW)']].notna().any(axis=1)]
    uprates['Type'] = 'Uprate'

    aggregated_data = uprates.groupby(['Planned Uprate Year', 'Energy Source 1']).size().reset_index(name='Count')
    aggregated_data = aggregated_data.rename(columns={'Planned Uprate Year': 'Year', 'Energy Source 1': 'Energy Source'})

    aggregated_data['Year'] = pd.to_numeric(aggregated_data['Year'], errors='coerce')
    aggregated_data = aggregated_data.dropna(subset=['Year'])
    aggregated_data['Year'] = aggregated_data['Year'].astype(int).astype(str)

    return aggregated_data

month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
             7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

def clean_derates_data(df, selected_states):
    """Clean and filter the data for derates based on the selected states."""
    filtered_df = filter_by_states(df, selected_states)
    df_derates = filtered_df[['Planned Net Summer Capacity Derate (MW)', 
                              'Planned Net Winter Capacity Derate (MW)', 
                              'Planned Derate Month', 
                              'Energy Source 1']].dropna(subset=['Planned Derate Month'])

    derates = df_derates[df_derates[['Planned Net Summer Capacity Derate (MW)', 
                                     'Planned Net Winter Capacity Derate (MW)']].notna().any(axis=1)]

    aggregated_data = derates.groupby(['Planned Derate Month', 'Energy Source 1']).size().reset_index(name='Count')
    
    aggregated_data['Month'] = pd.to_numeric(aggregated_data['Planned Derate Month'], errors='coerce')
    valid_months = aggregated_data[aggregated_data['Month'].between(1, 12)].copy()
    if not valid_months.empty:
        valid_months['Month'] = valid_months['Month'].map(month_map)
        aggregated_data = valid_months
    else:
        aggregated_data = pd.DataFrame(columns=['Month', 'Energy Source 1', 'Count'])

    return aggregated_data

def clean_repowers_data(df, selected_states):
    """Cleans and prepares the repower data for charting."""
    filtered_df = filter_by_states(df, selected_states)
    filtered_df = drop_na_for_columns(filtered_df, ['Planned Repower Year'])
    filtered_df = convert_columns_to_numeric(filtered_df, ['Planned Repower Year'])
    
    return filtered_df.groupby(['Planned Repower Year', 'Planned Energy Source 1']).size().reset_index(name='Repower Count')

def clean_capacity_by_technology_data(df, selected_states):
    """Cleans and prepares the data for generating the capacity by technology table."""
    filtered_df = filter_by_states(df, selected_states)
    pivot_table = pd.pivot_table(
        filtered_df,
        index='Technology',
        columns='Operating Year',
        aggfunc='size',
        fill_value=0,
    )
    pivot_table['TOTAL'] = pivot_table.sum(axis=1)
    total_row = pd.DataFrame(pivot_table.sum(axis=0)).T
    total_row.index = ['TOTAL']
    pivot_table = pd.concat([pivot_table, total_row])

    pivot_table.columns.name = 'Tech/Year'
    return pivot_table

def clean_effective_date_data(df, selected_states):
    """Filters the data based on selected states and processes the date columns."""
    filtered_df = filter_by_states(df, selected_states)
    filtered_df['Effective Date'] = (
        filtered_df['Effective Year'].astype(str) + '-' + filtered_df['Effective Month'].astype(str).str.zfill(2)
    )
    filtered_df['Current Date'] = (
        filtered_df['Current Year'].astype(str) + '-' + filtered_df['Current Month'].astype(str).str.zfill(2)
    )
    
    return filtered_df

def clean_retirement_year_data(df, selected_states):
    """Cleans and prepares the data for retirement year chart."""
    df = convert_columns_to_numeric(df, ['Retirement Year', 'Nameplate Capacity (MW)'])
    df = drop_na_for_columns(df, ['Retirement Year', 'Technology', 'Generator ID', 'Nameplate Capacity (MW)'])
    
    filtered_df = filter_by_states(df, selected_states)
    
    cap_by_year_tech = df.groupby(['Retirement Year', 'Technology'])['Nameplate Capacity (MW)'].sum().reset_index()
    gen_count = df.groupby('Retirement Year')['Generator ID'].count().reset_index(name='Generator Count')

    return cap_by_year_tech, gen_count
# ------------------------ PLOTTING FUNCTIONS ------------------------

def create_capacity_chart(df, selected_states, figsize=(14, 10)):
    """Creates a grouped bar chart for capacity by technology with dynamic y-axis scaling."""
    filtered_df = clean_capacity_columns(filter_by_states(df, selected_states))
    capacity_columns = ['Nameplate Capacity (MW)', 'Summer Capacity (MW)', 'Winter Capacity (MW)']

    # Aggregate capacity by technology
    capacity_by_technology = (
        filtered_df.groupby('Technology')[capacity_columns]
        .sum().reset_index()
        .sort_values(by='Nameplate Capacity (MW)', ascending=False)
    )

    # Flatten values to check range for dynamic y-axis scaling
    flat_values = capacity_by_technology[capacity_columns].values.flatten()
    non_zero_values = flat_values[flat_values > 0]

    yaxis_type = 'linear'
    if non_zero_values.size > 0:
        value_range = non_zero_values.max() / non_zero_values.min()
        if value_range > 100:
            yaxis_type = 'log'

    # Create chart
    fig = px.bar(
        capacity_by_technology,
        x='Technology',
        y=capacity_columns,
        title="Capacity by Technology (Nameplate, Summer, and Winter)",
        labels={"Technology": "Technology", 'value': "Total Capacity (MW)", 'variable': "Capacity Type"},
        barmode='group',
        color_discrete_map={
            'Nameplate Capacity (MW)': '#0F52BA',
            'Summer Capacity (MW)': '#ADD8E6',
            'Winter Capacity (MW)': '#87CEEB'
        }
    )

    fig.update_traces(hovertemplate='%{y:,.0f} MW')
    fig.update_layout(
        yaxis=dict(title="Capacity (MW)", tickformat=',.0f', type=yaxis_type)
    )

    return ui.HTML(fig.to_html(full_html=False))



def create_technology_distribution_plot(df, selected_states, figsize=(12, 10)):
    technology_counts = get_technology_counts(df, selected_states)

    fig = go.Figure(go.Scatter(
        x=technology_counts['Count'],
        y=technology_counts['Technology'],
        mode='markers',
        marker=dict(
            size=8,
            color=technology_counts['Count'],
            colorscale='Blues',
            line=dict(width=1, color='Black'),
            showscale=True,
            colorbar=dict(title='Count', thickness=20, x=1.05, tickformat=',.0f')
        )
    ))

    fig.update_traces(hovertemplate='Technology: %{y}<br>Count: %{x:,.0f}')
    shapes = [
        dict(type='line', x0=0, y0=tech, x1=count, y1=tech, line=dict(color='lightgray', width=1))
        for tech, count in zip(technology_counts['Technology'], technology_counts['Count'])
    ]

    fig.update_layout(
        title="Distribution of technologies",
        xaxis_title="Count",
        yaxis_title="Technology",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200, r=200, t=50, b=50),
        shapes=shapes
    )

    return ui.HTML(fig.to_html(full_html=False))


def create_status_by_sector_chart(df, selected_states):
    """Render a grouped bar chart of generator counts by sector and status with tooltips and adaptive Y-axis scaling."""
    df_status = get_status_by_sector_counts(df, selected_states)

    if df_status.empty:
        fig = px.bar(x=[], y=[], title="No data available")
        return ui.HTML(fig.to_html(full_html=False))

    # Status code to description mapping
    status_descriptions = {
        'OP': 'Operating - in service and producing electricity',
        'SB': 'Standby/Backup - available but not normally used',
        'OS': 'Out of service, not returning next year',
        'OA': 'Out of service, expected to return next year',
        'RE': 'Retired - no longer in service',
        'CN': 'Cancelled (was planned)',
        'IP': 'Planned new, indefinitely postponed',
        'TS': 'Construction complete, not yet operational',
        'P':  'Planned, approvals not started',
        'L':  'Regulatory approvals pending',
        'T':  'Regulatory approvals received',
        'U':  'Under construction ≤ 50% complete',
        'V':  'Under construction > 50% complete',
        'OT': 'Other',
    }

    # Add status description and combined label
    df_status['Status Description'] = df_status['Status'].map(status_descriptions)
    df_status['Status Type'] = df_status['Status'] + ' - ' + df_status['Status Description']

    # Determine sector sorting order
    sorted_sectors = (
        df_status.groupby('Sector Name')['Count']
        .sum()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # Calculate min and max for Y-axis optimization
    non_zero_counts = df_status['Count'][df_status['Count'] > 0]
    yaxis_type = 'linear'
    if not non_zero_counts.empty and non_zero_counts.max() / non_zero_counts.min() > 100:
        yaxis_type = 'log'

    # Create plot
    fig = px.bar(
        df_status,
        x='Sector Name',
        y='Count',
        color='Status',
        color_discrete_sequence=px.colors.sequential.Blues_r,
        title="Generator Count by Sector and Status",
        labels={'Count': 'Generator Count', 'Sector Name': 'Sector'},
        barmode='group',
        hover_data={
            'Status': False,
            'Sector Name': True,
            'Count': True,
            'Status Type': True
        }
    )

    fig.update_layout(
        xaxis_title="Sector",
        yaxis_title="Generator Count",
        xaxis={'categoryorder': 'array', 'categoryarray': sorted_sectors},
        yaxis=dict(type=yaxis_type, tickformat=',.0f'),
        showlegend=True
    )

    return ui.HTML(fig.to_html(full_html=False))

def create_other_technology_distribution_chart(df, selected_states):
    df_tech_count = clean_other_tech_columns(df, selected_states)

    if df_tech_count.empty:
        fig_empty = go.Figure()
        fig_empty.add_annotation(
            text="No other advanced technologies used in the selected state",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20)
        )
        fig_empty.update_layout(title="Technology Implementation Across Power Plants")
        return ui.HTML(fig_empty.to_html(full_html=False))

    fig = px.pie(
        df_tech_count,
        names='Technology',
        values='Count',
        title='Technology Implementation Across Power Plants',
        color_discrete_sequence=px.colors.sequential.Blues
    )

    fig.update_layout(legend_title="Technology")
    return ui.HTML(fig.to_html(full_html=False))

#-----------------------------------------------Operable Page Functions---------------------------------------------------------------------------------------------
#-------------------------------------------------------Planned Retirement Distribution-----------------------------------------------------------------------------
def create_planned_retirement_chart(df, selected_states):
    """
    Creates a bubble chart showing the distribution of retired plants 
    by year, technology, and capacity using cleaned grouped data.
    """
    grouped = clean_planned_retirement_data(df, selected_states)

    if grouped.empty:
        fig_empty = go.Figure()
        fig_empty.add_annotation(
            text="No planned retirements found for selected states",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20)
        )
        fig_empty.update_layout(title="Planned Retirement Year Distribution by Technology and Capacity")
        return ui.HTML(fig_empty.to_html(full_html=False))

    fig = px.scatter(
        grouped,
        x='Planned Retirement Year',
        y='Technology',
        size='Capacity_MW',
        color='Count',
        color_continuous_scale='Blues',
        title='Planned Retirement Year Distribution by Technology and Capacity',
        labels={
            'Planned Retirement Year': 'Year',
            'Technology': 'Technology',
            'Capacity_MW': 'Capacity (MW)',
            'Count': 'Number of Plants'
        },
        hover_data={
            'Capacity_MW': ':.1f',
            'Count': ':.1f'
        }
    )

    unique_years = sorted(grouped['Planned Retirement Year'].unique())
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=50),
        xaxis=dict(tickmode='array', tickvals=unique_years, tickangle=90),
        coloraxis_colorbar=dict(title="Number of Plants", tickformat=',.1f')
    )

    sizeref = 0.5 * grouped['Capacity_MW'].max() / (40.0 ** 2)
    fig.update_traces(
        marker=dict(
            opacity=0.8,
            line=dict(width=0.5, color='Black'),
            sizemode='area',
            sizeref=sizeref
        ),
        hovertemplate=(
            'Year: %{x}<br>'
            'Technology: %{y}<br>'
            'Capacity: %{marker.size:,.1f} MW<br>'
            'Count: %{marker.color:,.1f}'
        )
    )

    return ui.HTML(fig.to_html(full_html=False))



#--------------------------------------------------------Capacity Added in the Last 5 years ------------------------------------------------------------------
def create_capacity_added_chart(capacity_by_tech):
    """Create a bar chart of homes powered by technology with dynamic Y-axis scaling."""

    if capacity_by_tech.empty:
        fig = px.bar(x=[], y=[], title="No data available")
        return fig

    # Determine if log scale is needed based on value range
    homes_powered_values = capacity_by_tech['Homes Powered'].values
    non_zero_values = homes_powered_values[homes_powered_values > 0]

    yaxis_type = 'linear'
    min_y_value = None

    if non_zero_values.size > 0:
        value_range = non_zero_values.max() / non_zero_values.min()
        if value_range > 100:
            yaxis_type = 'log'
            min_y_value = 1  # Clip small values for visibility on log scale

    # Add clipped values column for display
    capacity_by_tech['Homes Powered (Clipped)'] = capacity_by_tech['Homes Powered'].clip(lower=min_y_value)

    # Create the bar chart
    fig = px.bar(
        capacity_by_tech,
        x="Technology",
        y="Homes Powered (Clipped)",
        labels={"Technology": "Technology", "Homes Powered (Clipped)": "Homes Powered"},
        color="Homes Powered",
        color_continuous_scale="Blues",
        hover_data={
            "Technology": True,
            "Capacity Added (MW)": True,
            "Homes Powered": True  # Show original values in hover
        },
    )

    # Format hover and layout
    fig.update_traces(
        hovertemplate=(
            "Technology=%{x}<br>"
            "Homes Powered=%{customdata[1]:,.0f}<br>"
            "Capacity Added (MW)=%{customdata[0]:,.0f}"
        ),
        customdata=capacity_by_tech[['Capacity Added (MW)', 'Homes Powered']].values
    )

    fig.update_layout(
        yaxis=dict(
            title="Homes Powered",
            type=yaxis_type,
            tickformat="~s" if yaxis_type == 'log' else ',.0f'
        ),
        yaxis2=dict(
            title="Capacity Added (MW)",
            overlaying="y",
            side="right"
        ),
        showlegend=False
    )

    return fig



def create_capacity_added_plot(df, selected_states, last_n_years=5):
    """Main function to clean data, create the chart, and display modal."""
    # Clean the data
    capacity_by_tech = clean_capacity_data(df, selected_states, last_n_years)

    # Determine if log scale should be used
    use_log_scale = len(selected_states) == 0  # Use log scale when no states are selected (all states)

    # Create the chart
    fig = create_capacity_added_chart(capacity_by_tech)

    # Title and modal HTML - combined for efficiency
    html_components = f"""
    <div style="display: flex; justify-content: flex-start; align-items: center; gap: 10px; margin-bottom: 10px;">
        <h5 style="margin: 0;">Capacity Added in the Last {last_n_years} Years</h5>
        <button type="button" class="btn btn-outline-info btn-sm" data-bs-toggle="modal" data-bs-target="#formulaModal" title="Click to view formula explanation">
            ℹ️
        </button>
    </div>
    <div class="modal fade" id="formulaModal" tabindex="-1" aria-labelledby="formulaModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="formulaModalLabel">Formula Explanation</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body" style="font-size: 15px;">
            <p><strong>Formula:</strong><br>
            Homes Powered = Capacity Added (MW) × <strong>834</strong></p>
            <p><strong>Where does 834 come from?</strong><br>
            - 1 MW = 1,000 kW<br>
            - 1 MW generates: 1,000 kW × 24 hrs/day × 365 days = 8,760,000 kWh/year<br>
            - Average U.S. household uses ~10,500 kWh/year<br>
            - 8,760,000 ÷ 10,500 ≈ <strong>834 homes per MW</strong></p>
            <p>This helps translate technical capacity data into everyday impact.</p>
          </div>
        </div>
      </div>
    </div>
    """ + fig.to_html(full_html=False)

    return ui.HTML(html_components)
#----------------------------------------------------------------Planned Uprate------------------------------------------------------------------------
def create_uprate_chart(df, selected_states):
    """Creates a bar chart showing uprates per year by energy source based on the selected state."""
    # Clean and prepare the data
    aggregated_data = clean_uprates_data(df, selected_states)

    # Handle empty dataset scenario
    if aggregated_data.empty:
        fig_uprates = go.Figure()
        fig_uprates.add_annotation(
            text="No uprates data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20)
        )
        fig_uprates.update_layout(title='Uprates per Year by Energy Source')
        return ui.HTML(fig_uprates.to_html(full_html=False))
    
    # Sort the data and prepare colors
    aggregated_data = aggregated_data.sort_values(by='Count', ascending=True)
    num_colors = len(aggregated_data['Energy Source'].unique())
    colors = px.colors.sequential.Blues[:num_colors]

    # Create the bar chart
    fig = px.bar(
        aggregated_data,
        x='Year',
        y='Count',
        color='Energy Source',
        title='Uprates per Year by Energy Source',
        labels={'Year': 'Year', 'Count': 'Number of Uprates', 'Energy Source': 'Energy Source'},
        barmode='stack',
        category_orders={'Year': sorted(aggregated_data['Year'].unique())},
        color_discrete_sequence=colors
    )

    return ui.HTML(fig.to_html(full_html=False))

#-------------------------------------------------------------Planned Derate-------------------------------------------------------------------------------
def create_derate_chart(df, selected_states):
    """Creates a bar chart showing derates per month by energy source based on the selected state."""
    # Clean and prepare the data
    aggregated_data = clean_derates_data(df, selected_states)

    # Handle empty dataset scenario
    if aggregated_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No derates data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20)
        )
        fig.update_layout(title='Derates per Month by Energy Source')
        return ui.HTML(fig.to_html(full_html=False))

    # Sort the data and prepare color mapping
    aggregated_data = aggregated_data.sort_values(by='Count', ascending=True)
    color_map = {
        source: px.colors.sequential.Blues[-(i % len(px.colors.sequential.Blues)) - 1] 
        for i, source in enumerate(aggregated_data['Energy Source 1'].unique())
    }

    # Create the bar chart
    fig = px.bar(
        aggregated_data,
        x='Month',
        y='Count',
        color='Energy Source 1',
        title='Derates per Month by Energy Source',
        labels={'Month': 'Month', 'Count': 'Number of Derates', 'Energy Source 1': 'Energy Source'},
        barmode='stack',
        category_orders={'Month': list(month_map.values())},
        color_discrete_map=color_map
    )

    return ui.HTML(fig.to_html(full_html=False))
#--------------------------------------------------repower with Planned Energy Source 1---------------------------------------------------------------
def create_repower_chart(df, selected_states):
    """Creates a bar chart showing repowers per year by energy source based on the selected state."""
    # Clean and prepare the data
    df_repowers = clean_repowers_data(df, selected_states)

    # Handle empty dataset scenario
    if df_repowers.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No repowers data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20)
        )
        fig.update_layout(title='Repowers per Year by Energy Source')
        return ui.HTML(fig.to_html(full_html=False))

    # Sort and prepare color mapping
    df_repowers = df_repowers.sort_values(by='Repower Count', ascending=True)
    unique_sources = df_repowers['Planned Energy Source 1'].unique()
    color_map = {
        source: px.colors.sequential.Blues[i % len(px.colors.sequential.Blues)] 
        for i, source in enumerate(unique_sources)
    }

    # Create the bar chart
    fig = px.bar(
        df_repowers, 
        x='Planned Repower Year', 
        y='Repower Count', 
        color='Planned Energy Source 1',
        title='Number of Repowers per Year by Energy Source',
        labels={'Repower Count': 'Number of Repowers', 'Planned Repower Year': 'Year'},
        barmode='stack',
        color_discrete_map=color_map
    )

    # Update x-axis ticks
    fig.update_xaxes(tickmode='array', tickvals=df_repowers['Planned Repower Year'].unique())

    return ui.HTML(fig.to_html(full_html=False))
#-------------------------------------------------------Age of technology table-------------------------------------------------------------------------------------
def create_capacity_by_technology_table(df, selected_states):
    """Creates an HTML table showing generating capacity by technology and operating year, with comma formatting."""
    # Clean and prepare the data
    pivot_table = clean_capacity_by_technology_data(df, selected_states)

    # Format the table with commas for better readability
    formatted_pivot = pivot_table.applymap(lambda x: f"{x:,}")

    # Generate HTML table with clean formatting
    return ui.HTML(formatted_pivot.to_html(classes="table table-striped table-hover"))
#----------------------------------------------------Proposed Page Functions----------------------------------------------------------------------------------------

#-----------------------------------------------------Effective Year vs Current Year------------------------------------------------------------------------------
def create_effective_date_chart(df, selected_states):
    """
    Creates a scatter plot comparing Effective and Current Dates, marking same dates in red.
    """
    # Clean and prepare the data
    filtered_df = clean_effective_date_data(df, selected_states)

    # Create a list of colors based on date equality
    colors = ['mediumspringgreen' if eff == curr else 'navy' for eff, curr in zip(filtered_df['Effective Date'], filtered_df['Current Date'])]

    # Plot using Plotly
    fig = go.Figure(go.Scatter(
        x=filtered_df['Effective Date'],
        y=filtered_df['Current Date'],
        mode='markers',
        marker=dict(color=colors)  # Use the colors list
    ))

    # Customize Layout
    fig.update_layout(
        title=f'Generator Operational Timelines: Planned vs. Revised Dates',
        xaxis_title='Effective Date',
        yaxis_title='Current Date',
        template='plotly_white'
    )

    # Return the chart as an HTML component
    return ui.HTML(fig.to_html(full_html=False))

#-------------------------------------------------------Retired Page Functions-------------------------------------------------------------------------------------

#----------------------------------------------- Capacity by Technology & Generator Retirements Over Time----------------------------------------------------------

def create_retirement_year_combo_chart(df, selected_states):
    """Generates a combo chart of capacity by technology and generator retirements over time."""
    # Clean and prepare the data
    cap_by_year_tech, gen_count = clean_retirement_year_data(df, selected_states)
    
    # Custom color palette for technologies
    custom_blue_gray_palette = [
        "#0d47a1", "#1565c0", "#1976d2", "#1e88e5", "#2196f3", "#42a5f5", "#64b5f6",
        "#90caf9", "#bbdefb", "#e3f2fd", "#546e7a", "#607d8b", "#78909c", "#90a4ae",
        "#b0bec5", "#cfd8dc", "#eceff1", "#b3cde0", "#a9c0d9", "#8eaecb", "#6a9fb5"
    ]
    
    # List of technologies
    technologies = [
        "All Other", "Batteries", "Conventional Hydroelectric", "Conventional Steam Coal",
        "Geothermal", "Landfill Gas", "Municipal Solid Waste", "Natural Gas Fired Combined Cycle",
        "Natural Gas Fired Combustion Turbine", "Natural Gas Internal Combustion Engine",
        "Natural Gas Steam Turbine", "Nuclear", "Onshore Wind Turbine", "Other Gases",
        "Other Natural Gas", "Other Waste Biomass", "Petroleum Coke", "Petroleum Liquids",
        "Solar Photovoltaic", "Solar Thermal without Energy Storage", "Wood/Wood Waste Biomass"
    ]
    
    # Map each technology to a color from the palette
    tech_color_map = dict(zip(technologies, custom_blue_gray_palette))

    # Create the figure for the combo chart
    fig = go.Figure()

    # Add bar traces for each technology
    for tech in technologies:
        tech_data = cap_by_year_tech[cap_by_year_tech['Technology'] == tech]
        if not tech_data.empty:
            fig.add_trace(go.Bar(
                x=tech_data['Retirement Year'],
                y=tech_data['Nameplate Capacity (MW)'],
                name=tech,
                marker=dict(color=tech_color_map.get(tech, "#e3f2fd"))  # Default color if tech not in map
            ))

    # Add a scatter trace for the generator count
    fig.add_trace(go.Scatter(
        x=gen_count['Retirement Year'],
        y=gen_count['Generator Count'],
        name='Retired Generators',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='black', width=2, dash='dot'),
        marker=dict(size=8)
    ))

    # Update the layout of the figure
    fig.update_layout(
        title='Capacity by Technology & Generator Retirements Over Time',
        template='plotly_white',
        barmode='stack',
        xaxis_title='Retirement Year',
        yaxis=dict(
            title='Total Retired Capacity (MW)',
            side='left',
            type='log',
            tickvals=[1, 10, 100, 1000, 10000, 100000],
            ticktext=["1", "10", "100", "1,000", "10,000", "100,000"]
        ),
        yaxis2=dict(
            title='Number of Retired Generators',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=1.05, y=1,
            traceorder="normal",
            bordercolor="LightGray",
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(r=300),
        height=650
    )

    # Return the chart as an HTML component
    return ui.HTML(fig.to_html(full_html=False))

#-----------------------------------------------------------COMBINED PAGE---------------------------------------------------------------------------------------------
# Common data cleaning function
def get_filtered_df(selected_states):
    """Base filter function for standard dataset."""
    return df_cleaned if not selected_states else df_cleaned[df_cleaned['State'].isin(selected_states)]

def get_filtered_netgen_df(selected_states):
    """Base filter function for net generation dataset."""
    # Clean column names once
    cleaned_df = df_netgencleaned.copy()
    cleaned_df.columns = cleaned_df.columns.str.replace('\n', ' ').str.strip().str.replace(' +', ' ', regex=True)
    
    return cleaned_df if not selected_states else cleaned_df[cleaned_df['State'].isin(selected_states)]

def clean_data_for_capacity_map(selected_states):
    """Clean and prepare data for capacity map."""
    return get_filtered_df(selected_states)

def clean_data_for_summary(selected_states):
    """Clean and prepare data for summary dataframe calculation."""
    filtered_df = get_filtered_df(selected_states)
    
    # Convert capacity to numeric if needed
    filtered_df['Nameplate Capacity (MW)'] = pd.to_numeric(filtered_df['Nameplate Capacity (MW)'], errors='coerce').fillna(0)
    
    return filtered_df

def clean_data_for_capacity_treemap(selected_states):
    """Clean and prepare data for capacity treemap."""
    filtered_df = get_filtered_df(selected_states)
    
    # Ensure numeric data
    numeric_cols = ['Nameplate Capacity (MW)', 'Summer Capacity (MW)', 'Winter Capacity (MW)']
    filtered_df[numeric_cols] = filtered_df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Group and calculate capacities for all Technologies and Status_Types
    tech_status_capacity_df = filtered_df.groupby(['Technology', 'Status_Type'])[numeric_cols].sum().reset_index()
    
    # Round the capacities
    tech_status_capacity_df[numeric_cols] = tech_status_capacity_df[numeric_cols].round(2)
    
    # Add highlight condition
    tech_status_capacity_df['Highlight'] = tech_status_capacity_df.apply(
        lambda row: 'red' if (row['Summer Capacity (MW)'] > row['Nameplate Capacity (MW)']) or
                          (row['Winter Capacity (MW)'] > row['Nameplate Capacity (MW)']) else 'blue',
        axis=1
    )
    
    return tech_status_capacity_df

def clean_data_for_generator_count(selected_states):
    """Clean and prepare data for generator count treemap."""
    filtered_df = get_filtered_df(selected_states)
    
    # Handle missing values
    filtered_df['Technology'] = filtered_df['Technology'].fillna('Unknown')
    filtered_df['Status_Type'] = filtered_df['Status_Type'].fillna('Unknown')
    
    # Group by Technology and Status_Type
    generator_counts = (
        filtered_df
        .groupby(['Technology', 'Status_Type'])
        .size()
        .reset_index(name='Generator Count')
    )
    
    return generator_counts

def clean_data_for_energy_source(selected_states):
    """Clean and prepare data for energy source summary table."""
    filtered_df = get_filtered_df(selected_states)
    
    # Ensure numeric
    filtered_df['Nameplate Capacity (MW)'] = pd.to_numeric(filtered_df['Nameplate Capacity (MW)'], errors='coerce').fillna(0)
    
    # Aggregate
    summary = filtered_df.groupby(['State', 'Energy Source 1', 'Status_Type'])['Nameplate Capacity (MW)'].sum().reset_index()
    
    # Pivot
    pivot = summary.pivot_table(index=['State', 'Energy Source 1'],
                             columns='Status_Type',
                             values='Nameplate Capacity (MW)',
                             fill_value=0).reset_index()
    
    pivot.columns.name = None
    pivot = pivot.rename(columns={
        'Operable': 'Operable Capacity (MW)',
        'Proposed': 'Proposed Capacity (MW)',
        'Retired': 'Retired Capacity (MW)'
    })
    
    # Total capacity and numeric version
    pivot['Total Capacity (MW)'] = (
        pivot.get('Operable Capacity (MW)', 0) +
        pivot.get('Proposed Capacity (MW)', 0) -
        pivot.get('Retired Capacity (MW)', 0)
    )
    pivot["Total Capacity Num"] = pivot["Total Capacity (MW)"]
    
    # Format numeric columns
    for col in ['Operable Capacity (MW)', 'Proposed Capacity (MW)', 'Retired Capacity (MW)', 'Total Capacity (MW)']:
        if col in pivot.columns:
            pivot[col] = pivot[col].apply(lambda x: f"{x:,.1f}")
    
    return pivot

def clean_data_for_sector_chart(selected_states):
    """Clean and prepare data for sector sunburst chart."""
    filtered_df = get_filtered_df(selected_states)
    
    # Drop rows with missing values
    filtered_df = filtered_df.dropna(subset=['Status_Type', 'Sector Name', 'Nameplate Capacity (MW)'])
    
    # Ensure numeric
    filtered_df['Nameplate Capacity (MW)'] = pd.to_numeric(filtered_df['Nameplate Capacity (MW)'], errors='coerce').fillna(0)
    
    return filtered_df

def clean_data_for_ownership_chart(selected_states):
    """Clean and prepare data for ownership sunburst chart."""
    filtered_df = get_filtered_df(selected_states)
    
    # Drop rows with missing values
    filtered_df = filtered_df.dropna(subset=['Ownership', 'Status_Type', 'Nameplate Capacity (MW)'])
    
    # Map full descriptions to a new column for hover info
    ownership_mapping = {
        'S': 'Single ownership by respondent',
        'J': 'Jointly owned with another entity',
        'W': 'Wholly owned by an entity other than respondent'
    }
    filtered_df['Ownership_Description'] = filtered_df['Ownership'].map(ownership_mapping)
    
    # Ensure numeric
    filtered_df['Nameplate Capacity (MW)'] = pd.to_numeric(filtered_df['Nameplate Capacity (MW)'], errors='coerce').fillna(0)
    
    return filtered_df

def clean_data_for_fuel_type_sankey(selected_states):
    """Clean and prepare data for fuel type sankey chart."""
    filtered_df = get_filtered_df(selected_states)
    
    # Fuel Type mapping from image
    energy_to_fuel_type = {
        'ANT': 'Coal', 'BIT': 'Coal', 'LIG': 'Coal', 'SGC': 'Coal', 'SUB': 'Coal', 'WC': 'Coal', 'RC': 'Coal',
        'DFO': 'Petroleum', 'JF': 'Petroleum', 'KER': 'Petroleum', 'PC': 'Petroleum', 'PG': 'Petroleum',
        'RFO': 'Petroleum', 'SGP': 'Petroleum', 'WO': 'Petroleum',
        'BFG': 'Natural Gas and Other Gases', 'NG': 'Natural Gas and Other Gases', 'H2': 'Natural Gas and Other Gases', 'OG': 'Natural Gas and Other Gases',
        'AB': 'Solid Renewable Fuels', 'MSW': 'Solid Renewable Fuels', 'OBS': 'Solid Renewable Fuels', 'WDS': 'Solid Renewable Fuels',
        'OBL': 'Liquid Renewable Fuels', 'SLW': 'Liquid Renewable Fuels', 'BLQ': 'Liquid Renewable Fuels', 'WDL': 'Liquid Renewable Fuels',
        'LFG': 'Gaseous Renewable Fuels', 'OBG': 'Gaseous Renewable Fuels',
        'SUN': 'Other Renewable Fuels', 'WND': 'Other Renewable Fuels', 'GEO': 'Other Renewable Fuels', 'WAT': 'Other Renewable Fuels',
        'NUC': 'All Other Energy Sources', 'PUR': 'All Other Energy Sources', 'WH': 'All Other Energy Sources', 'TDF': 'All Other Energy Sources',
        'OTH': 'All Other Energy Sources', 'MWH': 'All Other Energy Sources'
    }
    
    # Map fuel types
    filtered_df['Fuel Type'] = filtered_df['Energy Source 1'].map(energy_to_fuel_type).fillna('Unknown')
    
    # Ensure numeric
    numeric_cols = ['Nameplate Capacity (MW)']
    filtered_df[numeric_cols] = filtered_df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Group capacity
    capacity_df = filtered_df.groupby(['State', 'Fuel Type', 'Status_Type'])['Nameplate Capacity (MW)'].sum().reset_index()
    total_capacity_df = capacity_df.groupby(['State', 'Fuel Type'])['Nameplate Capacity (MW)'].sum().reset_index()
    total_capacity_df['Status_Type'] = 'Total Capacity (MW)'
    
    final_df = pd.concat([capacity_df, total_capacity_df])
    
    # Create Node Labels
    all_labels = list(pd.concat([
        final_df['State'],
        final_df['Fuel Type'],
        final_df['Status_Type']
    ]).unique())
    label_map = {label: i for i, label in enumerate(all_labels)}
    
    # Map Source and Target
    final_df['source'] = final_df.apply(
        lambda x: label_map[x['State']] if x['Status_Type'] != 'Total Capacity (MW)' else label_map[x['Fuel Type']], 
        axis=1
    )
    final_df['target'] = final_df.apply(
        lambda x: label_map[x['Fuel Type']] if x['Status_Type'] != 'Total Capacity (MW)' else label_map['Total Capacity (MW)'], 
        axis=1
    )
    
    return final_df, all_labels


def clean_data_for_netgen_map(selected_states):
    """Clean and prepare data for net generation map."""
    filtered_df = get_filtered_netgen_df(selected_states)
    
    # Ensure numeric
    filtered_df['Net Generation Year To Date'] = pd.to_numeric(filtered_df['Net Generation Year To Date'], errors='coerce').fillna(0)
    
    # Group by state and status
    state_summary = filtered_df.groupby(['State', 'Status_Type']).agg({
        'Generator ID': 'count',
        'Net Generation Year To Date': 'sum',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()
    
    state_summary.rename(columns={'Generator ID': 'Generator Count'}, inplace=True)
    
    return state_summary

#-------------------------------------------------------------------
# Distribution of plants Map
#-------------------------------------------------------------------

def create_capacity_map(selected_states):
    """Create an interactive map displaying power plant locations based on Status_Type."""
    filtered_df = clean_data_for_capacity_map(selected_states)

    fig = px.scatter_mapbox(
        filtered_df,
        lat="Latitude",
        lon="Longitude",
        color="Status_Type",
        hover_name="Technology",
        hover_data=["Plant Code", "Utility ID", "State"],
        zoom=4,
        mapbox_style="carto-positron"
    )

    return ui.HTML(fig.to_html(full_html=False))

#-------------------------------------------------------------------
# Recommendation table
#-------------------------------------------------------------------

def calculate_summary_df(selected_states):
    """Calculate plant status and capacity summary based on the state filter."""
    filtered_df = clean_data_for_summary(selected_states)
 
    # Aggregate plant count
    plant_count = filtered_df.groupby(['State', 'Status_Type']).size().reset_index(name='Plant Count')
 
    # Aggregate capacities
    capacity_df = filtered_df.groupby(['State', 'Status_Type'])['Nameplate Capacity (MW)'].sum().reset_index()
 
    # Pivot for calculations
    pivot_capacity = capacity_df.pivot(index='State', columns='Status_Type', values='Nameplate Capacity (MW)').fillna(0)
    pivot_count = plant_count.pivot(index='State', columns='Status_Type', values='Plant Count').fillna(0)
 
    # Combine into a single summary dataframe
    summary_df = pd.DataFrame({
        'State': pivot_capacity.index,
        'Proposed Plants': pivot_count.get('Proposed', 0),
        'Retired Plants': pivot_count.get('Retired', 0),
        'Proposed Capacity (MW)': pivot_capacity.get('Proposed', 0).round(2),
        'Retired Capacity (MW)': pivot_capacity.get('Retired', 0).round(2),
    }).reset_index(drop=True)
 
    # Calculate metrics
    summary_df['Adjusted NPPI'] = summary_df['Proposed Plants'] - summary_df['Retired Plants']
    summary_df['Adjusted Net Capacity Change (MW)'] = (
        summary_df['Proposed Capacity (MW)'] - summary_df['Retired Capacity (MW)']
    ).round(2)
 
    # Calculate Recommendation Score
    summary_df['Score'] = summary_df.apply(
        lambda row: 0.5 * (row['Proposed Plants'] / (row['Retired Plants'] + 1)) + 
                  0.5 * (row['Proposed Capacity (MW)'] / (row['Retired Capacity (MW)'] + 1)),
        axis=1
    ).round(2)
 
    # Determine Recommendation
    summary_df['Recommendation'] = summary_df['Score'].apply(
        lambda score: 'Highly Favorable' if score >= 2 else
                     'Favorable' if score >= 1 else
                     'Neutral' if score >= 0.5 else
                     'Unfavorable'
    )
 
    # Format numeric columns with commas
    numeric_cols = ['Proposed Plants', 'Retired Plants', 'Proposed Capacity (MW)',
                   'Retired Capacity (MW)', 'Adjusted NPPI', 'Adjusted Net Capacity Change (MW)']
    for col in numeric_cols:
        summary_df[col] = summary_df[col].apply(lambda x: "{:,}".format(x) if isinstance(x, (int, float)) else x)
 
    return summary_df

#-------------------------------------------------------------------
# Nameplate and Seasonal Capacity treemap
#-------------------------------------------------------------------

def create_capacity_treemap(selected_states):
    """Create an interactive treemap displaying power generation capacity."""
    tech_status_capacity_df = clean_data_for_capacity_treemap(selected_states)

    # Create a Treemap
    treemap_fig = px.treemap(
        tech_status_capacity_df,
        path=['Status_Type', 'Technology'],
        values='Nameplate Capacity (MW)',
        color='Nameplate Capacity (MW)',
        color_continuous_scale='Blues',
        hover_data={
            'Technology': True,
            'Status_Type': True,
            'Nameplate Capacity (MW)': True,
            'Summer Capacity (MW)': True,
            'Winter Capacity (MW)': True,
        }
    )

    # Format hover data
    treemap_fig.update_traces(
        hovertemplate=(
            '<b>%{label}</b><br>' +
            'Technology: %{customdata[0]}<br>' +
            'Status Type: %{customdata[1]}<br>' +
            'Nameplate Capacity: %{customdata[2]:,}<br>' +
            'Summer Capacity: %{customdata[3]:,}<br>' +
            'Winter Capacity: %{customdata[4]:,}<br>'
        )
    )

    # Highlight technologies
    treemap_fig.update_traces(
        marker=dict(
            colorscale='Blues',
            line=dict(color=tech_status_capacity_df['Highlight'], width=2)
        )
    )

    return ui.HTML(treemap_fig.to_html(full_html=False))

#-------------------------------------------------------------------
# Technology distribution
#-------------------------------------------------------------------

def create_generator_count_treemap(selected_states):
    """Create a treemap showing the number of generators by Technology and Status_Type."""
    generator_counts = clean_data_for_generator_count(selected_states)
 
    # Plot
    fig = px.treemap(
        generator_counts,
        path=['Status_Type', 'Technology'],
        values='Generator Count',
        color='Generator Count',
        color_continuous_scale='Blues',
        hover_data={
            'Technology': True,
            'Status_Type': True,
            'Generator Count': True
        }
    )
 
    # Fix hovertemplate
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Status Type: %{customdata[1]}<br>Technology: %{customdata[0]}<br>Generator Count: %{customdata[2]:,}'
    )
 
    return ui.HTML(fig.to_html(full_html=False))

#-------------------------------------------------------------------
# Energy source Table
#-------------------------------------------------------------------

def create_energy_source_summary_table(selected_states):
    """Creates a sortable HTML table with sticky headers and sticky 'State' column."""
    pivot = clean_data_for_energy_source(selected_states)
    
    # Soft blue gradient
    soft_blues = px.colors.sequential.Blues[2:]
    max_val = pivot["Total Capacity Num"].max()
    min_val = pivot["Total Capacity Num"].min()
 
    def color_cell(value):
        index = int((value - min_val) / (max_val - min_val + 1e-6) * (len(soft_blues) - 1))
        color = soft_blues[index]
        return f'background-color: {color};'
 
    # Build HTML rows
    table_rows = ""
    for _, row in pivot.iterrows():
        row_html = "<tr>"
        for col in pivot.columns[:-1]:  # skip Total Capacity Num
            if col == "Total Capacity (MW)":
                row_html += f'<td style="{color_cell(row["Total Capacity Num"])}">{row[col]}</td>'
            else:
                row_html += f"<td>{row[col]}</td>"
        row_html += "</tr>"
        table_rows += row_html
 
    # Table headers
    headers = ''.join(f"<th>{col}</th>" for col in pivot.columns[:-1])  # skip Total Capacity Num
    
    # Return HTML table with JavaScript for sorting
    return ui.HTML(f"""
        <style>
            .energy-scroll-wrapper {{
                height: 400px;
                overflow: auto;
                border: 1px solid #ccc;
            }}
 
            #energy-table {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
            }}
 
            #energy-table thead th {{
                position: sticky;
                top: 0;
                background-color: #f8f9fa;
                z-index: 2;
                text-align: center;
                border: 1px solid #dee2e6;
                font-weight: bold;
            }}
 
            #energy-table th:first-child {{
                position: sticky;
                left: 0;
                background-color: #f8f9fa;
                z-index: 3;
                text-align: center;
            }}
            
            #energy-table td:first-child {{
                position: sticky;
                left: 0;
                background-color: #f8f9fa;
                z-index: 3;
                text-align: center;
                font-weight: normal;
            }}
            
            #energy-table thead th:first-child {{
                position: sticky;
                top: 0;
                left: 0;
                background-color: #f8f9fa;
                z-index: 4;
                font-weight: bold;
            }}
 
            #energy-table td {{
                border: 1px solid #dee2e6;
                text-align: center;
                padding: 4px;
                font-weight: normal;
            }}
        </style>
 
        <div class="energy-scroll-wrapper">
            <table id="energy-table" class="table table-bordered table-striped">
                <thead><tr>{headers}</tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
 
        <script>
            setTimeout(function() {{
                if (!$.fn.dataTable.isDataTable('#energy-table')) {{
                    var table = $('#energy-table').DataTable({{
                        paging: false,
                        info: false,
                        ordering: true
                    }});
 
                    function fixStickyHeader() {{
                        $('#energy-table thead th:first-child').css({{
                            position: 'sticky',
                            top: 0,
                            left: 0,
                            background: '#f8f9fa',
                            zIndex: 4,
                            fontWeight: 'bold'
                        }});
                        $('.dataTables_scrollHeadInner thead th:first-child').css({{
                            position: 'sticky',
                            top: 0,
                            left: 0,
                            background: '#f8f9fa',
                            zIndex: 4,
                            fontWeight: 'bold'
                        }});
                        
                        $('#energy-table tbody td:first-child').css({{
                            fontWeight: 'normal'
                        }});
                    }}
 
                    fixStickyHeader();
                    table.on('draw.dt', fixStickyHeader);
                }}
            }}, 300);
        </script>
    """)

#-------------------------------------------------------------------
# Sector for combined page
#-------------------------------------------------------------------

def create_sunburst_sector_chart(selected_states):
    """Create an interactive sunburst chart displaying power plant data by Status_Type and Sector Name."""
    filtered_df = clean_data_for_sector_chart(selected_states)

    # Define a custom dark blue color sequence
    dark_blues = ['#08306B', '#08519C', '#2171B5', '#4292C6', '#6BAED6', '#9ECAE1']

    # Create a sunburst chart
    fig = px.sunburst(
        filtered_df,
        path=['Status_Type', 'Sector Name'],
        values='Nameplate Capacity (MW)',
        color='Status_Type',
        color_discrete_sequence=dark_blues
    )

    # Return the chart as an HTML component
    return ui.HTML(fig.to_html(full_html=False))

#-------------------------------------------------------------------
# Ownership for combined page
#-------------------------------------------------------------------

def create_sunburst_ownership_chart(selected_states):
    """Create an interactive sunburst chart displaying power plant data by Ownership and Sector Name."""
    filtered_df = clean_data_for_ownership_chart(selected_states)
 
    # Define a custom dark blue color sequence
    dark_blues = ['#08306B', '#08519C', '#2171B5', '#4292C6', '#6BAED6', '#9ECAE1']
 
    # Create a sunburst chart using short labels but full descriptions in hover
    fig = px.sunburst(
        filtered_df,
        path=['Status_Type', 'Ownership'],
        values='Nameplate Capacity (MW)',
        color='Ownership',
        color_discrete_sequence=dark_blues,
        hover_data={'Ownership_Description': True, 'Ownership': False}  # Show only full description
    )
 
    # Return the chart as an HTML component
    return ui.HTML(fig.to_html(full_html=False))

#-------------------------------------------------------------------
# Sankey
#-------------------------------------------------------------------

def create_fuel_type_sankey_chart(selected_states):
    """Create a Sankey diagram showing flow from states to fuel types to status types."""
    final_df, all_labels = clean_data_for_fuel_type_sankey(selected_states)

    # Sankey plot
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color='lightblue'  # Node color: Blue
        ),
        link=dict(
            source=final_df['source'],
            target=final_df['target'],
            value=final_df['Nameplate Capacity (MW)'],
            color='rgba(31, 119, 180, 0.5)'  # Transparent Blue Links
        )
    ))

    return ui.HTML(fig.to_html(full_html=False))

#-------------------------------------------------------------------
# Net Gen map
#-------------------------------------------------------------------

def create_netgen_map(selected_states):
    """Generate an interactive map visualizing power plant locations, colored by Net Generation Year To Date."""
    state_summary = clean_data_for_netgen_map(selected_states)

    # Create map
    fig = px.scatter_geo(
        state_summary,
        lat='Latitude',
        lon='Longitude',
        scope='usa',
        size='Generator Count',
        color='Net Generation Year To Date',
        color_continuous_scale=px.colors.sequential.Viridis,
        hover_name='State',
        hover_data={
            'Generator Count': True,
            'Net Generation Year To Date': True,
            'Status_Type': True,
            'Latitude': False,
            'Longitude': False
        }
    )

    # Format hover text
    fig.update_traces(
        hovertemplate=(
            '<b>%{hovertext}</b><br>' + 
            'Generator Count: %{customdata[0]:,}<br>' +
            'Net Generation YTD: %{customdata[1]:,} MWh<br>' +
            'Status Type: %{customdata[2]}<br>' +
            '<extra></extra>'
        ),
        customdata=state_summary[['Generator Count', 'Net Generation Year To Date', 'Status_Type']].values
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(coloraxis_colorbar=dict(title="Net Generation Capacity"))

    return ui.HTML(fig.to_html(full_html=False))
spinner_html = ui.tags.div(
    {"id": "custom-spinner-overlay"},
    ui.tags.div(
        ui.tags.div(*[ui.tags.div() for _ in range(12)], class_="lds-spinner"),
        ui.tags.div("", id="spinner-message", style="color:white; margin-top:15px; font-size: 18px;"),
        class_="spinner-wrapper"
    )
)
#----------------------------------------------SideBar and Styling------------------------------------------------------------------------------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.p("Select State:", class_="select-state-title"),
            ui.div(
                ui.input_checkbox_group(
                    "state_filter",
                    "",
                    choices=states,
                    selected=["CA"]
                ),
                class_="state-filter-scroll"
            )
        ),
        ui.tags.script("""
        document.addEventListener("DOMContentLoaded", function() {
            const stateMap = {
                'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
                'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
                'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
                'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
                'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
                'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
                'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
                'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
                'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
                'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
                'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
                'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
                'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia', 'PR': 'Puerto Rico'
            };
            
            setTimeout(() => {
                document.querySelectorAll('#state_filter label').forEach(label => {
                    const abbr = label.textContent.trim();
                    if (stateMap[abbr]) {
                        label.setAttribute('title', stateMap[abbr]);
                    }
                });
            }, 500); // wait for UI to load
        });
        """),

        ui.div(
            ui.input_action_button(
                "clear_states",
                "Clear All",
                class_="btn btn-secondary me-2"
            ),
            ui.input_action_button(
                "select_all_states",
                "Select All",
                class_="btn btn-primary"
            ),
            class_="d-flex"
        ),

        ui.input_radio_buttons(
            "sheet_select",
            ui.p("Select Sheet:", class_="input-section-title"),
            choices=["Operable", "Proposed", "Retired", "Combined Analysis"],
            selected="Operable",
        ),
        
        tags.style("""
        #state_filter {
            max-height: none !important;
            overflow: visible !important;
        }
        """),
        
        tags.style("""
        /* State filter styles with bluish-grayish theme */
        #state_filter label {
            display: inline-block;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: pointer;
            padding: 5px;
            border-radius: 6px;
            color: #2c3e50;
            margin: 3px;
            background-color: #e2e8f0;
        }
        
        .select-state-title {
            position: sticky;
            top: 0;
            background-color: #203e5a;
            font-weight: bold;
            font-size: 1.1rem;
            padding: 6px 8px;
            z-index: 10;
            margin-bottom: 0;
            color: #f5f8fb;
            border-radius: 4px 4px 0 0;
        }

        .input-section-title {
            font-size: 1.1rem;
            font-weight: bold;
            color: #f5f8fb;
            margin-bottom: 6px;
            background-color: #203e5a;
            padding: 6px 8px;
            border-radius: 4px;
        } 
                
        .state-filter-scroll {
            max-height: 250px;
            overflow-y: auto;
            background-color: #ECF1F5;
            padding: 8px;
            border-radius: 6px;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        }

        #state_filter label:hover {
            transform: perspective(300px) translateZ(5px) scale(1.05);
            box-shadow: 0 4px 8px rgba(44, 62, 80, 0.15);
            background-color: #b8c6d9;
        }
        
        #state_filter input[type="checkbox"]:checked + span {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .shiny-input-container {
            margin-bottom: 15px;
        }
        
        /* Sidebar styling */
        .sidebar {
            background-color: #e2e8f0;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            padding-top: 15px;
        }
        #custom-spinner-overlay {
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background-color: rgba(0, 0, 0, 0.6);
            z-index: 9999;
            display: none;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }

        .spinner-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .lds-spinner {
            display: inline-block;
            position: relative;
            width: 80px;
            height: 80px;
        }
        .lds-spinner div {
            transform-origin: 40px 40px;
            animation: lds-spinner 1.2s linear infinite;
        }
        .lds-spinner div:after {
            content: " ";
            display: block;
            position: absolute;
            top: 3px;
            left: 37px;
            width: 6px;
            height: 18px;
            border-radius: 20%;
            background: #00aaff;
        }
        .lds-spinner div:nth-child(1)  { transform: rotate(0deg);   animation-delay: -1.1s; }
        .lds-spinner div:nth-child(2)  { transform: rotate(30deg);  animation-delay: -1.0s; }
        .lds-spinner div:nth-child(3)  { transform: rotate(60deg);  animation-delay: -0.9s; }
        .lds-spinner div:nth-child(4)  { transform: rotate(90deg);  animation-delay: -0.8s; }
        .lds-spinner div:nth-child(5)  { transform: rotate(120deg); animation-delay: -0.7s; }
        .lds-spinner div:nth-child(6)  { transform: rotate(150deg); animation-delay: -0.6s; }
        .lds-spinner div:nth-child(7)  { transform: rotate(180deg); animation-delay: -0.5s; }
        .lds-spinner div:nth-child(8)  { transform: rotate(210deg); animation-delay: -0.4s; }
        .lds-spinner div:nth-child(9)  { transform: rotate(240deg); animation-delay: -0.3s; }
        .lds-spinner div:nth-child(10) { transform: rotate(270deg); animation-delay: -0.2s; }
        .lds-spinner div:nth-child(11) { transform: rotate(300deg); animation-delay: -0.1s; }
        .lds-spinner div:nth-child(12) { transform: rotate(330deg); animation-delay: 0s; }
        @keyframes lds-spinner {
            0% { opacity: 1; }
            100% { opacity: 0; }
        }

        """)
        
    ),
    ui.div(
        ui.h1("Data is Power - 2023", class_="app-title"),
        spinner_html,
        ui.output_ui("dashboard_output"),
        class_="main-content-area"
    ),

    tags.head(
        tags.link(
            href="https://cdnjs.cloudflare.com/ajax/libs/bootswatch/5.3.2/flatly/bootstrap.min.css",
            rel="stylesheet",
        ),
        tags.script(src="https://code.jquery.com/jquery-3.6.0.min.js"),
        tags.script("""
            document.addEventListener("DOMContentLoaded", function () {
                const spinner = document.getElementById('custom-spinner-overlay');
                const messageBox = document.getElementById("spinner-message");
                const inputs = document.querySelectorAll('input, select');

                const messages = [
                    "Please wait, loading…",
                    "Fetching data, hang tight!",
                    "Loading, just a moment…",
                    "Almost there, please wait…",
                    "Processing your request…",
                    "Preparing awesome content for you…",
                    "Hold on, we’re getting things ready!",
                    "Loading… Good things take time!",
                    "Sit tight, setting things up…",
                    "One moment, we’re working on it…"
                ];

                function showSpinnerWithMessage() {
                    const msg = messages[Math.floor(Math.random() * messages.length)];
                    messageBox.textContent = msg;
                    spinner.style.display = "flex";
                }

                //  Show on page load
                showSpinnerWithMessage();

                //  Show on any filter input change
                inputs.forEach(el => {
                    el.addEventListener('change', showSpinnerWithMessage);
                });

                //  ALSO show on Clear All / Select All button clicks
                const clearBtn = document.querySelector('#clear_states');
                const selectAllBtn = document.querySelector('#select_all_states');

                if (clearBtn) clearBtn.addEventListener('click', showSpinnerWithMessage);
                if (selectAllBtn) selectAllBtn.addEventListener('click', showSpinnerWithMessage);

                //  Hide when dashboard updates
                new MutationObserver(() => {
                    spinner.style.display = "none";
                }).observe(document.getElementById('dashboard_output'), { childList: true });
            });
        """),
        tags.script(src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"),
        tags.link(
            href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css",
            rel="stylesheet"
        ),
        tags.style("""
        /* Global styles with bluish-grayish theme */
        body {
            background-color: #edf2f7;
            color: #2c3e50;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .app-title {
            color: #f5f8fb;
            font-weight: 600;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #203e5a;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .main-content-area {
            background-color: #d1dce7;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 15px;
        }
        
        .scrollable-table {
            max-height: 400px;
            overflow-y: auto;
            overflow-x: auto;
            border: 1px solid #b8c6d9;
            border-radius: 6px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            background-color: #f8fafc;
        }
        
        .scrollable-table table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .scrollable-table th, .scrollable-table td {
            border: 1px solid #cbd5e0;
            padding: 8px;
            text-align: left;
        }
        
        .scrollable-table thead th {
            position: sticky;
            top: 0;
            background-color: #4a5568;
            color: white;
            z-index: 1;
        }
        
        .scrollable-table thead tr:nth-child(2) th,
        .scrollable-table thead tr:nth-child(2) td {
            top: 110px;
            background-color: #718096;
            color: white;
        }
        
        .scrollable-table th:first-child,
        .scrollable-table td:first-child {
            position: sticky;
            left: 0;
            background-color: #edf2f7;
            z-index: 2;
        }
        
        .scrollable-table thead th:first-child {
            background-color: #4a5568;
            z-index: 3;
        }
        
        /* Button styling */
        .btn-primary {
            background-color: #4a5568 !important;
            border-color: #4a5568 !important;
            color: white !important;
            transition: all 0.3s ease;
        }
        
        .btn-primary:hover {
            background-color: #2d3748 !important;
            border-color: #2d3748 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        .btn-secondary {
            background-color: #718096 !important;
            border-color: #718096 !important;
            color: white !important;
            transition: all 0.3s ease;
        }
        
        .btn-secondary:hover {
            background-color: #4a5568 !important;
            border-color: #4a5568 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Custom radio buttons and checkboxes */
        input[type="radio"], input[type="checkbox"] {
            accent-color: #4a5568;
        }
        
        /* Cards and containers */
        .card {
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: #f8fafc;
            border: none;
            margin-bottom: 20px;
        }
        
        .card-header {
            background-color: #b8c6d9;
            color: #2c3e50;
            font-weight: 600;
            border-radius: 8px 8px 0 0 !important;
            border-bottom: none;
        }
        
        /* Page content container */
        .page-content-container {
            min-height: 100vh;
        }
        """)
    ),
    
    # Main container styling
    style="background-color: #203e5a; color: #2c3e50; min-height: 100vh;"
)

def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.clear_states, input.select_all_states)
    def handle_state_selection():
        if input.clear_states():
            ui.update_checkbox_group("state_filter", selected=[])
        elif input.select_all_states():
            ui.update_checkbox_group("state_filter", selected=states)

    @reactive.Calc
    def filtered_df():
        selected_states = input.state_filter()
        selected_sheet = input.sheet_select()
        sheet_mapping = {
            "Operable": df_operable,
            "Proposed": df_proposed,
            "Retired": df_retired,
            "Combined Analysis": df_cleaned
        }
        df = sheet_mapping.get(selected_sheet, df_cleaned)
        return df if not selected_states else df[df["State"].isin(selected_states)]

    @reactive.Calc
    def current_sheet():
        return input.sheet_select()

    def create_flex_row(*components, max_width="100%"):
        return ui.div(*components, class_="d-flex", style=f"width: {max_width}; margin: 0; padding: 0; align-items: stretch;")

    def create_section(title, component, max_width="98.3%"):
        return ui.div(
            ui.h5(title),
            ui.div(component, class_="chart-container", style=f"max-width: {max_width};"),
            class_="d-flex flex-column"
        )

    def create_chart_container(component, flex=1, max_width="50%"):
        return ui.div(component, style=f"flex: {flex}; padding: 0; min-width: 0; max-width: {max_width}")

    def create_metric_cards(df, selected_states, include_extra=False):
        metrics = calculate_metrics(df, selected_states, operable=include_extra)
        cards = [
            create_card(metrics[0], "Total Utilities", "#6495ED"),
            create_card(metrics[1], "Unique Plant Codes", "#4169E1"),
            create_card(metrics[2], "Unique Generators", "#1E90FF")
        ]
        if include_extra:
            extra = [
                create_card(metrics[i], label, "#00BFFF") for i, label in enumerate([
                    "Uprates/Derates", "Planned Uprates", "Planned Derates", "Planned Repowers"], start=3)
            ]
            cards.extend(extra)
        return ui.div(*cards, class_="d-flex flex-wrap")

    def layout_factory(components, layout_type):
        common = [components["metrics"], components["capacity_chart"], components["tech_dist_plot"]]
        pair_row = create_flex_row(
            create_chart_container(components["status_by_sector"]),
            create_chart_container(components["tech_distribution"])
        )

        if layout_type == "operable":
            triple_row = create_flex_row(
                create_chart_container(components["uprate_chart"], max_width="33.3%"),
                create_chart_container(components["derate_chart"], max_width="33.3%"),
                create_chart_container(components["repower_chart"], max_width="33.3%")
            )
            return ui.div(
                *common,
                components["planned_retirement_chart"],
                pair_row,
                triple_row,
                components["capacity_added_chart"],
                ui.h5("Age of Technology", class_="mt-4 mb-2"),
                ui.div(components["capacity_by_technology_table"], class_="scrollable-table")
            )

        elif layout_type == "proposed":
            return ui.div(*common, pair_row, components["effective_date_chart"])

        elif layout_type == "retired":
            return ui.div(*common, pair_row, components["retirement_year_chart"])

    def create_combined_analysis_layout(components):
        return ui.div(
            create_section("Geographic Distribution of Power Plants (Operable, Retired, and Proposed)", components["capacity_map"]),
            ui.div(components["summary_table"], class_="d-flex flex-column"),
            create_section("Distribution of Nameplate, Summer, Winter Capacity", components["capacity_treemap"]),
            create_section("Distribution of Technology", components["generator_count_treemap"]),
            create_section("Distribution of Sector", components["sunburst_sector_chart"]),
            create_section("Distribution of Ownership", components["sunburst_ownership_chart"]),
            create_section("Energy Source-wise Capacity Summary", components["energy_source_summary_table"]),
            create_section("Fuel Type-wise Capacity Summary", components["fuel_type_sankey_chart"]),
            create_section("Distribution of Net Generation Capacity", components["netgen_map"])
        )
    # Create summary table with optimized rendering
    def create_summary_table(selected_states):
        summary_df = calculate_summary_df(selected_states)
        col_index = summary_df.columns.get_loc("Recommendation")

        # Pre-process colors for recommendations
        color_map = {
            'Highly Favorable': 'background-color: #228B22; color: white; font-weight: bold;',
            'Favorable': 'background-color: #b0e57c;',
            'Neutral': 'background-color: #fff8b8;',
            'Unfavorable': 'background-color: #f4cccc;'
        }
        
        # Generate HTML rows more efficiently
        headers = ''.join(f'<th>{col}</th>' for col in summary_df.columns)
        
        rows = []
        for _, row in summary_df.iterrows():
            cells = []
            for col in summary_df.columns:
                if col == "Recommendation":
                    style = color_map.get(row[col], '')
                    cells.append(f'<td style="{style}">{row[col]}</td>')
                else:
                    cells.append(f'<td>{row[col]}</td>')
            rows.append(f"<tr>{''.join(cells)}</tr>")
            
        rows_html = ''.join(rows)

        # Create summary table with optimized JS
        html = f"""
        <style>
            #summary-table thead th {{
                position: sticky;
                top: 0;
                background-color: #f1f1f1;
                z-index: 2;
            }}
            #summary-table thead th:last-child::after,
            #summary-table thead th:last-child::before {{
                display: none !important;
            }}
            div.dataTables_filter, div.dataTables_length {{
                display: none;
            }}
        </style>

        <div style="display: flex; align-items: center; gap: 10px; border-bottom: 1px solid white; padding-bottom: 4px; margin-bottom: 10px;">
            <h5 style="margin: 0;">Suitability for Future Energy Projects</h5>
            <button type="button" class="btn btn-outline-info btn-sm" id="info-icon" title="Click to view formula explanation">
                ℹ️
            </button>
        </div>

        <div style="margin-bottom: 10px;">
            <label for="rec-filter" style="margin-right: 10px; font-weight: bold;">Filter by Recommendation:</label>
            <select id="rec-filter" style="width: 220px; padding: 4px;">
                <option value="">All</option>
                <option value="Highly Favorable">Highly Favorable</option>
                <option value="Favorable">Favorable</option>
                <option value="Neutral">Neutral</option>
                <option value="Unfavorable">Unfavorable</option>
            </select>
        </div>

        <div style="height: 400px; overflow-y: auto; border: 1px solid #ccc;">
            <table id="summary-table" class="table table-striped table-bordered compact" style="width:100%; text-align:center;">
                <thead><tr>{headers}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>

        <!-- Modal -->
        <div id="info-modal" style="display: none; position: fixed; top: 15%; left: 50%; transform: translateX(-50%);
            background: white; border: 1px solid #ccc; border-radius: 10px; padding: 20px; width: 600px; z-index: 9999; box-shadow: 0px 0px 10px rgba(0,0,0,0.2);">
            <h5><strong>Recommendation Score Formula</strong></h5>
            <img src="https://latex.codecogs.com/png.image?\dpi{{150}}&space;Score%3D0.5%5Cleft(%5Cfrac%7B%5Ctext%7BProposed%20Plants%7D%7D%7B%5Ctext%7BRetired%20Plants%7D%2B1%7D%5Cright)%2B0.5%5Cleft(%5Cfrac%7B%5Ctext%7BProposed%20Capacity%7D%7D%7B%5Ctext%7BRetired%20Capacity%7D%2B1%7D%5Cright)" alt="Score Formula" style="max-width: 100%; height: auto; margin-bottom: 10px;" />

            <p><strong>Adjusted NPPI:</strong> Proposed Plants − Retired Plants</p>
            <p><strong>Adjusted Net Capacity Change (MW):</strong> Proposed Capacity − Retired Capacity</p>

            <p><strong>Recommendation Categories:</strong></p>
            <ul>
                <li><strong>Highly Favorable</strong>: Score ≥ 2.0 — Strong growth and expansion.</li>
                <li><strong>Favorable</strong>: 1.0 ≤ Score &lt; 2.0 — Balanced or moderate growth.</li>
                <li><strong>Neutral</strong>: 0.5 ≤ Score &lt; 1.0 — Little or no change.</li>
                <li><strong>Unfavorable</strong>: Score &lt; 0.5 — More retirements than proposals.</li>
            </ul>
            <button onclick="document.getElementById('info-modal').style.display='none'" style="margin-top:10px;">Close</button>
        </div>

        <script>
            setTimeout(function() {{
                // Check if the DataTable is already initialized before initializing it again
                if (!$.fn.dataTable.isDataTable('#summary-table')) {{
                    var table = $('#summary-table').DataTable({{
                        paging: false,
                        info: false,
                        ordering: true,
                        columnDefs: [
                            {{ targets: [{col_index}], orderable: false }}
                        ]
                    }});

                    $('#rec-filter').on('change', function() {{
                        var val = $(this).val();
                        if (val === "") {{
                            table.column({col_index}).search("").draw();
                        }} else {{
                            var escapedVal = val.replace(/[-\/\\^$*+?.()|[\]{{}}]/g, '\\\\$&');
                            table.column({col_index}).search('^' + escapedVal + '$', true, false).draw();
                        }}
                    }});

                    $('#info-icon').on('click', function() {{
                        $('#info-modal').fadeIn();
                    }});
                }}
            }}, 300);
        </script>
        """

        return ui.HTML(html)


    @output
    @render.ui
    def dashboard_output():
        df = filtered_df()
        selected_states = input.state_filter()
        sheet = current_sheet()

        if sheet == "Operable":
            components = {
                "metrics": create_metric_cards(df, selected_states, include_extra=True),
                "capacity_chart": create_capacity_chart(df, selected_states),
                "tech_dist_plot": create_technology_distribution_plot(df, selected_states),
                "status_by_sector": create_status_by_sector_chart(df, selected_states),
                "tech_distribution": create_other_technology_distribution_chart(df, selected_states),
                "uprate_chart": create_uprate_chart(df, selected_states),
                "derate_chart": create_derate_chart(df, selected_states),
                "repower_chart": create_repower_chart(df, selected_states),
                "planned_retirement_chart": create_planned_retirement_chart(df, selected_states),
                "capacity_added_chart": create_capacity_added_plot(df, selected_states),
                "capacity_by_technology_table": create_capacity_by_technology_table(df, selected_states)
            }
            return layout_factory(components, "operable")

        elif sheet == "Proposed":
            components = {
                "metrics": create_metric_cards(df, selected_states),
                "capacity_chart": create_capacity_chart(df, selected_states),
                "tech_dist_plot": create_technology_distribution_plot(df, selected_states),
                "status_by_sector": create_status_by_sector_chart(df, selected_states),
                "tech_distribution": create_other_technology_distribution_chart(df, selected_states),
                "effective_date_chart": create_effective_date_chart(df, selected_states)
            }
            return layout_factory(components, "proposed")

        elif sheet == "Retired":
            components = {
                "metrics": create_metric_cards(df, selected_states),
                "capacity_chart": create_capacity_chart(df, selected_states),
                "tech_dist_plot": create_technology_distribution_plot(df, selected_states),
                "status_by_sector": create_status_by_sector_chart(df, selected_states),
                "tech_distribution": create_other_technology_distribution_chart(df, selected_states),
                "retirement_year_chart": create_retirement_year_combo_chart(df, selected_states)
            }
            return layout_factory(components, "retired")

        elif sheet == "Combined Analysis":
            components = {
                "capacity_map": create_capacity_map(selected_states),
                "summary_table": create_summary_table(selected_states),
                "capacity_treemap": create_capacity_treemap(selected_states),
                "generator_count_treemap": create_generator_count_treemap(selected_states),
                "sunburst_sector_chart": create_sunburst_sector_chart(selected_states),
                "sunburst_ownership_chart": create_sunburst_ownership_chart(selected_states),
                "energy_source_summary_table": create_energy_source_summary_table(selected_states),
                "fuel_type_sankey_chart": create_fuel_type_sankey_chart(selected_states),
                "netgen_map": create_netgen_map(selected_states)
            }
            return create_combined_analysis_layout(components)

        return ui.div("Invalid sheet selection")

app = App(app_ui, server, debug=True)
