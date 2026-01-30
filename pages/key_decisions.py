import streamlit as st
from utils.auth import require_authentication
import pandas as pd
from datetime import datetime, timedelta
import json

# Require authentication
require_authentication()

st.title("📅 Key Decisions & Milestones")
st.markdown("### Timeline of significant fund events and strategic decisions")

# Sample milestone data
if 'milestones' not in st.session_state:
    st.session_state.milestones = [
        {
            "date": "2024-01-15",
            "title": "Series B Investment in TechCorp",
            "category": "Investment",
            "description": "Led $50M Series B round in TechCorp, taking 15% equity stake",
            "impact": "High"
        },
        {
            "date": "2024-02-01",
            "title": "New Partner Onboarding",
            "category": "Team",
            "description": "Welcomed Jane Smith as Managing Partner, bringing 20 years of fintech experience",
            "impact": "Medium"
        },
        {
            "date": "2024-03-10",
            "title": "Portfolio Company Exit",
            "category": "Exit",
            "description": "Successfully exited DataAnalytics Inc. with 3.5x return on investment",
            "impact": "High"
        },
        {
            "date": "2024-04-05",
            "title": "Strategy Pivot",
            "category": "Strategy",
            "description": "Shifted focus to AI/ML startups in healthcare sector",
            "impact": "High"
        },
        {
            "date": "2024-05-20",
            "title": "Compliance Filing",
            "category": "Regulatory",
            "description": "Completed SEC Form ADV annual update",
            "impact": "Low"
        }
    ]

# Add new milestone section
st.subheader("Add New Milestone")
with st.expander("➕ Add Milestone"):
    col1, col2 = st.columns(2)
    
    with col1:
        new_date = st.date_input("Date", datetime.now())
        new_title = st.text_input("Title")
        new_category = st.selectbox("Category", ["Investment", "Exit", "Team", "Strategy", "Regulatory", "Other"])
    
    with col2:
        new_impact = st.selectbox("Impact", ["High", "Medium", "Low"])
        new_description = st.text_area("Description")
    
    if st.button("Add Milestone"):
        if new_title and new_description:
            new_milestone = {
                "date": new_date.strftime("%Y-%m-%d"),
                "title": new_title,
                "category": new_category,
                "description": new_description,
                "impact": new_impact
            }
            st.session_state.milestones.append(new_milestone)
            st.success("Milestone added successfully!")
            st.rerun()
        else:
            st.error("Please fill in all fields")

# Filter options
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    filter_category = st.multiselect(
        "Filter by Category",
        options=["All"] + list(set(m["category"] for m in st.session_state.milestones)),
        default=["All"]
    )

with col2:
    filter_impact = st.multiselect(
        "Filter by Impact",
        options=["All", "High", "Medium", "Low"],
        default=["All"]
    )

with col3:
    sort_order = st.selectbox(
        "Sort by",
        options=["Date (Newest First)", "Date (Oldest First)", "Impact"],
        index=0
    )

# Filter milestones
filtered_milestones = st.session_state.milestones.copy()

if "All" not in filter_category:
    filtered_milestones = [m for m in filtered_milestones if m["category"] in filter_category]

if "All" not in filter_impact:
    filtered_milestones = [m for m in filtered_milestones if m["impact"] in filter_impact]

# Sort milestones
if sort_order == "Date (Newest First)":
    filtered_milestones.sort(key=lambda x: x["date"], reverse=True)
elif sort_order == "Date (Oldest First)":
    filtered_milestones.sort(key=lambda x: x["date"])
else:  # Sort by impact
    impact_order = {"High": 0, "Medium": 1, "Low": 2}
    filtered_milestones.sort(key=lambda x: impact_order[x["impact"]])

# Display timeline
st.markdown("---")
st.subheader(f"Timeline ({len(filtered_milestones)} milestones)")

for milestone in filtered_milestones:
    with st.container():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown(f"**{milestone['date']}**")
            
            # Impact badge
            impact_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            st.markdown(f"{impact_colors[milestone['impact']]} {milestone['impact']} Impact")
        
        with col2:
            st.markdown(f"### {milestone['title']}")
            st.markdown(f"**Category:** {milestone['category']}")
            st.markdown(milestone['description'])
        
        st.markdown("---")

# Statistics
st.subheader("Milestone Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Milestones", len(st.session_state.milestones))

with col2:
    high_impact = len([m for m in st.session_state.milestones if m["impact"] == "High"])
    st.metric("High Impact Events", high_impact)

with col3:
    categories = len(set(m["category"] for m in st.session_state.milestones))
    st.metric("Categories", categories)

with col4:
    # Calculate events in last 90 days
    ninety_days_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    recent_events = len([m for m in st.session_state.milestones if m["date"] >= ninety_days_ago])
    st.metric("Last 90 Days", recent_events)

# Export milestones
st.markdown("---")
if st.button("Export Milestones as JSON"):
    json_data = json.dumps(st.session_state.milestones, indent=2)
    st.download_button(
        label="Download JSON",
        data=json_data,
        file_name=f"milestones_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    ) 