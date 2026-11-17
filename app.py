import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗", layout="wide")


# ---------- load model, data and metrics (cached so they load only once) ----------
@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("cars_clean_sample.csv")


@st.cache_data
def load_metrics():
    with open("model_metrics.json") as f:
        return json.load(f)


model = load_model()
df = load_data()
metrics = load_metrics()


def options(col):
    return sorted(df[col].dropna().astype(str).unique().tolist())


def predict_price(car):
    input_data = pd.DataFrame({k: [v] for k, v in car.items()})
    return float(model.predict(input_data)[0])


# ---------- header ----------
html_temp = """
<div style="background-color:#1f4e79;padding:14px;border-radius:10px">
<h2 style="color:white;text-align:center;margin:0">🚗 Used Car Price Prediction</h2>
<p style="color:#dbe9f6;text-align:center;margin:4px 0 0">Craigslist Cars &amp; Trucks · CRISP-DM project · CDS-4003</p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.write("")

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
f_make = st.sidebar.multiselect("Manufacturer", options("manufacturer"))
f_type = st.sidebar.multiselect("Type", options("type"))
f_cond = st.sidebar.multiselect("Condition", options("condition"))
f_price = st.sidebar.slider("Price range ($)", 1000, 100000, (1000, 100000), step=1000)

view = df.copy()
if f_make:
    view = view[view["manufacturer"].isin(f_make)]
if f_type:
    view = view[view["type"].isin(f_type)]
if f_cond:
    view = view[view["condition"].isin(f_cond)]
view = view[view["price"].between(*f_price)]

# ---------- KPI cards ----------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Cars (sample)", f"{len(view):,}")
k2.metric("Avg price", f"${view['price'].mean():,.0f}" if len(view) else "-")
k3.metric("Median price", f"${view['price'].median():,.0f}" if len(view) else "-")
k4.metric("Avg mileage", f"{view['odometer'].mean():,.0f} mi" if len(view) else "-")
k5.metric("Avg year", f"{view['year'].mean():.0f}" if len(view) else "-")

tab_pred, tab_price, tab_car, tab_geo, tab_model, tab_data = st.tabs(
    ["🔮 Predict price", "💰 Price analysis", "🚙 Car features", "🗺️ States", "📈 Model results", "📋 Data"])

# ---------- 1) prediction ----------
with tab_pred:
    st.subheader("Enter the car details to predict its price")
    c1, c2, c3 = st.columns(3)
    with c1:
        year = st.number_input("Year", min_value=1990, max_value=2025, value=2016)
        odometer = st.number_input("Odometer (miles)", min_value=100, max_value=400000, value=60000, step=1000)
        manufacturer = st.selectbox("Manufacturer", options("manufacturer"),
                                    index=options("manufacturer").index("toyota"))
        cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12], index=3)
    with c2:
        condition = st.selectbox("Condition", options("condition"))
        fuel = st.selectbox("Fuel", options("fuel"), index=options("fuel").index("gas"))
        transmission = st.selectbox("Transmission", options("transmission"))
        drive = st.selectbox("Drive", options("drive"))
        seller_type = st.selectbox("Seller type", options("seller_type"))
    with c3:
        car_type = st.selectbox("Type", options("type"))
        title_status = st.selectbox("Title status", options("title_status"))
        paint_color = st.selectbox("Paint color", options("paint_color"))
        state = st.selectbox("State", options("state"))

    if st.button("Predict", type="primary"):
        car = {"year": year, "odometer": odometer, "manufacturer": manufacturer,
               "condition": condition, "cylinders": cylinders, "fuel": fuel,
               "title_status": title_status, "transmission": transmission, "drive": drive,
               "type": car_type, "paint_color": paint_color, "state": state,
               "seller_type": seller_type}
        price = predict_price(car)
        mae = metrics["final"]["MAE"]
        st.success(f"Estimated price: **${price:,.0f}**")
        st.caption(f"Average model error (MAE): about ±${mae:,.0f}. "
                   f"Suggested range: ${max(price - mae, 0):,.0f} – ${price + mae:,.0f}")

        similar = df[(df["manufacturer"] == manufacturer) & (df["year"].between(year - 1, year + 1))]
        if len(similar) > 5:
            fig = px.histogram(similar, x="price", nbins=30,
                               title=f"Real prices of similar {manufacturer} cars ({year - 1}–{year + 1})")
            fig.add_vline(x=price, line_dash="dash", line_color="red", annotation_text="prediction")
            st.plotly_chart(fig, use_container_width=True)

# ---------- 2) price analysis ----------
with tab_price:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(view, x="price", nbins=50, title="Price distribution"),
                    use_container_width=True)
    by_year = view.groupby("year", as_index=False)["price"].mean()
    c2.plotly_chart(px.line(by_year, x="year", y="price", markers=True, title="Average price by year"),
                    use_container_width=True)
    c3, c4 = st.columns(2)
    smp = view.sample(min(3000, len(view)), random_state=1) if len(view) else view
    c3.plotly_chart(px.scatter(smp, x="odometer", y="price", color="condition", opacity=.5,
                               title="Price vs mileage"), use_container_width=True)
    top = view["manufacturer"].value_counts().head(15).index
    c4.plotly_chart(px.box(view[view["manufacturer"].isin(top)], x="manufacturer", y="price", points=False,
                           title="Price by manufacturer (top 15)"), use_container_width=True)

# ---------- 3) car features ----------
with tab_car:
    c1, c2 = st.columns(2)
    cnt = view["manufacturer"].value_counts().head(15).reset_index()
    c1.plotly_chart(px.bar(cnt, x="manufacturer", y="count", title="Most listed manufacturers"),
                    use_container_width=True)
    c2.plotly_chart(px.pie(view, names="type", title="Type share"), use_container_width=True)
    c3, c4 = st.columns(2)
    cond = view.groupby("condition", as_index=False)["price"].mean()
    c3.plotly_chart(px.bar(cond, x="condition", y="price", title="Average price by condition"),
                    use_container_width=True)
    seller = view.groupby("seller_type", as_index=False)["price"].mean()
    c4.plotly_chart(px.bar(seller, x="seller_type", y="price", title="Average price: dealer vs owner"),
                    use_container_width=True)

# ---------- 4) states ----------
with tab_geo:
    geo = view.groupby("state", as_index=False).agg(avg_price=("price", "mean"), cars=("price", "size"))
    geo["state"] = geo["state"].str.upper()
    st.plotly_chart(px.choropleth(geo, locations="state", locationmode="USA-states", color="avg_price",
                                  scope="usa", hover_data=["cars"], color_continuous_scale="Blues",
                                  title="Average price by state"), use_container_width=True)
    top_states = geo.sort_values("cars", ascending=False).head(20)
    st.plotly_chart(px.bar(top_states, x="state", y="avg_price", hover_data=["cars"],
                           title="Average price in the 20 states with the most cars"), use_container_width=True)

# ---------- 5) model results ----------
with tab_model:
    st.subheader("Model comparison (test set)")
    comp = pd.DataFrame(metrics["comparison"]).set_index("model")
    st.dataframe(comp.style.format("{:,.3f}"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(comp.reset_index(), x="model", y="R2_test", title="R² on test (higher is better)"),
                    use_container_width=True)
    c2.plotly_chart(px.bar(comp.reset_index(), x="model", y="MAE_test", title="MAE on test in $ (lower is better)"),
                    use_container_width=True)
    f = metrics["final"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Final model R²", f"{f['R2']:.3f}")
    m2.metric("Final model MAE", f"${f['MAE']:,.0f}")
    m3.metric("Final model RMSE", f"${f['RMSE']:,.0f}")
    st.caption(f"Raw rows: {metrics['rows_raw']:,} → clean rows: {metrics['rows_clean']:,}. "
               "Final model: Random Forest inside a scikit-learn Pipeline (imputer + scaler + one-hot encoder).")

# ---------- 6) data ----------
with tab_data:
    st.write("Sample of the cleaned data (filters applied):")
    st.dataframe(view.head(500), use_container_width=True)
    st.download_button("Download filtered data (CSV)", view.to_csv(index=False), "cars_filtered.csv", "text/csv")
