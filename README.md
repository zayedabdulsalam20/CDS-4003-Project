# Used Car Price Prediction (Craigslist Cars & Trucks)

CDS-4003 Project · CRISP-DM · Deployed with Streamlit

**Live app:** https://cds-4003-project-fnyyvaxzh9xnbnmj9uyppf.streamlit.app/
**Dataset:** [Craigslist Cars & Trucks on Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)

## What's in this repo

- `Vechiles.ipynb` is our main notebook. It has every step of the project, from exploring the data to saving the final model.
- `app.py` is the Streamlit app. You can enter a car's details and get a price estimate, and there are also some dashboard tabs to explore the data.
- `car_price_model.pkl` is the trained model (about 48.5 MB).
- `cars_clean_sample.csv` is a 20,000-row sample of the cleaned data. The app uses it for the charts.
- `model_metrics.json` stores the model scores that we show inside the app.
- `requirements.txt` lists the packages Streamlit Cloud needs to install.

We didn't upload the original `vehicles.csv` because it's bigger than GitHub's 100 MB limit. You can download it from the Kaggle link above.

## 1. Business Understanding

Pricing a used car is harder than it looks. If a dealer asks for too much, the car just sits there. If they ask for too little, they lose money. We wanted to build something that looks at a car's details and gives a fair price.

So our question was simple: **given a car's year, mileage, brand, condition and so on, what should it sell for?**

Since the answer is a number (the price), this is a **regression** problem. Before we started, we set a goal for ourselves: the final model should reach an R² of at least 0.75 on the test data, and it should beat a basic Linear Regression model.

## 2. Data Understanding

The data comes from used-car ads posted on Craigslist all over the US. Our file has **295,451 rows and 41 columns**, which is more than the project minimum of 200k rows and 20 columns.

We started with the usual checks: `head()`, `shape`, `info()`, `isna().sum()` and `describe()`. It didn't take long to see that the data was messy:

- The `price` column was stored as text instead of numbers. When we converted it, we found prices of $0 and some ridiculous values, the biggest one being 1e+24.
- Some cars were listed as made in 1900, and a few even in 2027, which hasn't happened yet.
- Mileage went from 1 mile all the way up to 10 million miles.
- `cylinders` was written as text, like "6 cylinders", instead of just the number 6.
- `size` and `county` were completely empty. `monthly_payment` and `other_attributes` were almost empty too (98–99% missing).
- A lot of other useful columns had missing values. `condition` was missing in 46% of the ads, `cylinders` in 27%, `paint_color` in 25% and `drive` in 24%.
- Some latitude/longitude values were outside the US.
- Many cars were posted more than once. Dealers often repost the same car in different areas.

We also made some charts. The pairplot showed what you'd expect: newer cars cost more, and cars with higher mileage cost less. Most cars are under $30,000, with a few expensive ones pulling the distribution to the right. When we looked at condition, "new" and "like new" cars averaged around $19,500, while "fair" cars averaged only about $4,000. A lot of sellers didn't fill in the condition at all, so "unknown" ended up being the biggest group.

## 3. Data Preparation

We made a copy of the data first (`df_new = df.copy()`) so the original stayed untouched. Then we kept only the 16 columns we thought would actually affect the price: price, year, odometer, manufacturer, condition, cylinders, fuel, title_status, transmission, drive, type, paint_color, state, seller_type, plus the two odometer flags.

After that, we cleaned the data step by step:

1. We converted `price` to a number and pulled the number out of `cylinders`.
2. We removed unrealistic values. We kept prices between $1,000 and $100,000, years between 1990 and 2025, and mileage between 100 and 400,000 miles. We also dropped cars whose odometer was marked as broken or rolled back. This left us with 257,202 cars.
3. We removed duplicate rows. There were 76,470 of them, which surprised us. After this we had **180,732 cars**.
4. We filled in the missing values. Missing cylinders got the median, and missing text values became "unknown". We didn't want to guess something like "good" for half the cars, because that could mislead the model.

To get the data ready for modeling, we split it into inputs (X, 13 columns) and the target (y = price). We turned the text columns into numbers with `get_dummies`, which gave us 144 columns. Then we split the data 80/20 into training (144,585 cars) and testing (36,147 cars), and scaled it with `StandardScaler`.

## 4. Modeling

We tried three models:

| Model | MAE (test) | RMSE (test) | R² (train) | R² (test) |
|---|---|---|---|---|
| Linear Regression | $5,552 | $8,494 | 0.622 | 0.622 |
| Decision Tree | $4,100 | $7,513 | 1.000 | 0.704 |
| Random Forest | $3,511 | $5,909 | 0.897 | 0.817 |
| Random Forest (tuned) | $3,500 | $5,901 | 0.897 | 0.818 |

**Linear Regression** was our baseline. It did okay, but it assumes everything follows a straight line, and car prices don't work like that. A car loses a lot of value in its first few years, and then the drop slows down.

**Decision Tree** got a perfect score on the training data and much less on the test data. That's a clear sign of overfitting: it basically memorized the training cars instead of learning the patterns.

**Random Forest** gave the best results. It builds many trees and averages their predictions, so it's more accurate and doesn't overfit as much.

To make sure we weren't just getting lucky with one train/test split, we ran **5-fold cross validation** on the Random Forest. The R² stayed between 0.762 and 0.777 across all five folds (average 0.769 ± 0.005), so the model is stable.

Then we used **GridSearchCV** to try different settings (`n_estimators` of 50 or 100, and `max_depth` of 10, 20 or no limit). Technically, the best option was no depth limit with 100 trees (R² 0.749). We went with **max_depth = 20 and 100 trees** instead, which scored 0.748. The difference is tiny, and the model file is much smaller, which matters because GitHub doesn't accept files over 100 MB. It also overfits a bit less.

## 5. Evaluation

We measured our models with three metrics:
- **MAE**: how far off the predictions are, on average, in dollars.
- **RMSE**: similar to MAE, but big mistakes count more.
- **R²**: how much of the variation in prices the model can explain.

Our final model (a Random Forest inside a full pipeline) got these results on the test set:

| R² | MAE | RMSE |
|---|---|---|
| 0.823 | $3,439 | $5,817 |

So the model explains about 82% of the differences in car prices, and on average it's off by around $3,400. That's above the 0.75 target we set at the start.

When we looked at which features mattered most, the year was by far the biggest factor (37%), followed by mileage (20%). Cylinders (10%) and fuel type (8.5%) came next, and then whether the car is a truck or pickup and whether it's sold by a dealer or a private owner. That all makes sense to us. Age and mileage are the first things anyone asks about when buying a used car.

## 6. Deployment

The app needs to take normal input like "toyota" or "SUV", not 144 columns of zeros and ones. So we put all the preparation steps and the model into one scikit-learn **Pipeline**:

- For number columns (year, odometer, cylinders), it fills missing values with the median, then scales them.
- For text columns, it fills missing values with "unknown", then one-hot encodes them. We used `handle_unknown="ignore"`, so the app won't crash if it sees a new value.
- At the end there's the Random Forest model.

We saved the pipeline with `joblib` and tested it on one example car: a 2016 Toyota SUV with 60,000 miles in excellent condition. It predicted **$24,424**, which sounds about right.

Then we built the Streamlit app. The main tab is the price predictor. There are also tabs for price analysis, car features, prices by state, model results, and a data table, plus some filters in the sidebar. Finally, we pushed everything to this GitHub repo and deployed the app on Streamlit Community Cloud.


```bash
git clone https://github.com/zayedabdulsalam20/CDS-4003-Project.git
cd CDS-4003-Project
pip install -r requirements.txt
streamlit run app.py
```