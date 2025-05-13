# AutoInsight360 🚗📊

**AutoInsight360** is an intelligent car sales and trend analysis dashboard designed for two types of users:

- 🔍 **Customers** looking for personalized car recommendations
- 📈 **Business Analysts** seeking data-driven insights and sales forecasting

Built with **Django**, **Tailwind CSS**, and **machine learning models**, AutoInsight360 bridges smart recommendations with real-time analytics.

---

## 🌟 Features

### 👤 **Customer Dashboard**
- Smart Car Finder with filters for price, brand, fuel type, etc.
- **Scenario Analysis** powered by a trained regression model:
  - Economy Downturn
  - EV Rise
  - Gas Price Surge
- Personalized Wishlist to save and manage favorite cars
- ML-powered clustering using **KMeans** and **Grid Search** optimization

### 📊 **Business Analyst Dashboard**
- Upload `.csv` car datasets
- Generate insights:
  - Top 5 brands
  - Average MSRP
  - Most popular vehicle sizes
- Brand breakdown with model stats and sales trends
- Forecast MSRP values for future years using **Prophet**
- Interactive visualizations via **Chart.js**

---

## ⚙️ Technologies Used

- Python 3.11 / Django 5.2
- Tailwind CSS (via PostCSS & npm)
- SQLite3 (local database)
- Scikit-learn (KMeans, Linear Regression)
- Prophet (for time-series forecasting)
- Chart.js (frontend graph rendering)

---

## 🛠️ Installation & Setup

1. **Clone the repository**

bash
git clone https://github.com/sonika-0224/AutoInsight360.git
cd AutoInsight360

----
**Steps to run :
**
Create and activate virtual environment

bash

python -m venv venv
source venv/bin/activate    # macOS/Linux
# OR
venv\Scripts\activate.bat   # Windows
Install Python dependencies

bash

pip install -r requirements.txt
Install Tailwind CSS dependencies

bash

npm install
npm run build-css
Run migrations and start server

bash

python manage.py makemigrations
python manage.py migrate
python manage.py runserver
