from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import Group
from django.contrib.auth import login
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .recommend import recommend_cars, get_form_choices
import os
import joblib
from django.conf import settings

import joblib
scenario_model = joblib.load(os.path.join(settings.BASE_DIR, 'ml_models/scenario_regressor.pkl'))
def predict_adjusted_price(row, scenario_code, model):
    features = [
        row['Year'],
        row['Engine HP'],
        row['Engine Cylinders'],
        row['Fuel Type'],
        row['Transmission Type'],
        row['Vehicle Size'],
        scenario_code
    ]
    return model.predict([features])[0]

def home(request):
    return render(request, 'dashboard/home.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        role = request.POST.get('role')
        if form.is_valid() and role:
            user = form.save()
            group = Group.objects.get(name=role)
            user.groups.add(group)
            messages.success(request, 'Registration successful! You can now log in.')
            return redirect('login')
        else:
            messages.error(request, 'Registration failed. Please check the form.')
    else:
        form = UserCreationForm()
    return render(request, 'dashboard/register.html', {'form': form})

@login_required
def customer_dashboard(request):
    return render(request, 'dashboard/customer_dashboard.html')

@login_required
def analyst_dashboard(request):
    return render(request, 'dashboard/analyst_dashboard.html')

@login_required
def role_based_redirect(request):
    if request.user.groups.filter(name='Customer').exists():
        return redirect('customer_dashboard')
    elif request.user.groups.filter(name='BusinessAnalyst').exists():
        return redirect('analyst_dashboard')
    else:
        return redirect('home')

from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.conf import settings
import pandas as pd
import joblib
import os

from .recommend import get_form_choices  

from dashboard.recommend import recommend_cars, get_form_choices

@login_required
def search_cars(request):
    choices = get_form_choices()
    recommended_cars = None

    if request.method == 'POST':
        user_filters = {
            'min_price': int(request.POST.get('min_price')),
            'max_price': int(request.POST.get('max_price')),
            'brand': request.POST.get('brand'),
            'fuel_type': request.POST.get('fuel_type'),
            'transmission': request.POST.get('transmission'),
            'body_type': request.POST.get('body_type'),
            'min_year': int(request.POST.get('min_year')),
            'max_year': int(request.POST.get('max_year')),
        }

        scenario = request.POST.get("scenario")
        recommended_cars = recommend_cars(user_filters, scenario)

        if recommended_cars and scenario and scenario != 'none':
            scenario_map = {
                'none': 0,
                'economy_downturn': 1,
                'ev_rise': 2,
                'gas_price_surge': 3
            }
            scenario_code = scenario_map.get(scenario, 0)

            for car in recommended_cars:
                try:
                    features = [
                        int(car['Year']),
                        float(car['Engine HP']),
                        int(car.get('Engine Cylinders', 4)),
                        2 if user_filters['fuel_type'] == 'ELECTRIC' else 0,
                        1 if user_filters['transmission'] == 'AUTOMATIC' else 0,
                        1 if user_filters['body_type'] == 'MIDSIZE' else 0,
                        scenario_code
                    ]
                    car['Adjusted_Price'] = round(scenario_model.predict([features])[0], 2)
                except Exception as e:
                    car['Adjusted_Price'] = car['Price']

    return render(request, 'dashboard/search_cars.html', {
        'choices': choices,
        'recommended_cars': recommended_cars,
    })


from .models import Wishlist
from django.contrib.auth.decorators import login_required

@login_required
def add_to_wishlist(request):
    if request.method == 'POST':
        brand = request.POST.get('brand')
        price = request.POST.get('price')
        year = request.POST.get('year')
        Wishlist.objects.create(
            user=request.user,
            brand=brand,
            price=price,
            year=year
        )
        return redirect('search_cars')

@login_required
def view_wishlist(request):
    wishlist_items = Wishlist.objects.filter(user=request.user)
    return render(request, 'dashboard/view_wishlist.html', {'wishlist_items': wishlist_items})

@login_required
def remove_from_wishlist(request, item_id):
    Wishlist.objects.filter(id=item_id, user=request.user).delete()
    return redirect('view_wishlist')

import os
from django.conf import settings

@login_required
def upload_dataset(request):
    if request.method == 'POST' and request.FILES['dataset']:
        dataset = request.FILES['dataset']
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        with open(os.path.join(upload_dir, dataset.name), 'wb+') as destination:
            for chunk in dataset.chunks():
                destination.write(chunk)
        
        # After upload, redirect to dashboard (later we show insights)
        return redirect('analyst_dashboard')
    
    return render(request, 'dashboard/upload_dataset.html')

import pandas as pd

@login_required
def generate_insights(request):
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    uploaded_files = os.listdir(upload_dir)

    if not uploaded_files:
        return render(request, 'dashboard/generate_insights.html', {'error': 'No dataset found. Please upload a file first.'})

    latest_file = max([os.path.join(upload_dir, f) for f in uploaded_files], key=os.path.getctime)
    df = pd.read_csv(latest_file)

    top_brands = df['Make'].value_counts().head(5).to_dict()
    avg_price = round(df['MSRP'].mean(), 2)
    top_vehicle_type = None
    if 'Vehicle Size' in df.columns:
        top_vehicle_type = df['Vehicle Size'].value_counts().idxmax()

    # NEW: Normalize top brands
    max_count = max(top_brands.values()) if top_brands else 1
    normalized_brands = {}
    for brand, count in top_brands.items():
        normalized_width = round((count / max_count) * 100, 2)
        normalized_brands[brand] = {'count': count, 'width': normalized_width}

    insights = {
        'top_brands': normalized_brands,  # updated!
        'average_price': avg_price,
        'most_popular_vehicle_size': top_vehicle_type
    }

    return render(request, 'dashboard/generate_insights.html', {'insights': insights})



import pandas as pd
from sklearn.cluster import KMeans
import os
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

@login_required
def process_dataset(request):
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    files = sorted(os.listdir(upload_dir), key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)), reverse=True)

    if not files:
        return render(request, 'dashboard/insights.html', {'error': 'No uploaded files found.'})
    
    latest_file = os.path.join(upload_dir, files[0])
    df = pd.read_csv(latest_file)

    # Preprocessing
    df.fillna(method='ffill', inplace=True)

    # Detect correct brand column
    possible_brand_cols = ['Make', 'Brand', 'Manufacturer', 'Car_Name', 'Model']
    brand_column = None
    for col in df.columns:
        if col.strip() in possible_brand_cols:
            brand_column = col.strip()
            break

    if brand_column is None:
        return render(request, 'dashboard/insights.html', {'error': 'No suitable brand column found (like Make, Brand, Model).'})

    # Detect correct price column
    if 'MSRP' in df.columns:
        price_column = 'MSRP'
    elif 'Price' in df.columns:
        price_column = 'Price'
    else:
        return render(request, 'dashboard/insights.html', {'error': 'No price column (MSRP or Price) found.'})

    # Check Year
    if 'Year' not in df.columns:
        return render(request, 'dashboard/insights.html', {'error': 'Year column not found.'})

    # Select only needed columns
    df_selected = df[[brand_column, price_column, 'Year']]
    df_selected[brand_column] = df_selected[brand_column].astype(str).str.upper()

    # Clustering
    try:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df_selected['Cluster'] = kmeans.fit_predict(df_selected[[price_column, 'Year']])
    except:
        return render(request, 'dashboard/insights.html', {'error': 'Error while clustering. Check dataset quality.'})

    # Main Insights
    top_brands = df_selected[brand_column].value_counts().head(5)
    avg_price_per_cluster = df_selected.groupby('Cluster')[price_column].mean().round(2)
    cluster_sizes = df_selected['Cluster'].value_counts()

    # Safe check for Vehicle Size insight
    top_vehicle_type = None
    if 'Vehicle Size' in df.columns:
        top_vehicle_type = df['Vehicle Size'].value_counts().idxmax()

    # Send all insights
    insights = {
        'top_brands': top_brands.to_dict(),
        'avg_price_per_cluster': avg_price_per_cluster.to_dict(),
        'cluster_sizes': cluster_sizes.to_dict(),
        'top_vehicle_type': top_vehicle_type,
    }

    return render(request, 'dashboard/insights.html', {'insights': insights})

from prophet import Prophet
import pandas as pd
import os
from django.conf import settings
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required
def forecast_sales(request):
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    files = sorted(os.listdir(upload_dir), key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)), reverse=True)

    if not files:
        return render(request, 'dashboard/forecast_sales.html', {'error': 'No uploaded dataset found.'})

    latest_file = os.path.join(upload_dir, files[0])
    df = pd.read_csv(latest_file)

    if 'Year' not in df.columns or 'MSRP' not in df.columns:
        return render(request, 'dashboard/forecast_sales.html', {'error': 'Required columns (Year, MSRP) are missing.'})

    df_grouped = df.groupby('Year')['MSRP'].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    df_grouped['ds'] = pd.to_datetime(df_grouped['ds'], format='%Y')

    if df_grouped.empty:
        return render(request, 'dashboard/forecast_sales.html', {'error': 'Data not sufficient to forecast.'})

    model = Prophet(yearly_seasonality=True)
    model.fit(df_grouped)

    future = model.make_future_dataframe(periods=6, freq='YE')
    forecast = model.predict(future)

    # Only keep predictions beyond your original dataset
    latest_year = df_grouped['ds'].dt.year.max()

    future_forecast = forecast[forecast['ds'].dt.year > latest_year]

    forecast_data = [
        {'ds': row['ds'].strftime('%Y-%m-%d'), 'yhat': round(row['yhat'], 2)}
        for _, row in future_forecast[['ds', 'yhat']].iterrows()
    ]


    return render(request, 'dashboard/forecast_sales.html', {'forecast_data': forecast_data})

@login_required
def brand_details(request, brand):
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    files = sorted(os.listdir(upload_dir), key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)), reverse=True)

    if not files:
        return render(request, 'dashboard/brand_detail.html', {'error': 'No dataset found'})

    latest_file = os.path.join(upload_dir, files[0])
    df = pd.read_csv(latest_file)

    if 'Make' not in df.columns:
        return render(request, 'dashboard/brand_detail.html', {'error': 'Make column missing'})

    # 🔥 FIX: Filter for brand before using it
    brand_df = df[df['Make'].str.upper() == brand.upper()]

    if brand_df.empty:
        return render(request, 'dashboard/brand_detail.html', {'error': f'No data for brand {brand}'})

    # Prepare insights
    avg_price = round(brand_df['MSRP'].mean(), 2) if 'MSRP' in brand_df.columns else 0
    top_models = brand_df['Model'].value_counts().head(5).to_dict() if 'Model' in brand_df.columns else {}

    # ✅ FIXED: Now that brand_df exists, this works
    if 'Year' in brand_df.columns and 'MSRP' in brand_df.columns:
        sales_over_years = (
            brand_df.groupby('Year')['MSRP']
            .sum()
            .reset_index()
            .to_dict(orient='records')
        )
    else:
        sales_over_years = []

    return render(request, 'dashboard/brand_detail.html', {
        'brand': brand.upper(),
        'avg_price': avg_price,
        'top_models': top_models,
        'sales_over_years': sales_over_years
    })
