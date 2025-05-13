from django.contrib.auth import views as auth_views
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='dashboard/login.html', redirect_authenticated_user=True), name='login'),  # ✅ keep only this one
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),  
    path('customer_dashboard/', views.customer_dashboard, name='customer_dashboard'),
    path('analyst_dashboard/', views.analyst_dashboard, name='analyst_dashboard'),
    path('search_cars/', views.search_cars, name='search_cars'),
    path('afterlogin/', views.role_based_redirect, name='afterlogin'),  
    path('add_to_wishlist/', views.add_to_wishlist, name='add_to_wishlist'),
    path('view_wishlist/', views.view_wishlist, name='view_wishlist'),
    path('remove_from_wishlist/<int:item_id>/', views.remove_from_wishlist, name='remove_from_wishlist'),
    path('upload_dataset/', views.upload_dataset, name='upload_dataset'),
    path('generate_insights/', views.generate_insights, name='generate_insights'),
    path('process_dataset/', views.process_dataset, name='process_dataset'),
    path('forecast_sales/', views.forecast_sales, name='forecast_sales'),
    path('brand/<str:brand>/', views.brand_details, name='brand_details'),


]
