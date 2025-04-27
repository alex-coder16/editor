from django.urls import path
from . import views # Import views from the current app

urlpatterns = [
    path('', views.index_view, name='index'), # Map the root URL of the app to index_view
    # Add other app-specific URLs later
]