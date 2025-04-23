from django.contrib import admin
from django.urls import path, include
from .views import index, registration_view, profile, predict, upload,logout_success,grid,grid_view  # Import predict

urlpatterns = [
    path('admin/', admin.site.urls),
    path("accounts/", include("django.contrib.auth.urls")),
    path("index/", index, name="index"),
    path("", registration_view, name="register"),
    path('accounts/profile/', profile, name='profile'),
    path('predict/', predict, name='predict'),
    path('upload/', upload, name='upload'),  
    path('logout_success/', logout_success, name='logout_success'),
    # path('result/',result,name='result'),
    path('grid/',grid,name='grid'),
    path('grid_login/',grid_view,name='grid_login')
]
