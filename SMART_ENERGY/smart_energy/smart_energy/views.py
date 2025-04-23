import os
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from django import forms
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required

from sklearn.preprocessing import LabelEncoder

# ====== Load LSTM Model ======
H5_MODEL_PATH = os.path.join(settings.BASE_DIR, "D:\\Projects\\SMART_ENERGY\\smart_energy\\smart_energy\\models\\energy_consumption_model.h5")
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
model_h5 = tf.keras.models.load_model(H5_MODEL_PATH, compile=False) if os.path.exists(H5_MODEL_PATH) else None

try:
    model_h5 = tf.keras.models.load_model(H5_MODEL_PATH, compile=False) if os.path.exists(H5_MODEL_PATH) else None
    print("✅ Model path:", H5_MODEL_PATH)
    print("✅ Model loaded:", model_h5)
    print("✅ Model type:", type(model_h5))
except Exception as e:
    print("❌ Error loading model:", str(e))
    model_h5 = None

# ====== Grid Operator PIN ======
GRID_OPERATOR_PIN = "1234"

# ====== Forms ======
class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["username", "email"]

# ====== Authentication Views ======
def registration_view(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("index")
    else:
        form = RegistrationForm()
    return render(request, "registration/register.html", {"form": form})

def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect("index")
    else:
        form = AuthenticationForm()
    return render(request, "registration/login.html", {"form": form})

def logout_success(request):
    logout(request)
    return redirect("login")

# ====== Home & Profile ======
def index(request):
    return render(request, "index.html")

@login_required
def profile(request):
    if request.method == "POST":
        form = ProfileUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect("profile")
    else:
        form = ProfileUpdateForm(instance=request.user)
    return render(request, "profile.html", {"form": form})

# ====== Single Prediction ======
@login_required
def predict(request):
    prediction = None
    model_type = None
    if request.method == "POST":
        try:
            time_mapping = {
                "Early Morning": 5.0,
                "Morning": 9.0,
                "Afternoon": 13.0,
                "Evening": 17.0,
                "Night": 21.0,
                "Midnight": 0.0,
            }
            time_of_day = time_mapping.get(request.POST.get("time_of_day"), 12.0)

            input_data = [
                float(request.POST.get("temperature")),
                float(request.POST.get("humidity")),
                float(request.POST.get("appliance_usage")),
                int(request.POST.get("occupancy")),
                float(request.POST.get("renewable_energy")),
                float(request.POST.get("electricity_price")),
                float(request.POST.get("power_grid_load")),
                time_of_day,
            ]

            input_array = np.zeros((48, len(input_data)), dtype=np.float32)
            input_array[-1] = input_data
            input_tensor = np.expand_dims(input_array, axis=0)

            if model_h5:
                prediction = round(model_h5.predict(input_tensor)[0][0], 2)
                model_type = "TensorFlow (.h5)"
            else:
                prediction = "Model not found."

        except Exception as e:
            prediction = f"Error: {str(e)}"
            print(traceback.format_exc())

    return render(request, "predict.html", {"prediction": prediction, "model_type": model_type})

# ====== Batch Upload View ======
@login_required
def upload(request):
    context = {}
    if request.method == "POST" and request.FILES.get("data_file"):
        uploaded_file = request.FILES["data_file"]
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = fs.path(file_path)

        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(full_path)
            elif uploaded_file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(full_path)
            else:
                context["error"] = "Invalid file format. Please upload a .csv or Excel file."
                return render(request, "upload.html", context)

            os.remove(full_path)
            df.dropna(inplace=True)

            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

            sequence_length = 48
            input_tensor = []

            for row in df.to_numpy().astype(np.float32):
                padded = np.zeros((sequence_length, len(row)), dtype=np.float32)
                padded[-1] = row
                input_tensor.append(padded)

            input_tensor = np.array(input_tensor)

            if model_h5 and hasattr(model_h5, "predict"):
                predictions = model_h5.predict(input_tensor).flatten()
                df["Predicted Energy"] = predictions
                df_html = df.head(30).to_html(classes="table table-bordered", index=False)
                context.update({
                    "success": "Prediction successful!",
                    "dataset": df_html
                })
                df.to_csv(os.path.join(settings.MEDIA_ROOT, "latest_forecast.csv"), index=False)
            else:
                context["error"] = "Model not found or not valid."

        except Exception as e:
            context["error"] = f"Error occurred: {str(e)}"
            print(traceback.format_exc())

    return render(request, "upload.html", context)

# ====== Grid Operator PIN Login ======
@csrf_exempt
def grid_view(request):
    error = None
    if request.method == "POST":
        if request.POST.get("pin") == GRID_OPERATOR_PIN:
            return redirect("grid")
        error = "Incorrect PIN. Please try again."
    return render(request, "grid_login.html", {"error": error})

# ====== Grid Dashboard ======
@login_required
def grid(request):
    context = {}
    df = None

    try:
        if request.method == "POST" and request.FILES.get("csv_file"):
            uploaded_file = request.FILES["csv_file"]
            fs = FileSystemStorage()
            file_path = fs.save(uploaded_file.name, uploaded_file)
            full_path = fs.path(file_path)

            df = pd.read_csv(full_path)
            os.remove(full_path)

            if "Predicted Energy" not in df.columns:
                context["error"] = "'Predicted Energy' column is required."
                return render(request, "grid.html", context)

            df["Load Status"] = df["Predicted Energy"].apply(
                lambda x: "High Load" if x > df["Predicted Energy"].quantile(0.75) else "Normal"
            )
            df["Optimization Suggestion"] = df["Load Status"].apply(
                lambda status: "Shift tasks to off-peak hours" if status == "High Load" else "Optimal"
            )

            request.session["df_json"] = df.to_json()

        elif request.method == "POST" and "df_json" in request.session:
            df = pd.read_json(request.session["df_json"])

        if df is not None:
            context.update({
                "dataset": df.head(30).to_html(classes="table table-bordered", index=False),
                "max_load": round(df["Predicted Energy"].max(), 2),
                "min_load": round(df["Predicted Energy"].min(), 2),
                "avg_load": round(df["Predicted Energy"].mean(), 2),
                "high_load_count": df[df["Load Status"] == "High Load"].shape[0],
                "chart_labels": json.dumps(df["Time"].tolist() if "Time" in df.columns else list(range(len(df)))),
                "chart_data": json.dumps(df["Predicted Energy"].tolist()),
                "chart_status": json.dumps(df["Load Status"].tolist()),
                "csv_uploaded": True,
            })

            if "date" in request.POST and "hour" in request.POST:
                date = request.POST.get("date")
                hour = int(request.POST.get("hour"))

                mask = (df["Date"] == date) & (df["Hour"] == hour)
                if not df[mask].empty:
                    context["current_consumption"] = round(df.loc[mask, "Predicted Energy"].values[0], 2)

                    def plot_bar(data1, data2, title):
                        fig, ax = plt.subplots()
                        ax.bar(data1.index, data1.values, label="Current Day", alpha=0.7)
                        ax.bar(data2.index, data2.values, label="Previous", alpha=0.7)
                        ax.set_title(title)
                        ax.set_xlabel("Hour")
                        ax.set_ylabel("Energy")
                        ax.legend()
                        buffer = BytesIO()
                        plt.savefig(buffer, format="png")
                        buffer.seek(0)
                        return "data:image/png;base64," + base64.b64encode(buffer.read()).decode()

                    day_data = df[df["Date"] == date].groupby("Hour")["Predicted Energy"].mean()
                    prev_data = df[df["Date"] < date].groupby("Hour")["Predicted Energy"].mean()
                    weekly_avg = df.groupby("Hour")["Predicted Energy"].mean()

                    context["bar1"] = plot_bar(day_data, prev_data, "Day vs Previous Day")
                    context["bar2"] = plot_bar(day_data, weekly_avg, "Day vs Weekly Average")
        else:
            context["message"] = "Please upload a CSV file."

    except Exception as e:
        context["error"] = f"Error: {str(e)}"
        print(traceback.format_exc())

    return render(request, "grid.html", context)