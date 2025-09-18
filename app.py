!pip install ipywidgets --quiet
import joblib
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# ======== 1. تحميل النموذج الموجود في Notebook ========
model = joblib.load("exoplanet_model.joblib")
print("Model loaded successfully")

features = ['pl_rade','pl_bmasse','pl_orbper','st_lum','pl_eqt','st_mass']

# ======== 2. دالة التنبؤ ========
def predict_planet(radius, mass, period, luminosity, temp, stellar_mass):
    input_data = np.array([[radius, mass, period, luminosity, temp, stellar_mass]])
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0].max() * 100
    return ("Habitable" if prediction==1 else "Not Habitable"), round(confidence,2)

# ======== 3. واجهة المستخدم ========
radius_slider = widgets.FloatSlider(min=0.5, max=2.5, step=0.05, value=1.0, description='Radius:')
mass_slider = widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, description='Mass:')
period_slider = widgets.FloatSlider(min=50, max=500, step=5, value=365, description='Orbital Period:')
lum_slider = widgets.FloatSlider(min=0.1, max=2.0, step=0.05, value=1.0, description='Luminosity:')
temp_slider = widgets.FloatSlider(min=200, max=350, step=5, value=288, description='Temperature:')
stellar_slider = widgets.FloatSlider(min=0.5, max=1.5, step=0.05, value=1.0, description='Stellar Mass:')

output = widgets.Output()

def on_button_click(b):
    with output:
        output.clear_output()
        pred, conf = predict_planet(
            radius_slider.value,
            mass_slider.value,
            period_slider.value,
            lum_slider.value,
            temp_slider.value,
            stellar_slider.value
        )
        print(f"Prediction: {pred}, Confidence: {conf}%")

button = widgets.Button(description="Predict Habitability", button_style='success')
button.on_click(on_button_click)

display(radius_slider, mass_slider, period_slider, lum_slider, temp_slider, stellar_slider, button, output)