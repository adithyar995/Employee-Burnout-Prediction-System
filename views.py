from django.shortcuts import render
from .models import PredictionHistory

import os
import joblib
import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "burnout_model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "label_encoders.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = None
label_encoders = None


try:

    if os.path.exists(MODEL_PATH):

        model = joblib.load(MODEL_PATH)

    if os.path.exists(ENCODER_PATH):

        label_encoders = joblib.load(ENCODER_PATH)

    if model is not None and label_encoders is not None:

        print("ML MODEL: LOADED SUCCESSFULLY")

    else:

        print("ML MODEL: FILES NOT FOUND")

except Exception as e:

    print("ML MODEL LOAD ERROR:", e)

    model = None
    label_encoders = None


# =========================================================
# HOME
# =========================================================

def home(request):

    job_roles = []
    genders = []

    try:

        if label_encoders is not None:

            if "job_role" in label_encoders:

                job_roles = list(
                    label_encoders["job_role"].classes_
                )

            if "gender" in label_encoders:

                genders = list(
                    label_encoders["gender"].classes_
                )

    except Exception as e:

        print("DROPDOWN LOAD ERROR:", e)

    return render(
        request,
        "home.html",
        {
            "job_roles": job_roles,
            "genders": genders,
        }
    )


# =========================================================
# PREDICT
# =========================================================

def predict(request):

    if request.method != "POST":

        return render(
            request,
            "home.html"
        )


    # =====================================================
    # GET FORM VALUES
    # =====================================================

    age_raw = request.POST.get(
        "age",
        ""
    ).strip()

    gender = request.POST.get(
        "gender",
        ""
    ).strip()

    job_role = request.POST.get(
        "job_role",
        ""
    ).strip()

    experience_raw = request.POST.get(
        "experience_years",
        ""
    ).strip()

    work_hours_raw = request.POST.get(
        "work_hours_per_week",
        ""
    ).strip()

    overtime_raw = request.POST.get(
        "overtime_hours",
        ""
    ).strip()

    sleep_raw = request.POST.get(
        "sleep_hours",
        ""
    ).strip()

    work_life_raw = request.POST.get(
        "work_life_balance",
        ""
    ).strip()

    satisfaction_raw = request.POST.get(
        "job_satisfaction",
        ""
    ).strip()

    manager_raw = request.POST.get(
        "manager_support",
        ""
    ).strip()


    # =====================================================
    # REQUIRED FIELD CHECK
    # =====================================================

    required_fields = {
        "Age": age_raw,
        "Gender": gender,
        "Job Role": job_role,
        "Experience": experience_raw,
        "Work Hours": work_hours_raw,
        "Overtime Hours": overtime_raw,
        "Sleep Hours": sleep_raw,
        "Work-Life Balance": work_life_raw,
        "Job Satisfaction": satisfaction_raw,
        "Manager Support": manager_raw,
    }


    for field_name, value in required_fields.items():

        if value == "":

            return render_error(
                request,
                f"{field_name} is required."
            )


    # =====================================================
    # INTEGER CONVERSION
    # =====================================================

    try:

        age = int(age_raw)

    except ValueError:

        return render_error(
            request,
            "Age must be a whole number."
        )


    try:

        experience_years = int(
            experience_raw
        )

    except ValueError:

        return render_error(
            request,
            "Experience must be a whole number."
        )


    try:

        work_hours_per_week = int(
            work_hours_raw
        )

    except ValueError:

        return render_error(
            request,
            "Work Hours must be a whole number."
        )


    try:

        overtime_hours = int(
            overtime_raw
        )

    except ValueError:

        return render_error(
            request,
            "Overtime Hours must be a whole number."
        )


    # =====================================================
    # DECIMAL CONVERSION
    # =====================================================

    try:

        sleep_hours = float(
            sleep_raw
        )

    except ValueError:

        return render_error(
            request,
            "Sleep Hours must be a valid number."
        )


    # =====================================================
    # RATING CONVERSION
    # =====================================================

    try:

        work_life_balance = int(
            work_life_raw
        )

    except ValueError:

        return render_error(
            request,
            "Work-Life Balance must be a value from 1 to 5."
        )


    try:

        job_satisfaction = int(
            satisfaction_raw
        )

    except ValueError:

        return render_error(
            request,
            "Job Satisfaction must be a value from 1 to 5."
        )


    try:

        manager_support = int(
            manager_raw
        )

    except ValueError:

        return render_error(
            request,
            "Manager Support must be a value from 1 to 5."
        )


    # =====================================================
    # RANGE VALIDATION
    # =====================================================

    if age < 21 or age > 65:

        return render_error(
            request,
            "Age must be between 21 and 65."
        )


    if experience_years < 0 or experience_years > 35:

        return render_error(
            request,
            "Experience must be between 0 and 35 years."
        )


    if work_hours_per_week < 35 or work_hours_per_week > 60:

        return render_error(
            request,
            "Work Hours must be between 35 and 60."
        )


    if overtime_hours < 0 or overtime_hours > 15:

        return render_error(
            request,
            "Overtime Hours must be between 0 and 15."
        )


    if sleep_hours < 4.5 or sleep_hours > 9:

        return render_error(
            request,
            "Sleep Hours must be between 4.5 and 9."
        )


    if work_life_balance < 1 or work_life_balance > 5:

        return render_error(
            request,
            "Work-Life Balance must be between 1 and 5."
        )


    if job_satisfaction < 1 or job_satisfaction > 5:

        return render_error(
            request,
            "Job Satisfaction must be between 1 and 5."
        )


    if manager_support < 1 or manager_support > 5:

        return render_error(
            request,
            "Manager Support must be between 1 and 5."
        )


    # =====================================================
    # MODEL CHECK
    # =====================================================

    if model is None:

        return render_error(
            request,
            "Machine learning model is not available."
        )


    if label_encoders is None:

        return render_error(
            request,
            "Label encoders are not available."
        )


    # =====================================================
    # ENCODING
    # =====================================================

    try:

        gender_encoder = label_encoders["gender"]

        job_role_encoder = label_encoders["job_role"]

        encoded_gender = gender_encoder.transform(
            [gender]
        )[0]

        encoded_job_role = job_role_encoder.transform(
            [job_role]
        )[0]

    except ValueError:

        return render_error(
            request,
            "Selected Gender or Job Role is not available in the trained dataset."
        )

    except KeyError:

        return render_error(
            request,
            "Gender or Job Role encoder is missing."
        )

    except Exception as e:

        return render_error(
            request,
            f"Encoding error: {str(e)}"
        )


    # =====================================================
    # CREATE INPUT DATAFRAME
    # =====================================================

    input_data = pd.DataFrame(
        [[
            age,
            encoded_gender,
            encoded_job_role,
            experience_years,
            work_hours_per_week,
            overtime_hours,
            sleep_hours,
            work_life_balance,
            job_satisfaction,
            manager_support
        ]],
        columns=[
            "age",
            "gender",
            "job_role",
            "experience_years",
            "work_hours_per_week",
            "overtime_hours",
            "sleep_hours",
            "work_life_balance",
            "job_satisfaction",
            "manager_support"
        ]
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    try:

        prediction_value = model.predict(
            input_data
        )[0]

    except Exception as e:

        return render_error(
            request,
            f"Prediction error: {str(e)}"
        )


    # =====================================================
    # DECODE BURNOUT LEVEL
    # =====================================================

    try:

        burnout_encoder = label_encoders[
            "burnout_level"
        ]

        prediction_level = (
            burnout_encoder
            .inverse_transform(
                [prediction_value]
            )[0]
        )

    except Exception as e:

        return render_error(
            request,
            f"Burnout level decoding error: {str(e)}"
        )


    # =====================================================
    # WHY THIS PREDICTION
    # =====================================================

    why_prediction = []


    if work_hours_per_week > 45:

        why_prediction.append(
            "High weekly working hours may increase workload pressure."
        )


    if overtime_hours > 8:

        why_prediction.append(
            "Higher overtime hours may contribute to work-related stress."
        )


    if sleep_hours < 6:

        why_prediction.append(
            "Lower sleep duration may affect recovery and well-being."
        )


    if work_life_balance <= 2:

        why_prediction.append(
            "Low work-life balance may increase burnout risk."
        )


    if job_satisfaction <= 2:

        why_prediction.append(
            "Low job satisfaction may be associated with higher burnout risk."
        )


    if manager_support <= 2:

        why_prediction.append(
            "Low manager support may contribute to workplace stress."
        )


    if not why_prediction:

        why_prediction.append(
            "The prediction is based on the combined employee "
            "work, lifestyle, and well-being characteristics."
        )


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    recommendations = []


    if work_hours_per_week > 45:

        recommendations.append(
            "Consider maintaining a manageable weekly workload."
        )


    if overtime_hours > 8:

        recommendations.append(
            "Try to reduce excessive overtime when possible."
        )


    if sleep_hours < 6:

        recommendations.append(
            "Maintain a consistent and sufficient sleep routine."
        )


    if work_life_balance <= 2:

        recommendations.append(
            "Improve work-life balance by maintaining regular personal time."
        )


    if job_satisfaction <= 2:

        recommendations.append(
            "Identify workplace factors that may improve job satisfaction."
        )


    if manager_support <= 2:

        recommendations.append(
            "Encourage regular communication and support from management."
        )


    if not recommendations:

        recommendations.append(
            "Continue maintaining healthy work habits, adequate rest, "
            "and a balanced work-life routine."
        )


    recommendation = " ".join(
        recommendations
    )


    # =====================================================
    # SAVE PREDICTION HISTORY
    # =====================================================

    try:

        PredictionHistory.objects.create(

            age=age,

            gender=gender,

            job_role=job_role,

            experience=experience_years,

            work_hours=work_hours_per_week,

            overtime=overtime_hours,

            sleep=sleep_hours,

            work_life=work_life_balance,

            satisfaction=job_satisfaction,

            manager=manager_support,

            prediction=str(
                prediction_level
            ),

            recommendation=recommendation
        )

    except Exception as e:

        print(
            "HISTORY SAVE ERROR:",
            e
        )


    # =====================================================
    # RESULT PAGE
    # =====================================================

    return render(
        request,
        "result.html",
        {
            "prediction": prediction_level,

            "why_prediction": why_prediction,

            "recommendation": recommendation,
        }
    )


# =========================================================
# ERROR PAGE
# =========================================================

def render_error(
    request,
    message
):

    job_roles = []
    genders = []


    try:

        if label_encoders is not None:

            if "job_role" in label_encoders:

                job_roles = list(
                    label_encoders[
                        "job_role"
                    ].classes_
                )

            if "gender" in label_encoders:

                genders = list(
                    label_encoders[
                        "gender"
                    ].classes_
                )

    except Exception as e:

        print(
            "ERROR DROPDOWN LOAD:",
            e
        )


    return render(
        request,
        "home.html",
        {
            "error": message,
            "job_roles": job_roles,
            "genders": genders,
        }
    )


# =========================================================
# HISTORY
# =========================================================

def history(request):

    records = PredictionHistory.objects.all().order_by(
        "-created_at"
    )

    return render(
        request,
        "history.html",
        {
            "records": records
        }
    )