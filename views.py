from django.shortcuts import render
import joblib
import pandas as pd
import os

from dotenv import load_dotenv
from openai import OpenAI

from .models import PredictionHistory


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)


# =========================================================
# OPENAI API KEY
# =========================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)


# =========================================================
# OPENAI CLIENT
# =========================================================

client = None

if OPENAI_API_KEY:

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "model",
        "burnout_model.pkl"
    )
)


# =========================================================
# LOAD LABEL ENCODERS
# =========================================================

encoders = joblib.load(
    os.path.join(
        BASE_DIR,
        "model",
        "label_encoders.pkl"
    )
)


# =========================================================
# HOME PAGE
# =========================================================

def home(request):

    return render(
        request,
        "home.html"
    )


# =========================================================
# PREDICTION
# =========================================================

def predict(request):

    if request.method == "POST":

        # -------------------------------------------------
        # GET INPUT VALUES
        # -------------------------------------------------

        age = int(
            request.POST["age"]
        )

        gender_input = request.POST["gender"]

        job_role_input = request.POST["job_role"]

        experience = int(
            request.POST["experience"]
        )

        work_hours = int(
            request.POST["work_hours"]
        )

        overtime = int(
            request.POST["overtime"]
        )

        sleep = float(
            request.POST["sleep"]
        )

        work_life = int(
            request.POST["work_life"]
        )

        satisfaction = int(
            request.POST["satisfaction"]
        )

        manager = int(
            request.POST["manager"]
        )


        # -------------------------------------------------
        # ENCODE GENDER
        # -------------------------------------------------

        gender = encoders[
            "gender"
        ].transform(
            [gender_input]
        )[0]


        # -------------------------------------------------
        # ENCODE JOB ROLE
        # -------------------------------------------------

        job_role = encoders[
            "job_role"
        ].transform(
            [job_role_input]
        )[0]


        # -------------------------------------------------
        # CREATE DATAFRAME
        # -------------------------------------------------

        data = pd.DataFrame(

            [[
                age,
                gender,
                job_role,
                experience,
                work_hours,
                overtime,
                sleep,
                work_life,
                satisfaction,
                manager
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


        # -------------------------------------------------
        # MACHINE LEARNING PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            data
        )[0]


        # -------------------------------------------------
        # CONVERT PREDICTION TO TEXT
        # -------------------------------------------------

        prediction = encoders[
            "burnout_level"
        ].inverse_transform(
            [prediction]
        )[0]


        # =================================================
        # WHY THIS PREDICTION?
        # =================================================

        factors = []


        if work_hours >= 50:

            factors.append(
                "Long working hours"
            )


        if overtime >= 8:

            factors.append(
                "High overtime"
            )


        if sleep < 6:

            factors.append(
                "Low sleep"
            )


        if work_life <= 2:

            factors.append(
                "Low work-life balance"
            )


        if satisfaction <= 2:

            factors.append(
                "Low job satisfaction"
            )


        if manager <= 2:

            factors.append(
                "Low manager support"
            )


        if not factors:

            factors.append(
                "The prediction is based on the overall employee information provided."
            )


        # =================================================
        # RECOMMENDATION
        # =================================================

        if prediction == "Low":

            recommendation = (
                "Maintain your current work-life "
                "balance and healthy routine."
            )


        elif prediction == "Moderate":

            recommendation = (
                "Take regular breaks, improve sleep "
                "and reduce unnecessary overtime."
            )


        else:

            recommendation = (
                "Consider discussing workload with your "
                "manager or HR, reduce excessive workload "
                "and prioritize healthy rest and recovery."
            )


        # =================================================
        # SAVE PREDICTION HISTORY
        # =================================================

        PredictionHistory.objects.create(

            age=age,

            gender=gender_input,

            job_role=job_role_input,

            experience=experience,

            work_hours=work_hours,

            overtime=overtime,

            sleep=sleep,

            work_life=work_life,

            satisfaction=satisfaction,

            manager=manager,

            prediction=prediction,

            recommendation=recommendation

        )


        # =================================================
        # RESULT PAGE
        # =================================================

        return render(

            request,

            "result.html",

            {

                "prediction": prediction,

                "recommendation": recommendation,

                "factors": factors,

            }

        )


    # =====================================================
    # NON-POST REQUEST
    # =====================================================

    return render(
        request,
        "home.html"
    )


# =========================================================
# PREDICTION HISTORY
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


# =========================================================
# CHATBOT
# =========================================================

def chatbot(request):

    answer = ""

    user_message = ""


    # =====================================================
    # POST REQUEST
    # =====================================================

    if request.method == "POST":

        user_message = request.POST.get(
            "message",
            ""
        ).strip()


        # -------------------------------------------------
        # EMPTY MESSAGE
        # -------------------------------------------------

        if not user_message:

            answer = (
                "Please enter a question so I can help you."
            )


        # -------------------------------------------------
        # API KEY CHECK
        # -------------------------------------------------

        elif not OPENAI_API_KEY:

            answer = (
                "OpenAI API key was not found. "
                "Please check your .env file and make sure "
                "OPENAI_API_KEY is configured correctly."
            )

            print(
                "CHATBOT ERROR: OPENAI_API_KEY not found."
            )


        # -------------------------------------------------
        # OPENAI CLIENT CHECK
        # -------------------------------------------------

        elif client is None:

            answer = (
                "The AI assistant could not be initialized. "
                "Please check the OpenAI configuration."
            )

            print(
                "CHATBOT ERROR: OpenAI client is None."
            )


        else:

            try:

                # =========================================
                # SEND QUESTION TO OPENAI
                # =========================================

                response = client.responses.create(

                    model="gpt-5-mini",

                    instructions="""
You are a professional Employee Burnout and Workplace
Well-being Assistant for an academic machine learning project.

Your main area of expertise is:

• Employee burnout
• Workplace well-being
• Workplace stress
• Burnout causes
• Burnout symptoms
• Burnout warning signs
• Effects of burnout
• Burnout prevention
• Burnout management
• Working hours
• Overtime
• Sleep and employee well-being
• Work-life balance
• Job satisfaction
• Manager support
• Employee motivation
• Workplace workload
• Healthy workplace practices
• Burnout risk levels
• Employee burnout prediction
• Machine-learning-based burnout prediction
• General workplace well-being

IMPORTANT:

Understand the meaning and intention of the user's question.

Do NOT depend only on exact keywords.

The user can ask the same concept in many different ways.

For example:

"What is burnout?"

"Why do employees feel exhausted?"

"How does excessive work affect employees?"

"Can working long hours increase burnout?"

"Is poor sleep related to burnout?"

"Why is work-life balance important?"

"What happens when employees work too much?"

"Can job satisfaction affect burnout?"

"How can managers support employees?"

"How can companies prevent burnout?"

"What are the advantages of good work-life balance?"

"What are the disadvantages of poor work-life balance?"

"Can burnout reduce productivity?"

"What is the difference between stress and burnout?"

"What factors increase burnout risk?"

"What factors reduce burnout risk?"

"An employee works 60 hours a week and sleeps 5 hours.
What risks might they face?"

All of these should be understood and answered if they
are related to employee burnout or workplace well-being.

The user may ask:

• Positive questions
• Negative questions
• Advantages
• Disadvantages
• Causes
• Effects
• Prevention
• Solutions
• Comparisons
• Examples
• Scenarios
• Why questions
• How questions
• Conceptual questions
• Academic questions

Answer according to the meaning of the question.

If the user asks a "why" question, explain the reason.

If the user asks a "how" question, explain the process
or provide practical steps.

If the user asks for advantages and disadvantages,
separate them clearly.

If the user asks for causes and effects,
separate causes and effects clearly.

If the user asks for an example,
give a practical workplace example.

If the user asks a comparison,
clearly explain the differences.

If the user asks a scenario,
analyze the scenario using employee burnout
and workplace well-being concepts.

Keep answers understandable and suitable for an
academic project presentation or viva.

For simple questions, answer briefly.

For deeper questions, use structured points.

IMPORTANT DOMAIN RESTRICTION:

You are specialized in Employee Burnout and Workplace
Well-being.

If the question is completely unrelated, such as:

• House renovation
• Cooking
• Cricket
• Movies
• Politics
• Shopping
• Travel
• General entertainment
• Unrelated programming
• Unrelated technology
• Unrelated personal topics

do NOT answer the unrelated question.

Instead respond:

"Sorry, I am specialized in Employee Burnout and
Workplace Well-being, so I can only help with questions
related to employee burnout, workplace stress and
employee well-being."

However, if a question has even a meaningful connection
to workplace well-being, employee stress, employee health,
workload, productivity, motivation or burnout, answer it.

IMPORTANT FOR PREDICTION:

The machine-learning model provides a burnout risk estimate.

It is NOT a medical diagnosis.

Never claim that the system has medically diagnosed
a person with burnout.

For serious distress or significant mental-health concerns,
suggest appropriate professional or workplace support.

Tone:

Professional
Friendly
Clear
Academic
Helpful

Do not repeatedly say that you only understand predefined
questions. Understand natural-language questions.
""",

                    input=user_message

                )


                # =========================================
                # GET RESPONSE TEXT
                # =========================================

                answer = response.output_text


                # =========================================
                # EMPTY AI RESPONSE CHECK
                # =========================================

                if not answer:

                    answer = (
                        "I could not generate a response right now. "
                        "Please try asking the question again."
                    )


            except Exception as e:

                # =========================================
                # PRINT ACTUAL ERROR
                # =========================================

                print(
                    "CHATBOT ERROR:",
                    repr(e)
                )


                # =========================================
                # SHOW ERROR IN BROWSER
                # =========================================

                answer = (
                    "Sorry, the AI assistant could not process "
                    "your question right now.\n\n"
                    "Technical error:\n"
                    f"{str(e)}"
                )


    # =====================================================
    # CHATBOT PAGE
    # =====================================================

    return render(

        request,

        "chatbot.html",

        {

            "answer": answer,

            "user_message": user_message

        }

    )