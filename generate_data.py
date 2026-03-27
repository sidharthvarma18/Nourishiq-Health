"""
generate_data.py
Run this ONCE to create survey_data.csv in the same folder.
  python generate_data.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2000

# ── helpers ───────────────────────────────────────────────────────────────────
def choice(opts, probs, n):
    return rng.choice(opts, size=n, p=probs)

def multi(opts, prob_each, n):
    result = []
    for _ in range(n):
        chosen = [o for o, p in zip(opts, prob_each) if rng.random() < p]
        result.append("|".join(chosen) if chosen else "None")
    return result

# ── 1. Demographics ───────────────────────────────────────────────────────────
gender      = choice(["Male","Female","Non-binary","Prefer not to say"], [0.52,0.44,0.02,0.02], N)
age_group   = choice(["Under 18","18-24","25-34","35-44","45-60","60+"],  [0.08,0.28,0.30,0.18,0.12,0.04], N)
city_tier   = choice(["Metro","Tier-2","Tier-3","Rural"], [0.42,0.32,0.18,0.08], N)
occupation  = choice(["Student","Salaried-Private","Salaried-Govt","Self-employed","Homemaker","Retired"],
                     [0.20,0.35,0.12,0.18,0.10,0.05], N)
income_band = choice(["<20k","20k-50k","50k-1L","1L-2L",">2L"], [0.15,0.30,0.28,0.18,0.09], N)

# ── 2. Physical profile ───────────────────────────────────────────────────────
height_cm = rng.normal(165, 9, N).clip(140, 200).round()
weight_kg = rng.normal(68,  14, N).clip(38,  130).round()
# inject ~3% outliers
out_idx = rng.choice(N, size=60, replace=False)
weight_kg[out_idx] = rng.choice([18, 22, 145, 160], size=60)

bmi = (weight_kg / (height_cm / 100) ** 2).round(1)

health_conditions = multi(
    ["Diabetes","Hypertension","Thyroid","PCOD","Heart","None"],
    [0.14, 0.18, 0.10, 0.12, 0.06, 0.55], N)

food_allergy = multi(
    ["Dairy","Gluten","Nuts","Eggs","Soy","Shellfish","None"],
    [0.12, 0.08, 0.10, 0.09, 0.06, 0.05, 0.62], N)

# ── 3. Diet habits ────────────────────────────────────────────────────────────
food_pref   = choice(["Pure-veg","Eggetarian","Non-veg","Vegan","Jain"],
                     [0.38,0.18,0.34,0.06,0.04], N)
meals_per_day = choice(["1-2","3","4-5","No schedule"], [0.10,0.45,0.28,0.17], N)
eat_outside   = choice(["Daily","3-5/week","1-2/week","Rarely"], [0.14,0.28,0.33,0.25], N)
eating_habit  = choice(["Very unhealthy","Somewhat unhealthy","Neutral","Fairly healthy","Very healthy"],
                       [0.07,0.22,0.28,0.30,0.13], N)

# ── 4. Fitness & lifestyle ────────────────────────────────────────────────────
activity_level = choice(["Sedentary","Lightly active","Moderately active","Very active"],
                        [0.30,0.28,0.26,0.16], N)
fitness_goal   = choice(["Lose weight","Gain muscle","Improve energy","Manage condition","General wellness"],
                        [0.32,0.22,0.20,0.12,0.14], N)
workout_pref   = choice(["Home-no equipment","Home-with equipment","Gym","Outdoors","None"],
                        [0.28,0.12,0.28,0.16,0.16], N)
sleep_hours    = choice(["<5hrs","5-6hrs","7-8hrs",">8hrs"], [0.10,0.28,0.46,0.16], N)

# ── 5. Psychographics ─────────────────────────────────────────────────────────
health_motivation = choice(
    ["Fear of disease","Look better","Feel energetic","Social pressure","Doctor advice"],
    [0.22,0.28,0.26,0.12,0.12], N)
diet_break_response = choice(
    ["Give up","Feel guilty-restart","Shrug off","Never happens"],
    [0.18,0.40,0.28,0.14], N)
health_personality = choice(
    ["Research first","Need guidance","Know but no discipline","Don't think about it"],
    [0.20,0.28,0.35,0.17], N)

# ── 6. Tech & trust ───────────────────────────────────────────────────────────
data_sharing_comfort = choice(
    ["Very comfortable","Comfortable with limits","Uncomfortable","Won't share"],
    [0.22,0.40,0.25,0.13], N)
health_influencer = choice(
    ["Doctor","Family","Social media","Friends","Self-research"],
    [0.24,0.20,0.26,0.14,0.16], N)
app_stop_reason = choice(
    ["Lost interest","Didn't trust it","Too complicated","Found useless","Never used"],
    [0.28,0.15,0.18,0.14,0.25], N)

# ── 7. Services & ARM ─────────────────────────────────────────────────────────
services_wanted = multi(
    ["BMI-tracker","Diet-plans","Workout-routines","Hydration-reminders","Progress-reports","Recipe-suggestions"],
    [0.70, 0.78, 0.65, 0.60, 0.55, 0.72], N)

meal_plan_types = multi(
    ["Weight-loss","Muscle-gain","Diabetic-friendly","High-protein","Ayurvedic","Keto"],
    [0.62, 0.45, 0.28, 0.52, 0.30, 0.22], N)

paid_before = choice(
    ["Yes-currently","Yes-stopped","No-willing","No-prefer-free"],
    [0.12,0.18,0.38,0.32], N)

upgrade_triggers = multi(
    ["Personalised-plans","AI-form-fix","Recipe-library","Travel-diet","Expert-chat","Ad-free"],
    [0.68, 0.42, 0.55, 0.38, 0.35, 0.30], N)

reminder_pref = multi(
    ["Push-notifications","WhatsApp","Email","SMS","No-reminders"],
    [0.58, 0.62, 0.35, 0.20, 0.15], N)

barrier = choice(
    ["Too expensive","Too complicated","Lack of motivation","No time","I follow one"],
    [0.26,0.14,0.28,0.20,0.12], N)

# ── 8. Spending ───────────────────────────────────────────────────────────────
past_health_spend = choice(["₹0","₹1-500","₹500-2k","₹2k-5k","₹5k+"],
                           [0.20,0.28,0.28,0.16,0.08], N)
pricing_model = choice(
    ["One-time","Monthly-sub","Pay-per-feature","Freemium","Family-plan"],
    [0.14,0.32,0.10,0.30,0.14], N)
doc_recommendation = choice(["Yes definitely","Maybe","No"], [0.42,0.38,0.20], N)

# ── 9. Social / family ────────────────────────────────────────────────────────
household_health = choice(
    ["Yes-several","Yes-one","No","Not sure"],
    [0.22,0.28,0.32,0.18], N)
recommend_likelihood = rng.integers(1, 6, N)  # 1-5 scale

discovery_channel = choice(
    ["Instagram-YouTube","Doctor","Friend-Family","Google","Influencer"],
    [0.32,0.18,0.22,0.18,0.10], N)

# ── Regression target: monthly_budget (₹) ────────────────────────────────────
income_map   = {"<20k":12000,"20k-50k":35000,"50k-1L":72000,"1L-2L":145000,">2L":250000}
income_num   = np.array([income_map[i] for i in income_band])

# Use a wide random multiplier — breaks perfect linear learnability
base_budget  = income_num * rng.uniform(0.002, 0.018, N)

age_multiplier = np.where(np.isin(age_group,["25-34","35-44"]), 1.3,
                 np.where(np.isin(age_group,["45-60","60+"]),    1.1, 0.85))
city_mult      = np.where(city_tier=="Metro", 1.25,
                 np.where(city_tier=="Tier-2",1.05, 0.85))
paid_mult      = np.where(paid_before=="Yes-currently", 1.6,
                 np.where(paid_before=="Yes-stopped",   1.2,
                 np.where(paid_before=="No-willing",    1.0, 0.5)))

occ_bump  = np.where(np.isin(occupation, ["Salaried-Private","Self-employed"]), 200,
            np.where(occupation=="Salaried-Govt", 100,
            np.where(occupation=="Student", -150, 0)))

activity_bump = np.where(activity_level=="Very active",       250,
               np.where(activity_level=="Moderately active",  130,
               np.where(activity_level=="Lightly active",      50, 0)))

goal_bump  = np.where(fitness_goal=="Gain muscle",  180,
             np.where(fitness_goal=="Lose weight",  140,
             np.where(fitness_goal=="Improve energy", 80, 40)))

trust_bump = np.where(data_sharing_comfort=="Very comfortable", 120,
             np.where(data_sharing_comfort=="Comfortable with limits", 60, 0))

# Heavy realistic noise — makes it genuinely hard to predict perfectly
# This simulates real-world variance: behavioural, seasonal, impulsive spending
heavy_noise     = rng.normal(0, 350, N)
skew_noise      = rng.exponential(80, N) * rng.choice([-1,1], N)  # asymmetric spend shocks
lifestyle_noise = rng.uniform(-200, 200, N)                        # unexplained lifestyle factor

monthly_budget = (base_budget * age_multiplier * city_mult * paid_mult
                  + occ_bump + activity_bump + goal_bump + trust_bump
                  + heavy_noise + skew_noise + lifestyle_noise).clip(0, 3000).round()

# ── Classification target: signup_intent ─────────────────────────────────────
intent_score = np.zeros(N)
intent_score += np.where(paid_before=="Yes-currently",   3,
                np.where(paid_before=="No-willing",       2,
                np.where(paid_before=="Yes-stopped",      1, 0)))
intent_score += np.where(activity_level=="Very active",   2,
                np.where(activity_level=="Moderately active", 1, 0))
intent_score += np.where(eating_habit=="Very healthy",    1,
                np.where(eating_habit=="Fairly healthy",  1, 0))
intent_score += np.where(city_tier=="Metro",              1.5,
                np.where(city_tier=="Tier-2",             1.0, 0))
intent_score += np.where(data_sharing_comfort=="Very comfortable", 1.5,
                np.where(data_sharing_comfort=="Comfortable with limits", 0.8, 0))
intent_score += monthly_budget / 500
intent_score += rng.normal(0, 0.8, N)  # noise

signup_intent = np.where(intent_score >= 6, "High",
                np.where(intent_score >= 3, "Medium", "Low"))

# ── Assemble dataframe ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "gender": gender, "age_group": age_group, "city_tier": city_tier,
    "occupation": occupation, "income_band": income_band,
    "height_cm": height_cm, "weight_kg": weight_kg, "bmi": bmi,
    "health_conditions": health_conditions, "food_allergy": food_allergy,
    "food_pref": food_pref, "meals_per_day": meals_per_day,
    "eat_outside": eat_outside, "eating_habit": eating_habit,
    "activity_level": activity_level, "fitness_goal": fitness_goal,
    "workout_pref": workout_pref, "sleep_hours": sleep_hours,
    "health_motivation": health_motivation,
    "diet_break_response": diet_break_response,
    "health_personality": health_personality,
    "data_sharing_comfort": data_sharing_comfort,
    "health_influencer": health_influencer,
    "app_stop_reason": app_stop_reason,
    "services_wanted": services_wanted,
    "meal_plan_types": meal_plan_types,
    "paid_before": paid_before,
    "upgrade_triggers": upgrade_triggers,
    "reminder_pref": reminder_pref,
    "barrier": barrier,
    "past_health_spend": past_health_spend,
    "pricing_model": pricing_model,
    "doc_recommendation": doc_recommendation,
    "household_health": household_health,
    "recommend_likelihood": recommend_likelihood,
    "discovery_channel": discovery_channel,
    "monthly_budget": monthly_budget,
    "signup_intent": signup_intent,
})

df.to_csv("survey_data.csv", index=False)
print(f"✅  survey_data.csv written — {len(df)} rows, {len(df.columns)} columns")
print(df["signup_intent"].value_counts())
