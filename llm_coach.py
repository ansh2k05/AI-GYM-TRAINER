"""
AI Gym Trainer — LLM AI Fitness Coach Engine
============================================
Powered by Google Gemini API (gemini-2.5-flash).
Provides:
  - Personalized post-workout biomechanical critique & tempo analysis
  - Interactive fitness & diet coach Q&A console
  - Daily workout breakdown and recovery planning
  - Graceful rule-based offline fallback when API key is not set
"""

import os
import sys
import time
from dotenv import load_dotenv

# Configure UTF-8 stdout on Windows to prevent UnicodeEncodeError in console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables from .env if present
load_dotenv()

# Models to try in cascade order to guarantee UNLIMITED availability & zero downtime
_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
_CASCADE_MODELS = [
    _DEFAULT_MODEL,
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.6-flash",
]
# Deduplicate while preserving order
_CASCADE_MODELS = list(dict.fromkeys([m for m in _CASCADE_MODELS if m]))
_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

_client = None
if _API_KEY:
    try:
        from google import genai
        _client = genai.Client(api_key=_API_KEY)
    except Exception as e:
        print(f"[WARN] Failed to initialize Gemini Client: {e}")
        _client = None


# ─────────────────────────────────────────────
#  Multi-Model Cascade helper — Unlimited Availability
# ─────────────────────────────────────────────

def _gemini_generate(prompt: str, max_retries_per_model: int = 2) -> str | None:
    """
    Call Gemini with automatic multi-model failover and backoff retry.
    Iterates through _CASCADE_MODELS so quota exhaustion or 503 errors on one
    model automatically switch to another working model without interrupting the user.
    """
    if _client is None:
        return None

    for model_name in _CASCADE_MODELS:
        delay = 0.8
        for attempt in range(1, max_retries_per_model + 1):
            try:
                response = _client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text.strip()
                break
            except Exception as e:
                err_str = str(e)
                is_quota_or_overload = any(
                    code in err_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "quota", "Quota")
                )
                if is_quota_or_overload:
                    if attempt < max_retries_per_model:
                        time.sleep(delay)
                        delay *= 1.5
                        continue
                    else:
                        # Failover to next candidate model
                        break
                else:
                    # Model not found or unsupported -> try next model
                    break
    return None


def is_llm_available() -> bool:
    """Check if Gemini API client is configured and ready."""
    return _client is not None


def get_coach_status() -> dict:
    """Return status dictionary of the LLM Coach."""
    configured = is_llm_available()
    return {
        "configured": configured,
        "provider": "Google Gemini",
        "model": _CASCADE_MODELS[0] if configured else "Offline Heuristic Coach",
        "has_key": bool(_API_KEY),
        "unlimited": True,
    }


# ─────────────────────────────────────────────
#  1. Post-Workout Biomechanical Evaluation
# ─────────────────────────────────────────────

def evaluate_workout_session(exercise_name: str,
                             rep_count: int,
                             target_reps: int = 10,
                             pace_avg: float = 0.0,
                             streak: int = 0,
                             form_cues: list = None) -> str:
    """
    Generate an intelligent AI Coach critique after an exercise session.
    Returns 3 concise, bulleted insights: Cadence, Form Cue, and Next Goal.
    """
    if _client is not None:
        try:
            prompt = f"""
You are an elite AI Sports Biomechanics & Fitness Coach for the AI Gym Trainer app.
Analyze this workout session and give exactly 3 concise, punchy bullet points (max 2 sentences each):
- Exercise: {exercise_name}
- Completed: {rep_count} / {target_reps} reps
- Target Met: {'YES' if rep_count >= target_reps else 'NO'}
- Average Rep Pace: {pace_avg:.1f}s per rep
- Streak: {streak}
- Form Observations: {', '.join(form_cues) if form_cues else 'Consistent steady form'}

Structure (use plain ASCII headers, no emojis or markdown hashes):
1. [CADENCE & TEMPO]: evaluate time-under-tension and rep pacing.
2. [BIOMECHANICS & FORM]: give 1 specific technique tip for {exercise_name}.
3. [RECOVERY & GOAL]: quick hydration/breathing tip and motivation for next set.
Keep it direct, motivating, and scientifically sound.
"""
            result = _gemini_generate(prompt)
            if result:
                return result
        except Exception:
            # Fall through to heuristic fallback on API error
            pass

    # ── Rule-Based Heuristic Fallback (Offline) ───────────────────────
    return _heuristic_workout_evaluation(exercise_name, rep_count, target_reps, pace_avg)


def _heuristic_workout_evaluation(exercise_name: str,
                                   rep_count: int,
                                   target_reps: int,
                                   pace_avg: float) -> str:
    """High-quality sports science feedback when offline or unconfigured."""
    is_met = rep_count >= target_reps
    
    # Cadence analysis
    if pace_avg <= 0:
        cadence_tip = "[CADENCE] Logged a quick session. Aim to track consistent rep rhythm next time."
    elif pace_avg < 1.4:
        cadence_tip = f"[CADENCE] ({pace_avg:.1f}s/rep): Explosive tempo! Slow down the eccentric (lowering) phase to 2-3s for greater hypertrophy."
    elif pace_avg <= 3.5:
        cadence_tip = f"[CADENCE] ({pace_avg:.1f}s/rep): Optimal time-under-tension! Excellent control throughout the range of motion."
    else:
        cadence_tip = f"[CADENCE] ({pace_avg:.1f}s/rep): Sustained strength endurance pacing. Good stamina management under load."

    # Specific Exercise Technique Cues
    ex_lower = exercise_name.lower()
    if "curl" in ex_lower:
        technique_tip = "[FORM] Keep elbows strictly locked at your sides. Avoid swinging your shoulders to initiate the curl."
    elif "pec" in ex_lower or "fly" in ex_lower:
        technique_tip = "[FORM] Keep a slight soft bend in your elbows and focus on contracting the inner pectorals as hands cross."
    elif "shoulder" in ex_lower or "press" in ex_lower:
        technique_tip = "[FORM] Press overhead without arching the lower back. Lock out fully with biceps by ears at peak."
    else:
        technique_tip = "[FORM] Maintain a neutral spine and brace your core throughout every repetition."

    # Goal & Recovery
    if is_met:
        goal_tip = f"[NEXT GOAL] Outstanding! Full {target_reps}-rep target conquered. Hydrate, take 90s rest, and crush the next exercise!"
    else:
        goal_tip = f"[NEXT GOAL] Completed {rep_count}/{target_reps} reps. Shake it out, breathe deeply, and push to hit the full 10 on your next set!"

    return f"{cadence_tip}\n{technique_tip}\n{goal_tip}"


# ─────────────────────────────────────────────
#  2. Daily Dashboard Workout Brief
# ─────────────────────────────────────────────

def generate_daily_brief(daily_summary: dict) -> str:
    """Generate an AI summary of today's progress across all exercises."""
    done = daily_summary.get("total_done", 0)
    target = daily_summary.get("total_target", 40)
    remaining = daily_summary.get("remaining_exercises", [])
    completed = daily_summary.get("completed_exercises", [])

    if _client is not None:
        try:
            prompt = f"""
You are the AI Fitness Coach for AI Gym Trainer.
Give a brief, motivating daily workout briefing (3 short sentences):
- Total Reps Today: {done}/{target} ({int(done/target*100 if target else 0)}%)
- Completed Exercises: {', '.join(completed) if completed else 'None yet'}
- Remaining Exercises: {', '.join([f"{ex['name']} ({ex['needed']} reps left)" for ex in remaining]) if remaining else 'ALL DONE!'}
Provide an energizing game plan for finishing the remaining targets today.
"""
            result = _gemini_generate(prompt)
            if result:
                return result
        except Exception:
            pass

    # Heuristic fallback
    if not remaining:
        return (f"[DAILY MASTERY] You have completed all {target} reps across all exercises today!\n"
                "Prioritize protein intake and muscle recovery — stellar dedication!")
    elif done == 0:
        return ("[READY TO TRAIN] 3 exercises (10 reps each) await you today.\n"
                "Recommended order: Shoulder Press -> Pec Dec Fly -> Dumbbell Curl.")
    else:
        left_names = ", ".join([ex["name"] for ex in remaining])
        return (f"[DAILY PROGRESS] {done}/{target} reps done! You still have {len(remaining)} exercise(s) left: {left_names}.\n"
                f"Keep the momentum rolling and complete today's full {target}-rep target!")


# ─────────────────────────────────────────────
#  3. Intelligent Weekly Diet Plan Table Generator (Simple Ghar Ka Khana)
# ─────────────────────────────────────────────

def generate_diet_plan_table(user_query: str = "",
                             context: dict = None,
                             weight_kg: int = None,
                             height_cm: int = None,
                             **kwargs) -> str:
    """
    Generates a 7-Day Weekly Meal Schedule (Monday to Sunday) formatted in a clean Markdown Table.
    Personalized by Height & Weight with BMI calculation.
    Designed specifically for a normal person using simple, affordable, home-cooked food
    ('Ghar Ka Khana' — Dal, Roti, Rice, Curd/Dahi, Boiled Eggs, Paneer, Chana, Sabzi)
    without complicated grams, fancy measuring scales, or expensive exotic supplements.
    """
    import re
    q = user_query.lower()

    # Detect user body weight from query if not directly passed
    if weight_kg is None:
        m_wt = re.search(r'(\d{2,3})\s*(?:kg|kilos|kgs)', q)
        if m_wt:
            weight_kg = int(m_wt.group(1))

    # Detect user height from query if not directly passed
    if height_cm is None:
        # Check cm: e.g. 175cm, 168 cm
        m_ht_cm = re.search(r'(\d{2,3})\s*(?:cm|centimeters|centimeter)', q)
        if m_ht_cm:
            height_cm = int(m_ht_cm.group(1))
        else:
            # Check feet/inches: e.g. 5'9", 5 ft 9 in, 5 feet 10
            m_ht_ft = re.search(r"(\d)\s*(?:ft|feet|foot|')\s*(?:(\d{1,2})\s*(?:in|inch|inches|\")?)?", q)
            if m_ht_ft:
                feet = int(m_ht_ft.group(1))
                inches = int(m_ht_ft.group(2)) if m_ht_ft.group(2) else 0
                height_cm = int(feet * 30.48 + inches * 2.54)

    # Defaults if completely omitted
    effective_wt = weight_kg if weight_kg else 75
    effective_ht = height_cm if height_cm else 175

    # Calculate BMI & BMR (Mifflin-St Jeor)
    ht_m = effective_ht / 100.0
    bmi = round(effective_wt / (ht_m * ht_m), 1)

    ht_ft = int(effective_ht / 30.48)
    ht_in = int(round((effective_ht % 30.48) / 2.54))
    ht_str = f"{effective_ht} cm ({ht_ft}'{ht_in}\")"

    if bmi < 18.5:
        bmi_status = "Underweight"
        bmi_color_tag = "🔵 Underweight (Healthy Caloric Surplus Needed)"
    elif bmi < 25.0:
        bmi_status = "Healthy Normal Weight"
        bmi_color_tag = "🟢 Healthy / Ideal Weight (Lean Maintenance & Toning)"
    elif bmi < 30.0:
        bmi_status = "Overweight"
        bmi_color_tag = "🟡 Overweight (Targeted Fat Loss Deficit Recommended)"
    else:
        bmi_status = "Obese"
        bmi_color_tag = "🔴 Obese (Gentle Caloric Deficit & Daily Walking Recommended)"

    # Basal Metabolic Rate & Total Daily Energy Expenditure
    bmr = int(10 * effective_wt + 6.25 * effective_ht - 5 * 28 + 5)
    tdee = int(bmr * 1.35)  # Moderate active multiplier with 3 daily exercises

    # Detect primary goal
    is_fat_loss = any(k in q for k in ("fat loss", "weight loss", "lose", "lean", "burn fat", "cutting", "slim", "belly"))
    is_muscle_gain = any(k in q for k in ("muscle", "bulk", "gain", "mass", "hypertrophy", "size"))

    if not is_fat_loss and not is_muscle_gain:
        if bmi >= 25.0:
            is_fat_loss = True
        elif bmi < 19.0:
            is_muscle_gain = True
        else:
            is_fat_loss = True  # Standard default priority

    if is_fat_loss:
        target_cals = f"{max(1400, tdee - 450)} kcal"
        protein_target = f"{max(110, int(effective_wt * 1.4))}g"
        goal_title = f"🔥 7-Day Ultra-Simple Weekly Diet Plan — Fat Loss & Fitness (Ghar Ka Khana)"

        # 7-Day Ultra-Simple Weekly Schedule (Monday - Sunday) for Fat Loss
        weekly_rows = [
            ("Monday",
             "2 Besan Chilla with Green Chutney + 1 Bowl Curd *(or 2 Boiled Eggs + 1 Banana)*",
             "2 Phulka Rotis + 1 Katori Dal Tadka + 1 Katori Bhindi Sabzi + Cucumber Salad",
             "1 Cup Tea / Coffee (less sugar) + 1 Handful Bhuna Chana (Roasted Chana)",
             "2 Phulka Rotis + 1 Katori Moong Dal / Paneer Bhurji + Fresh Green Salad",
             "Start week with 4L water; avoid outside snacks"),

            ("Tuesday",
             "1 Bowl Vegetable Poha with Peanuts + 1 Glass Chaas *(or 2 Boiled Eggs)*",
             "1 Bowl Rice + 1 Katori Rajma (Kidney Beans) + Cucumber Tomato Salad",
             "1 Banana + 4-5 Soaked Almonds",
             "2 Phulka Rotis + 1 Katori Lauki (Ghiya) Sabzi + 1 Katori Curd (Dahi)",
             "Light fiber-rich dinner for easy digestion"),

            ("Wednesday",
             "1 Bowl Warm Dalia or Oats with Milk & 1 sliced Banana *(or 2 Boiled Eggs)*",
             "2 Phulka Rotis + 1 Katori Soya Chunks Curry *(or Chicken Curry)* + 1 Katori Curd + Salad",
             "1 Cup Green Tea / Lemon Water + 1 Bowl Roasted Makhana (Foxnuts)",
             "2 Phulka Rotis + 1 Katori Palak Paneer / Dal + Salad",
             "Mid-week clean protein: Keeps hunger away"),

            ("Thursday",
             "1 Bowl Vegetable Upma + 1 Glass Lemon Water or Chaas",
             "2 Phulka Rotis + 1 Katori Chole (Chickpeas) + Tomato Onion Salad",
             "1 Handful Bhuna Chana + 1 Apple or Orange",
             "2 Phulka Rotis + 1 Katori Toor Dal + French Beans Sabzi + Salad",
             "Good gut health: High natural dietary fiber"),

            ("Friday",
             "2 Whole Wheat Toast + 2 Boiled Eggs *(or 1 Paneer Roti with Dahi)* + Tea",
             "1 Bowl Rice + 2 Boiled Eggs in light curry *(or 1 Katori Paneer Curry)* + Salad",
             "1 Cup Chai / Coffee + Handful Roasted Peanuts",
             "2 Phulka Rotis + 1 Katori Mix Dal + Sautéed Green Sabzi + Salad",
             "Energy for workouts: Compound movements fuel"),

            ("Saturday",
             "3 Idlis or 1 Dosa with Sambar & Coconut Chutney",
             "2 Phulka Rotis + 1 Katori Dal Tadka *(or Chicken Masala)* + Curd + Salad",
             "Boiled Peanut & Sprout Chaat with a squeeze of fresh lemon",
             "1 Bowl Simple Vegetable Khichdi + 1 Katori Dahi + 1 Roasted Papad",
             "Saturday comfort food: Light and relaxing on stomach"),

            ("Sunday",
             "2 Moong Dal Chilla with Mint Chutney *(or 2 Boiled Eggs)*",
             "**Sunday Ghar Ka Khana:** 2 Phulka Rotis + 1/2 Bowl Rice + Chicken Curry / Paneer Curry + Salad",
             "1 Bowl Roasted Makhana or 1 Fresh Fruit (Papaya/Apple)",
             "2 Light Phulka Rotis + 1 Katori Yellow Moong Dal + Cucumber Salad",
             "Weekly reset: Sleep early, ready for Monday reps")
        ]

    else:
        # Muscle Gain / Bulking Plan
        target_cals = f"{max(2150, tdee + 350)} kcal"
        protein_target = f"{max(130, int(effective_wt * 1.8))}g"
        goal_title = f"💪 7-Day Ultra-Simple Weekly Diet Plan — Muscle Mass & Strength (Ghar Ka Khana)"

        # 7-Day Ultra-Simple Weekly Schedule (Monday - Sunday) for Muscle Gain
        weekly_rows = [
            ("Monday",
             "3 Boiled Eggs (2 whole + 1 white) *(Veg: 100g Paneer Bhurji)* + 2 Toast + 1 Banana Shake",
             "3 Phulka Rotis + 1 Bowl Rice + 1 Bowl Chicken Curry *(Veg: 1 Bowl Soya Curry)* + Dal + Curd",
             "1 Glass Sattu Drink or Warm Milk + 1 Handful Almonds & Raisins",
             "3 Phulka Rotis + 1 Bowl Paneer / Chicken / Fish Curry + 1 Katori Dal + Salad",
             "High glycogen and amino acids for heavy lifting"),

            ("Tuesday",
             "2 Stuffed Paneer Parathas with 1 Bowl Dahi + 2 Boiled Eggs *(Veg: 1 Glass Milk)*",
             "2 Bowls Rice + 1 Large Bowl Rajma + 2 Phulka Rotis + Curd + Onion Salad",
             "2 Bread Slices with Peanut Butter + 1 Banana",
             "3 Phulka Rotis + 1 Bowl Egg Curry (3 eggs) *(Veg: Paneer Curry)* + Dal + Salad",
             "Sustained caloric surplus for muscle growth"),

            ("Wednesday",
             "1 Large Bowl Oats with Milk, 1 spoon Peanut Butter & Banana + 2 Boiled Eggs",
             "3 Phulka Rotis + 1 Bowl Rice + 150g Chicken Breast / Paneer Bhurji + Dal + Curd",
             "1 Bowl Bhuna Chana + 1 Banana + Black Coffee (Pre-workout kick)",
             "3 Phulka Rotis + 1 Bowl Chole (Chickpeas) + 1 Katori Palak Paneer + Salad",
             "Peak muscle fullness and rep stamina"),

            ("Thursday",
             "3-Egg Scramble with onion & chili + 2 Toast *(Veg: 2 Besan-Paneer Chillas with Dahi)*",
             "3 Phulka Rotis + 1 Bowl Toor Dal + 1 Bowl Aloo Gobi / Bhindi + Curd + 2 Boiled Eggs",
             "1 Glass Chaas or Milk + Handful of Cashews & Walnuts",
             "3 Phulka Rotis + 1 Bowl Rice + Chicken / Fish Curry *(Veg: Soya Bhurji)* + Salad",
             "Joint lubrication & healthy dietary fats"),

            ("Friday",
             "1 Large Bowl Poha with lots of Peanuts + 3 Boiled Eggs *(Veg: 100g Paneer Cubes)*",
             "3 Phulka Rotis + 1 Bowl Dal Makhani + Chicken / Paneer + Rice + Salad",
             "2 Toast with Peanut Butter + 1 Banana + Tea / Coffee",
             "3 Phulka Rotis + 1 Katori Dal + 150g Paneer or Boiled Egg Curry + Salad",
             "Upper-body pump preparation for weekend sets"),

            ("Saturday",
             "4 Idlis with Sambar + 1 Bowl Dahi + 2 Boiled Eggs *(Veg: 1 Glass Badam Milk)*",
             "3 Phulka Rotis + 2 Bowls Rice + 1 Bowl Egg Curry *(Veg: Black Chana Masala)* + Curd",
             "Boiled Peanut & Sprout Salad with lemon + 1 Glass Chaas",
             "3 Phulka Rotis + 1 Large Bowl Khichdi with extra Dal & Paneer + 1 Papad",
             "High bioavailable protein without digestive bloat"),

            ("Sunday",
             "Sunday Hearty Breakfast: 2 Aloo-Paneer Parathas + 1 Bowl Dahi + 2 Boiled Eggs",
             "**Sunday Feast:** 3 Phulka Rotis + 1 Bowl Rice + Desi Chicken Curry *(Veg: Paneer Butter Masala)* + Salad + Chaas",
             "1 Fruit Chaat Bowl (Banana, Apple) + 1 Glass Milk with a spoon of Honey",
             "3 Phulka Rotis + 1 Bowl Mix Vegetable Dalia with Paneer / Chicken + Salad",
             "Full systemic replenishment for upcoming week")
        ]

    # Build Header Section with User's Exact Height & Weight Profile
    header = (
        f"### {goal_title}\n\n"
        f"| 👤 Your Profile | ⚖️ Body Mass Index (BMI) | 🎯 Daily Calories | 🥩 Daily Protein | 💧 Water Goal |\n"
        f"|:---|:---|:---|:---|:---|\n"
        f"| **{effective_wt} kg • {ht_str}** | **{bmi}** ({bmi_status}) | **{target_cals}** | **{protein_target}** | **3.5 – 4.0 Litres** |\n\n"
        f"> **Personal Health Note:** {bmi_color_tag}\n\n"
        f"#### 📅 7-Day Ultra-Simple Weekly Schedule (Monday to Sunday):\n\n"
        f"| Day | 🌅 Breakfast | 🥗 Lunch | ☕ Evening Snack | 🍲 Dinner | 💡 Simple Tip |\n"
        f"|:---|:---|:---|:---|:---|:---|\n"
    )

    # Build Table Rows
    rows = "\n".join([
        f"| **{r[0]}** | {r[1]} | {r[2]} | {r[3]} | {r[4]} | *{r[5]}* |"
        for r in weekly_rows
    ])

    # Simple Domestic Portion Guide for Normal Persons
    footer = f"""

---
### 🥄 Ultra-Simple Household Portions (No Kitchen Weighing Scale Needed!):
- **1 Katori (Standard Home Bowl ~150ml):** Use for Dal, Curd (Dahi), Rajma, Chole, or Sabzi.
- **2 to 3 Phulka Rotis:** Medium roti made from wheat flour (dry or minimal ghee).
- **1 Palm-Size Portion:** ~100g Paneer, ~120g Chicken, or 2-3 Boiled Eggs (delivers ~20–25g protein).
- **1 Handful:** Bhuna Chana (Roasted Chana) or roasted peanuts or Makhana.
- **1 Glass (250ml):** Buttermilk (Chaas), toned milk, or plain water.

### 🔄 Easy Veg & Non-Veg Budget Swaps:
- **Eggs** ↔ **Paneer (100g)** ↔ **Boiled Soya Chunks (50g)** ↔ **Sprouted Moong**
- **Chicken / Fish** ↔ **Rajma / Chole / Paneer Curry**
- **Supplements** ↔ **Desi Chana Sattu Drink in Chaas / Water**

### 📌 4 Simple Rules for Normal Persons:
1. **Pani (Water):** Drink 3 to 4 litres of water every day (keep a bottle beside you).
2. **Ghar Ka Khana:** Avoid outside fried snacks (Samosas, Pakoras, Chips, Sweets).
3. **Dinner Time:** Eat dinner at least 2 hours before sleeping.
4. **Workout Synergy:** Complete your 10 reps daily (**Shoulder Press, Pec Dec Fly, Dumbbell Curl**)!
"""
    return header + rows + footer


# ─────────────────────────────────────────────
#  4. Interactive AI Fitness Coach Q&A
# ─────────────────────────────────────────────

def ask_fitness_coach(user_query: str, context: dict = None) -> str:
    """
    Interactive Q&A with the AI Fitness Coach.
    Answers workout form, nutrition, diet plans, recovery, or routine questions.
    Personalizes diet by Height & Weight with BMI.
    ALWAYS formats diet plans in an Ultra-Simple 7-Day Weekly Markdown Table with everyday home foods.
    GUARANTEES unlimited chat access through multi-model failover and rich local intelligence.
    """
    if not user_query.strip():
        return "Ask me anything about your form, workout routine, weekly diet, or recovery!"

    q_lower = user_query.lower()
    is_diet_q = any(k in q_lower for k in (
        "diet", "food", "eat", "meal", "nutrition", "protein", "calorie", "calories",
        "fat loss", "weight loss", "gain", "bulk", "cutting", "breakfast", "dinner",
        "lunch", "veg", "non-veg", "chicken", "paneer", "kg", "height", "belly",
        "table", "dite", "weekly", "week", "simple", "ghar ka khana"
    ))

    if _client is not None:
        try:
            sys_context = ""
            if context:
                sys_context = f"\nUser's Current Workout Stats Today: {context}\n"

            prompt = f"""
You are the AI Gym Trainer Coach, an elite strength conditioning specialist and practical master nutritionist.
{sys_context}
User question: "{user_query}"

CRITICAL MANDATORY FORMATTING RULE FOR DIET & NUTRITION:
- If the user asks for ANY diet plan, meal schedule, weight loss/fat loss, muscle gain/bulking food, weekly diet, vegetarian or non-vegetarian diet:
  1. Detect or ask for user's HEIGHT and WEIGHT. Calculate their BMI (Body Mass Index) and explain if they are underweight, healthy weight, or overweight.
  2. YOU MUST ALWAYS PROVIDE A FULL 7-DAY WEEKLY MEAL SCHEDULE (MONDAY TO SUNDAY) IN A CLEAN MARKDOWN TABLE FORMAT.
  3. MAKE THE DIET AS SIMPLE AS POSSIBLE FOR A NORMAL PERSON:
     - Use ONLY everyday Indian home-cooked foods ('Ghar Ka Khana' — Dal, Roti, Rice, Curd/Dahi, Boiled Eggs, Paneer, Chicken, Soya, Chana, Sabzi).
     - AVOID complicated grams, expensive exotic foods, or requiring kitchen weighing scales. Use simple household domestic measurements ("2 rotis", "1 katori dal", "1 bowl curd", "2 boiled eggs", "1 handful roasted chana").
  
  Required Table Columns:
  | Day | 🌅 Breakfast | 🥗 Lunch | ☕ Evening Snack | 🍲 Dinner | 💡 Simple Tip |
  Include all 7 days: Monday through Sunday with varied, realistic everyday home dishes.
  
  Follow the weekly table with:
  1. Simple Household Portions (1 katori = 150ml, 2 phulkas, 1 palm-sized piece, 1 glass chaas/milk)
  2. Easy Veg & Non-Veg Budget Food Swaps
  3. 4 Simple Rules for Results (Hydration, 10-rep exercise synergy with Dumbbell Curl, Pec Dec Fly, Shoulder Press)

For exercise, form, and recovery questions:
- Provide concise, practical, science-backed coaching cues with bullet points.
- Be encouraging, energetic, and direct.
"""
            result = _gemini_generate(prompt)
            if result:
                # If user asked for a diet plan, verify result contains table markdown pipes
                if is_diet_q and "|" not in result:
                    table_plan = generate_diet_plan_table(user_query, context)
                    return f"{result}\n\n{table_plan}"
                return result
        except Exception:
            pass

    # Seamless Fallback — Guarantees Unlimited High-Quality Chat Answers
    if is_diet_q:
        return generate_diet_plan_table(user_query, context)
    elif "sore" in q_lower or "recovery" in q_lower or "rest" in q_lower:
        return (
            "### 😴 Biomechanical Recovery & Muscle Soreness Protocol\n\n"
            "| Phase | Actionable Recovery Protocol | Impact on Hypertrophy |\n"
            "|:---|:---|:---|\n"
            "| **Sleep Foundation** | 7.5 to 9 hours uninterrupted deep sleep | Drives 90% of nightly human growth hormone (HGH) release |\n"
            "| **Hydration & Electrolytes** | 3.5L water + pinch of pink salt & potassium | Prevents intracellular cramping & restores muscle volume |\n"
            "| **Active Flush** | 20 min low-intensity walking or light cycling | Clears metabolic byproducts & accelerates nutrient delivery |\n"
            "| **Cold/Contrast Therapy** | 3 min warm shower followed by 1 min cold rinse | Decreases systemic inflammation & reduces DOMS intensity |\n\n"
            "> **Coach Tip:** If soreness is localized to shoulders or chest, prioritize gentle dynamic mobility exercises before your next lifting set!"
        )
    elif "routine" in q_lower or "plan" in q_lower or "order" in q_lower:
        return (
            "### 📅 AI Gym Trainer — Daily Workout Synergy Guide\n\n"
            "| Order | Exercise | Target Reps | Primary Muscle Group | Biomechanical Focus |\n"
            "|:---|:---|:---|:---|:---|\n"
            "| **1st** | **Shoulder Press** | 10 Reps | Anterior & Lateral Deltoids | Press vertically with tight core; full overhead extension |\n"
            "| **2nd** | **Pec Dec Fly** | 10 Reps | Sternal Pectoralis Major | Soft elbow bend; squeeze inner chest across midline |\n"
            "| **3rd** | **Dumbbell Curl** | 10 Reps | Biceps Brachii | Pin elbows to ribs; 2-sec eccentric lowering control |\n\n"
            "> **Pro Tip:** Complete all 10 reps per exercise to conquer today's full daily target!"
        )
    elif "curl" in q_lower or "bicep" in q_lower:
        return (
            "### 💪 Dumbbell Curl — Form Mastery\n"
            "- **Elbow Position:** Pin elbows tightly against your ribcage. Do NOT swing shoulders to create momentum.\n"
            "- **Peak Contraction:** Squeeze biceps forcefully at the top for 1 full second.\n"
            "- **Eccentric Tempo:** Lower the weights slowly over 2–3 seconds to maximize muscular hypertrophy."
        )
    elif "pec" in q_lower or "chest" in q_lower or "fly" in q_lower:
        return (
            "### 🦅 Pec Dec Fly — Form Mastery\n"
            "- **Elbow Angle:** Maintain a soft, fixed 15-degree bend in your elbows throughout the movement.\n"
            "- **Chest Squeeze:** Bring your hands together and slightly overlap them at the peak to engage inner pectorals.\n"
            "- **Shoulder Safety:** Avoid letting the arms stretch excessively far back behind your torso."
        )
    elif "shoulder" in q_lower or "press" in q_lower:
        return (
            "### 🛡️ Shoulder Press — Form Mastery\n"
            "- **Core Bracing:** Tighten abs and glutes to avoid hyperextending your lumbar spine.\n"
            "- **Overhead Path:** Press upward until biceps align with your ears at peak lockout.\n"
            "- **Control:** Lower under strict control until elbows hit approximately 90 degrees."
        )
    else:
        return (
            f"### 🤖 AI Fitness Coach Online\n\n"
            f"You asked: *\"{user_query}\"*\n\n"
            "**Key Coaching Principles:**\n"
            "1. **Form Over Weight:** A strict 2-second negative eccentric tempo builds 40% more muscle than momentum.\n"
            "2. **Daily Consistency:** Complete your daily targets (Dumbbell Curls, Pec Dec Fly, Shoulder Press).\n"
            "3. **Ask Anything:** You can ask for a **complete daily diet plan (in table format)**, form corrections, protein targets, or recovery tips anytime — unlimited chats available!"
        )


# ─────────────────────────────────────────────
#  5. Interactive Terminal Console
# ─────────────────────────────────────────────

def run_coach_console(daily_summary: dict = None):
    """
    Launches an interactive AI Fitness Coach terminal session.
    Allows user to ask questions, view daily strategy, or study biomechanics guides.
    """
    status = get_coach_status()
    model_name = status["model"]
    is_online = status["configured"]

    print("\n" + "=" * 64)
    print("       AI GYM TRAINER  —  AI FITNESS COACH CONSOLE")
    print("=" * 64)
    if is_online:
        print(f"  [STATUS] ONLINE  —  Google Gemini Cascade ({model_name})")
        print("  [FEATURE] UNLIMITED CHATS & TABLE-FORMATTED DIET PLANS ACTIVE")
    else:
        print("  [STATUS] OFFLINE INTELLIGENT COACH (Unlimited Local Mode)")
    print("-" * 64)

    if daily_summary:
        done = daily_summary.get("total_done", 0)
        target = daily_summary.get("total_target", 30)
        pct = int(daily_summary.get("percentage", 0.0) * 100)
        print(f"  TODAY'S WORKOUT PROGRESS: {done} / {target} Reps ({pct}%)")
        rem = daily_summary.get("remaining_exercises", [])
        if rem:
            rem_str = ", ".join([f"{e['name']} ({e['needed']} reps left)" for e in rem])
            print(f"  REMAINING TODAY: {rem_str}")
        else:
            print(f"  STATUS: ALL {target} REPS COMPLETED! ULTRA INSTINCT MASTERED!")
        print("-" * 64)
        print("  [DAILY COACH BRIEFING]:")
        brief = generate_daily_brief(daily_summary)
        print(f"  {brief}")
        print("-" * 64)

    print("\nAvailable Commands:")
    print("  [1] Ask a custom question (form, nutrition, rest, routines)")
    print("  [2] View Biomechanical Form Cues (all 3 exercises)")
    print("  [3] Post-Workout Nutrition & Diet Plan Table")
    print("  [4] Muscle Soreness & Recovery Protocol")
    print("  [5] Refresh Daily Workout Brief")
    print("  [0] Return to Dashboard (or type 'back' / 'exit')")
    print("  Or: Simply type any fitness or diet question directly!")
    print("=" * 64)

    while True:
        try:
            choice = input("\n[Coach] Enter command or ask question > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Returning to Dashboard...")
            break

        if not choice:
            continue

        if choice.lower() in ("0", "back", "exit", "quit", "q", "menu"):
            print("[INFO] Returning to Dashboard...")
            break

        if choice == "1":
            try:
                q = input("[Coach Question] Type your fitness question: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if q:
                print("\nConsulting AI Coach...")
                ans = ask_fitness_coach(q, context=daily_summary)
                print(f"\n{ans}\n")

        elif choice == "2":
            print("\n--- BIOMECHANICAL FORM GUIDES ---")
            print("1. DUMBBELL CURLS:")
            print("   - Pin elbows to ribcage. Avoid swinging shoulders to lift the dumbbell.")
            print("   - Squeeze biceps at peak for 1 second; lower weights over 2-3 seconds.")
            print("2. PEC DEC FLY:")
            print("   - Maintain a slight, fixed bend in elbows (~15 degrees).")
            print("   - Bring hands together and cross slightly to maximize sternal pec contraction.")
            print("3. SHOULDER PRESS:")
            print("   - Brace your core to prevent lower back hyperextension.")
            print("   - Press vertically until biceps align with ears; lower under control.")

        elif choice == "3":
            print("\n--- ULTRA-SIMPLE 7-DAY WEEKLY DIET PLAN (Ghar Ka Khana) ---")
            try:
                wt_inp = input("Enter your weight in kg [e.g. 75, or press Enter]: ").strip()
                ht_inp = input("Enter your height in cm or ft [e.g. 175 or 5'9, or press Enter]: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            query = f"simple weekly diet plan {wt_inp} {ht_inp}"
            diet_table = generate_diet_plan_table(query, context=daily_summary)
            print(f"\n{diet_table}\n")

        elif choice == "4":
            print("\n--- MUSCLE SORENESS & RECOVERY PROTOCOL ---")
            print("  - Sleep: 7 to 9 hours of uninterrupted sleep drives 90% of growth hormone release.")
            print("  - Active Recovery: 15-20 min light walk improves blood circulation and clears lactic waste.")
            print("  - Cold/Heat Therapy: Contrast showers stimulate lymphatic drainage and ease DOMS.")

        elif choice == "5":
            if daily_summary:
                print("\nRefreshing Daily Brief...")
                brief = generate_daily_brief(daily_summary)
                print(f"\n{brief}\n")
            else:
                print("[INFO] No daily summary loaded.")

        else:
            # User typed a direct question
            print("\nConsulting AI Coach...")
            ans = ask_fitness_coach(choice, context=daily_summary)
            print(f"\n{ans}\n")


