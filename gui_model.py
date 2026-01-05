import os
import tkinter as tk
from tkinter import ttk
from clips import Environment

from PIL import Image, ImageTk
import pandas as pd
import joblib
#import numpy as np

# ==========================================
# Load ML model and columns
# ==========================================
try:
    # Load ML model
    ml = joblib.load("tomato_disease_model.pkl")

    # Load columns from the data file to display symptoms name in list
    df = pd.read_csv("PC_SYM_2.csv")
    symptom_columns = df.columns[1:].tolist()

    # Get all possible disease names from the model's classes
    all_diseases = ml.classes_.tolist()

except Exception as e:
    # if the model/data is missing
    print(f"Error loading model or data: {e}")
    ml = None
    symptom_columns = []
    all_diseases = []

# ==========================================
# Initialize CLIPS environment
# ==========================================
env = Environment()
try:
    env.load("clip_rules.clp")
except Exception as e:
    print(f"Error loading CLIPS rules: {e}")

# ==========================================
# Disease Info
# ==========================================
disease_info = {
    "Early-blight": {"treatment": "Use copper-based fungicide. Remove affected leaves.",
                     "prevention": "Avoid overhead watering and rotate crops annually."},
    "late-blight": {"treatment": "Apply fungicides with chlorothalonil. Remove infected plants.",
                    "prevention": "Avoid wet conditions. Use resistant tomato varieties."},
    "Septoria-leaf-spot": {"treatment": "Spray with copper fungicide weekly until controlled.",
                           "prevention": "Use disease-free seeds and practice crop rotation."},
    "Bacterial-Spot": {"treatment": "Use copper-based bactericide. Remove infected plants.",
                       "prevention": "Avoid working with wet plants and sanitize tools."},
    "mosaic-virus": {"treatment": "No chemical cure. Remove and destroy infected plants immediately.",
                            "prevention": "Sanitize tools regularly. Use virus-free seeds."},
    "fusarium-wilt": {"treatment": "No cure. Remove infected plants. Solarize soil.",
                      "prevention": "Use Fusarium wilt-resistant varieties."},
    "Yellow-Leaf-Curl-Virus": {"treatment": "No cure. Remove and destroy infected plants.",
                                      "prevention": "Control whiteflies (the vector) using insecticides or netting."},
    "Root-knot-nematode": {"treatment": "Soil solarization or application of biological nematicides.",
                           "prevention": "Plant resistant varieties or practice crop rotation with non-hosts."},
    "Blossom-End-Rot": {"treatment": "Apply calcium foliar sprays immediately. Adjust soil pH.",
                        "prevention": "Ensure consistent watering and proper soil calcium levels."},
    "southern-blight": {"treatment": "Remove infected plants and apply fungicides to the soil.",
                        "prevention": "Deep plowing or soil solarization to reduce fungal spores."},
    "Anthracnose": {"treatment": "Copper sprays, and apply tomato fungicide to the entire crop at the first sign of infection.",
                "prevention": "Remove the lower leaves to prevent contact with the soil and maintain a good weed control."},
    "healthy": {"treatment": "Maintain regular watering and fertilization schedule.",
                "prevention": "Monitor plants weekly for early signs of disease."}
}

# ==========================================
# Translation Dictionary
# ==========================================
translations = {
    "en": {
        "title": "Hybrid Tomato Disease Diagnoser",
        "select_symptoms": "Select Symptoms Observed:",
        "diagnose_btn": "Diagnose",
        "result_label": "Diagnose Result",
        "disease": "Disease Name",
        "ml_label": "ML Confidence",
        "clips_label": "CLIPS Confidence",
        "final_trust": "Final Trust Score (FTS)",
        "treatment": "Treatment",
        "prevention": "Prevention",
        "no_match": "No matching disease found",
        "reset_btn": "Reset",
        "exit_btn": "Exit",
    },
    "ar": {
        "title": "نظام هجين لتشخيص أمراض الطماطم",
        "select_symptoms": "اختر الأعراض:",
        "diagnose_btn": "تشخيص",
        "result_label": "نتيجة التشخيص",
        "disease": "المرض",
        "ml_label": "نتيجة تعلم الآلة",
        "clips_label": "نتيجة نظام كليبس",
        "final_trust": "النتيجة النهائية (FTS)",
        "treatment": "العلاج",
        "prevention": "الوقاية",
        "no_match": "لا يوجد تطابق",
        "reset_btn": "إعادة ضبط",
        "exit_btn": "إنهاء",
    }
}

current_lang = "en"


# ==========================================
#  CLIPS Logic
# ==========================================

# Asserts selected symptoms into CLIPS, runs, and retrieves CF for all asserted diseases
def get_clips_diagnosis(selected_symptoms):
    if not env:
        return {}

    env.reset()

    for symptom in selected_symptoms:
        try:
            env.assert_string(f'(symptom (name {symptom}))')
        except Exception as e:
            print(f"Error asserting symptom {symptom}: {e}")

    env.run()

    clips_results = {}

    for fact in env.facts():
        if fact.template.name == "disease":
            disease_name = fact["name"]

            try:
                # convert CF to percentage (0-100%)
                confidence_factor = float(fact["confidence"]) * 100

                # Take the MAX confidence and the strongest evidence
                if disease_name in clips_results:
                    clips_results[disease_name] = max(clips_results[disease_name], confidence_factor)
                else:
                    clips_results[disease_name] = confidence_factor

            except Exception as e:
                print(f"Error processing CLIPS fact for {disease_name}: {e}")
                clips_results[disease_name] = 0

    return clips_results


# ==========================================
# Utility Functions
# ==========================================

def reset_selections():
    symptom_listbox.selection_clear(0, tk.END)
    # clear image
    image_label.config(image='', text="")  
    diagnose()


def exit_app():
    app.destroy()


# ==========================================
#  Diagnosis Logic
# ==========================================

def diagnose():
    if not ml:
        print("ML Model not loaded. Cannot proceed with diagnosis.")
        return

    # Get selected symptoms from the listbox
    selected_symptoms_keys = [symptom_columns[i] for i in symptom_listbox.curselection()]

    # Clear previous results
    for row in tree.get_children():
        tree.delete(row)

    # Handle the "Healthy" case when no symptoms are selected
    if not selected_symptoms_keys:
        info = disease_info.get("healthy", {"treatment": "-", "prevention": "-"})

        # Insert a single, clear row for Healthy. values are FTS, ML, CLIPS, Treatment, Prevention
        tree.insert("", "end", text="Healthy",
                    values=("100.0%", "100.0%", "100.0%", info["treatment"], info["prevention"]),
                    tags=('high',))
        return

    # Prepare input for ML Model
    ml_input = {col: [1 if col in selected_symptoms_keys else 0] for col in symptom_columns}
    ml_df = pd.DataFrame(ml_input, columns=symptom_columns)

    # Get ML Predictions
    probabilities = ml.predict_proba(ml_df)[0]
    ml_results = {disease: prob * 100 for disease, prob in zip(ml.classes_, probabilities)}

    # Get CLIPS Expert Diagnosis
    clips_results = get_clips_diagnosis(selected_symptoms_keys)

    # Combine ML and CLIPS Results & Calculate Final Trust Score (FTS)
    # wieghts of prediction (why these numbers specificly?)
    W_ML = 0.6
    W_CF = 0.4

    combined_results = []

    all_unique_diseases = set(ml_results.keys()) | set(clips_results.keys())

    for disease in all_unique_diseases:
        if disease == "healthy":
            continue

        ml_conf = ml_results.get(disease, 0.0)
        clips_conf = clips_results.get(disease, 0.0)

        # Calculate Final Trust Score (FTS) use weight
        fts = (ml_conf * W_ML) + (clips_conf * W_CF)

        if fts >= 0.1:
            combined_results.append({
                "disease": disease,
                "ml_conf": ml_conf,
                "clips_conf": clips_conf,
                "fts": fts
            })

    # Sort results by Final Trust Score (highest first)
    combined_results.sort(key=lambda x: x["fts"], reverse=True)


    # Display Results
    # If symptoms were selected but nothing matched significantly
    if not combined_results:
        tree.insert("", "end", text=translations[current_lang]["no_match"],
                    values=("-", "-", "-", "Monitor closely.", "Re-check symptoms."), tags=('low',))
        return

    # Insert results in Treeview
    for result in combined_results:
        disease = result["disease"]
        ml_conf = result["ml_conf"]
        clips_conf = result["clips_conf"]
        fts = result["fts"]

        info = disease_info.get(disease, {"treatment": "-", "prevention": "-"})

        # Determine tag for coloring based on the Final Trust Score
        if fts >= 75:
            tag = "high"
        elif fts >= 40:
            tag = "medium"
        else:
            tag = "low"

        # Insert the row with all values 
        tree.insert("", "end", text=disease.replace('-', ' ').title(),
                    values=(f"{fts:.1f}%", f"{ml_conf:.1f}%", f"{clips_conf:.1f}%", info["treatment"],
                            info["prevention"]),
                    tags=(tag,))


# Updates all GUI elements with the current language translations
def update_language():
    app.title(translations[current_lang]["title"])
    symptom_label.config(text=translations[current_lang]["select_symptoms"])
    result_label.config(text=translations[current_lang]["result_label"])
    diagnose_button.config(text=translations[current_lang]["diagnose_btn"])
    reset_button.config(text=translations[current_lang]["reset_btn"])
    exit_button.config(text=translations[current_lang]["exit_btn"])
    lang_label.config(text=translations[current_lang].get("lang_label", "Language:"))
    

    # --- Treeview Column Configuration ---

    # Disease Name
    tree.heading("#0", text=translations[current_lang]["disease"])
    tree.column("#0", width=200, stretch=tk.NO, anchor="w", minwidth=150)

    # Final Trust Score (FTS)
    tree.heading("fts_score", text=translations[current_lang]["final_trust"])
    tree.column("fts_score", width=120, stretch=tk.NO, anchor="center", minwidth=80)

    # ML Confidence
    tree.heading("ml_confidence", text=translations[current_lang]["ml_label"])
    tree.column("ml_confidence", width=120, stretch=tk.NO, anchor="center", minwidth=80)

    # CLIPS Confidence
    tree.heading("clips_confidence", text=translations[current_lang]["clips_label"])
    tree.column("clips_confidence", width=120, stretch=tk.NO, anchor="center", minwidth=80)

    # Treatment
    tree.heading("treatment", text=translations[current_lang]["treatment"])
    tree.column("treatment", width=280, anchor="w", stretch=tk.YES, minwidth=150)

    # Prevention
    tree.heading("prevention", text=translations[current_lang]["prevention"])
    tree.column("prevention", width=280, anchor="w", stretch=tk.YES, minwidth=150)


def switch_language(lang):
    """Switches the application language."""
    global current_lang
    current_lang = lang
    update_language()

###################################
# Track previously selected indices to detect the most recently clicked symptom
###################################

previous_selection = set()

# to display image of symptoms
def show_last_selected_symptom_image():
    global current_symptom_image, previous_selection

    # Get all currently selected indices
    current_selection = set(symptom_listbox.curselection())
    
    if not current_selection:
        # clear image
        image_label.config(image='', text="")  
        previous_selection = set()
        return

    # find newly added symptoms
    newly_added = current_selection - previous_selection
    
    # Determine which symptom to display
    if newly_added:
        # Show the most recently added symptom
        # if multiple added somehow, pick one
        last_index = max(newly_added)  
    else:
        # If deselecting, show the last remaining selected symptom
        last_index = max(current_selection)
    
    # Update previous selection for next comparison
    previous_selection = current_selection

    # Get the selected symptom
    symptom_key = symptom_columns[last_index]

    img_path = os.path.join("images", f"{symptom_key}.jpg")

    if not os.path.exists(img_path):
        image_label.config(text=f"No image for: {symptom_key}", image='')
        return

    # Load & resize image
    img = Image.open(img_path)
    img = img.resize((250, 250))
    current_symptom_image = ImageTk.PhotoImage(img)

    # Display image
    image_label.config(image=current_symptom_image, text="")


# ==========================================
# GUI Setup (Tkinter)
# ==========================================

# Main application window
app = tk.Tk()
app.title(translations[current_lang]["title"])
app.resizable(True, True)

style = ttk.Style(app)
style.theme_use('clam')

# Configure Treeview style
style.configure('Treeview.Heading', font=('Arial', 10))
style.configure('Treeview', rowheight=25)

# Accent button style
style.configure('Accent.TButton', font=('Arial', 10, 'bold'), foreground='white', background='#4CAF50')
style.map('Accent.TButton', background=[('active', '#66BB6A')])

# Main frame
main_frame = ttk.Frame(app, padding="15")
main_frame.pack(padx=20, pady=20, fill="both", expand=True)


# Language selection frame
lang_frame = ttk.Frame(main_frame)
lang_frame.pack(fill='x', pady=5, anchor='w')

lang_label = ttk.Label(lang_frame, text="Language:")
lang_label.pack(side=tk.LEFT, padx=(0, 10))

ttk.Button(lang_frame, text="English", command=lambda: switch_language("en")).pack(side=tk.LEFT, padx=5)
ttk.Button(lang_frame, text="العربية", command=lambda: switch_language("ar")).pack(side=tk.LEFT, padx=5)

# ------------------------------------------
# Symptoms Selection Section
# ------------------------------------------
symptom_label = ttk.Label(main_frame, text=translations[current_lang]["select_symptoms"], font=('Arial', 12, 'bold'))
symptom_label.pack(pady=(10, 5),padx=30, anchor='w')

# Horizontal frame to hold listbox (left) and image (right)
selection_frame = ttk.Frame(main_frame)
selection_frame.pack(fill="both",padx=30, expand=True)

# Container for list of symptoms
list_frame = ttk.Frame(selection_frame)
list_frame.pack(side=tk.LEFT, fill="y",pady=5)

scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

symptom_listbox = tk.Listbox(
    list_frame,
    selectmode=tk.MULTIPLE,
    height=14,
    width=80,
    yscrollcommand=scrollbar.set,
    exportselection=0
)

symptom_listbox.pack(side=tk.LEFT, fill="both", expand=True)
scrollbar.config(command=symptom_listbox.yview)

# Display selected image
image_frame = ttk.Frame(selection_frame)
image_frame.pack(side=tk.LEFT, padx=20, fill="both", expand=True)
# Call method to display last selected image
symptom_listbox.bind("<<ListboxSelect>>", lambda e: show_last_selected_symptom_image())

image_label = ttk.Label(image_frame)
image_label.pack(pady=10)
current_symptom_image = None  


if symptom_columns:
    display_symptoms = [s.replace('-', ' ').title() for s in symptom_columns]
    for symptom in display_symptoms:
        symptom_listbox.insert(tk.END, symptom)
else:
    symptom_listbox.insert(tk.END, "Error: Symptoms data not loaded.")


# --- Button Frame (Diagnose, Reset, Exit) ---
button_frame = ttk.Frame(main_frame, padding="0 0 0 15")
button_frame.pack(fill="x", pady=10)

diagnose_button = ttk.Button(button_frame, text=translations[current_lang]["diagnose_btn"], command=diagnose,
                             style='Accent.TButton')
diagnose_button.pack(side=tk.LEFT, padx=(0, 15))

reset_button = ttk.Button(button_frame, text=translations[current_lang]["reset_btn"], command=reset_selections)
reset_button.pack(side=tk.LEFT, padx=5)

exit_button = ttk.Button(button_frame, text=translations[current_lang]["exit_btn"], command=exit_app)
exit_button.pack(side=tk.RIGHT, padx=0)

# ------------------------------------------
# Results Table Section
# ------------------------------------------

result_label = ttk.Label(main_frame, text=translations[current_lang]["result_label"], font=('Arial', 12, 'bold'))
result_label.pack(pady=(10, 5), anchor='w')

result_frame = ttk.Frame(main_frame)
result_frame.pack(fill="both", expand=True)

# Define columns
columns = ("fts_score", "ml_confidence", "clips_confidence", "treatment", "prevention")
tree = ttk.Treeview(result_frame, columns=columns, show="tree headings", height=10)

# Tags for row coloring
tree.tag_configure('high', background='#c8e6c9', foreground='#2e7d32')  # Light Green (High Confidence)
tree.tag_configure('medium', background='#fff9c4', foreground='#fbc02d')  # Light Yellow (Medium Confidence)
tree.tag_configure('low', background='#ffcdd2', foreground='#c62828')  # Light Red (Low Confidence)

tree.pack(fill="both", expand=True)

# Add a vertical scrollbar
vsb = ttk.Scrollbar(result_frame, orient="vertical", command=tree.yview)
vsb.pack(side='right', fill='y')
tree.configure(yscrollcommand=vsb.set)

# Finalize setup
update_language()
diagnose()

app.mainloop()
