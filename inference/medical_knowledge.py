"""Comprehensive Medical Knowledge Base & Clinical Inference Engine for Pathology AI.

Contains deep medical definitions, reference ranges, etiology (causes of high/low),
evidence-based dietary and lifestyle recommendations, condition monographs,
test preparation protocols, and symptom correlation logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. BIOMARKERS KNOWLEDGE GRAPH (60+ Clinical Biomarkers)
# ---------------------------------------------------------------------------

BIOMARKERS_DB: Dict[str, Dict[str, Any]] = {
    "Hemoglobin": {
        "aliases": ["hb", "haemoglobin", "hgb", "haemogloblin", "hemogloblin", "hemoglobin level", "haemoglobin level"],
        "category": "Hematology / CBC",
        "role": "An iron-rich protein in red blood cells that transports oxygen from your lungs to all tissues and organs.",
        "normal_range": "Men: 13.5 - 17.5 g/dL | Women: 12.0 - 15.5 g/dL",
        "high_causes": "Dehydration, smoking, chronic hypoxia (COPD, high altitude living), polycythemia vera, heavy endurance training.",
        "low_causes": "Iron deficiency anemia, vitamin B12 or folate deficiency, chronic blood loss (e.g. gastrointestinal, heavy menstruation), kidney disease, bone marrow suppression, chronic inflammatory diseases.",
        "diet_lifestyle": "• **Iron-Rich Foods:** Incorporate dark green leafy vegetables (spinach, kale, moringa), lentils, chickpeas, pumpkin seeds, beetroots, pomegranate, black raisins, dates, and jaggery. For non-vegetarians, lean poultry, fish, and eggs provide bioavailable heme iron.\n• **Absorption Boosters:** Pair iron meals with **Vitamin C** (fresh lemon juice, amla/Indian gooseberry, oranges, tomatoes) which multiplies intestinal iron absorption by up to 3x.\n• **Inhibitors to Avoid:** Do NOT drink tea, coffee, green tea, or cola within 1 to 2 hours of iron-rich meals (tannins and polyphenols strongly inhibit iron uptake). Avoid taking calcium supplements or dairy simultaneously.\n• **Vital Co-Factors:** Ensure adequate Vitamin B12 (dairy, eggs, fish) and Folate/Vitamin B9 (beans, greens) necessary for healthy red blood cell production.",
        "doctor_questions": "Ask your doctor: Should we check a Complete Iron Profile (Ferritin, TIBC, Transferrin Saturation) and Vitamin B12/Folate to confirm the exact etiology of anemia?",
    },
    "RBC Count": {
        "aliases": ["red blood cell", "red blood cells", "erythrocyte count", "rbc"],
        "category": "Hematology / CBC",
        "role": "The total number of oxygen-carrying erythrocytes circulating per microliter of blood.",
        "normal_range": "Men: 4.5 - 5.9 million/mcL | Women: 4.0 - 5.2 million/mcL",
        "high_causes": "Polycythemia vera, severe dehydration, congenital heart defects, pulmonary fibrosis, living at high altitudes.",
        "low_causes": "Anemia (nutritional, hemolytic, aplastic), acute or chronic hemorrhage, kidney failure (low erythropoietin), bone marrow disorders.",
        "diet_lifestyle": "Ensure balanced nutrition including iron, vitamin B12, folate, and copper. Avoid smoking.",
        "doctor_questions": "Should we investigate erythropoietin (EPO) levels or bone marrow function?",
    },
    "Total Leucocyte Count": {
        "aliases": ["wbc", "wbc count", "white blood cells", "leucocytes", "tlc"],
        "category": "Hematology / CBC",
        "role": "The primary cellular defense of the immune system that neutralizes infections and removes cellular debris.",
        "normal_range": "4,000 - 11,000 cells/mcL",
        "high_causes": "Bacterial or severe viral infection, acute systemic inflammation, tissue necrosis, severe physical/emotional stress, corticosteroid use, leukemia.",
        "low_causes": "Viral infections (dengue, influenza, HIV), autoimmune conditions (lupus), bone marrow damage, chemotherapy/radiation, severe sepsis.",
        "diet_lifestyle": "Focus on immune-supporting foods rich in Vitamin C, zinc, garlic, turmeric, and quality protein. Maintain strict hand hygiene if count is low.",
        "doctor_questions": "Is this elevation reactive to an infection, or does it require a repeat test with differential count in 1-2 weeks?",
    },
    "Platelet Count": {
        "aliases": ["platelets", "thrombocytes", "plt"],
        "category": "Hematology / CBC",
        "role": "Disc-shaped cell fragments essential for primary hemostasis (blood clotting and stopping bleeding).",
        "normal_range": "150,000 - 450,000 /mcL",
        "high_causes": "Essential thrombocythemia, reactive thrombocytosis due to acute infection/inflammation, iron deficiency anemia, post-splenectomy.",
        "low_causes": "Immune thrombocytopenic purpura (ITP), dengue or viral fever, drug-induced destruction (heparin, NSAIDs), liver cirrhosis, hypersplenism.",
        "diet_lifestyle": "• **If Low:** Avoid contact sports or activities with high injury risk; avoid self-medicating with aspirin or ibuprofen. Papaya leaf extract and vitamin C support platelet recovery.\n• **If High:** Ensure optimal hydration to lower blood viscosity.",
        "doctor_questions": "Do I need a peripheral blood smear examination to rule out platelet clumping (pseudothrombocytopenia)?",
    },
    "Plateletcrit": {
        "aliases": ["pct", "plateletcrit", "plateletcrit pct", "platelet crit", "platelet hematocrit"],
        "category": "Hematology / Platelet Indices (CBC)",
        "role": "Plateletcrit (PCT) is the volume percentage of whole blood occupied by platelets, calculated as (Platelet Count × Mean Platelet Volume) / 10,000. It is the platelet equivalent of the red blood cell Hematocrit (PCV). Note: In critical care or biochemistry, PCT can also refer to Procalcitonin (a biomarker for bacterial infection and sepsis).",
        "normal_range": "0.10% – 0.40% (typically 0.16% – 0.36%)",
        "high_causes": "Essential thrombocythemia, reactive thrombocytosis from acute or chronic systemic inflammation, severe bacterial infection, tissue damage, iron deficiency anemia, or post-splenectomy.",
        "low_causes": "Thrombocytopenia (low circulating platelets), impaired bone marrow production (aplastic anemia, leukemia, myelodysplastic syndrome), chemotherapy/radiotherapy, immune-mediated destruction (ITP), drug toxicity (heparin, NSAIDs), severe viral infections (dengue, HIV).",
        "diet_lifestyle": "• **If Low:** Avoid contact sports or sharp trauma. Do not take aspirin, ibuprofen, or blood-thinning NSAIDs without physician authorization. Eat foods rich in vitamin C, folate, and B12 to support platelet production.\n• **If High:** Maintain generous daily hydration (2–2.5 L) to lower blood viscosity and thrombotic risk.",
        "doctor_questions": "Does my Plateletcrit (PCT) result correlate with my total Platelet Count and MPV (Mean Platelet Volume), and is a peripheral blood smear needed to evaluate platelet morphology?",
    },
    "Mean Platelet Volume": {
        "aliases": ["mpv", "mean platelet volume", "platelet volume"],
        "category": "Hematology / Platelet Indices (CBC)",
        "role": "Measures the average physical size of circulating platelets. Younger platelets newly released from bone marrow are larger and metabolically more active than older platelets.",
        "normal_range": "7.4 – 10.4 fL (analyzer-dependent: 7.0 – 11.5 fL)",
        "high_causes": "Rapid platelet turnover or destruction where bone marrow compensates by releasing large immature platelets (e.g. Immune Thrombocytopenic Purpura - ITP, recovery from bone marrow suppression, myeloproliferative disorders, pre-eclampsia, cardiovascular disease).",
        "low_causes": "Impaired bone marrow production (aplastic anemia, cytotoxic chemotherapy, viral infections), hypersplenism, or autoimmune destruction of megakaryocytes.",
        "diet_lifestyle": "Maintain optimal anti-inflammatory nutrition with antioxidants, vitamin C, and omega-3 fatty acids. Avoid smoking which increases platelet reactivity.",
        "doctor_questions": "Does my MPV suggest increased platelet turnover/destruction or impaired marrow synthesis?",
    },
    "Platelet Distribution Width": {
        "aliases": ["pdw", "platelet distribution width"],
        "category": "Hematology / Platelet Indices (CBC)",
        "role": "Measures the degree of variation in platelet size (platelet anisocytosis), indicating whether circulating platelets are uniform or varied in dimension.",
        "normal_range": "9.0% – 17.0% (or 25 – 65 fL depending on assay)",
        "high_causes": "Active bone marrow compensation releasing platelets of various ages and sizes, reactive thrombocytosis, myeloproliferative disorders, or microvascular thrombosis.",
        "low_causes": "Platelets are exceptionally uniform in size; generally normal or seen in marrow hypoplasia.",
        "diet_lifestyle": "Ensure adequate daily water intake and a nutrient-dense whole-food diet.",
        "doctor_questions": "Does elevated PDW correlate with my MPV and Platelet Count on the automated differential?",
    },
    "Platelet Large Cell Ratio": {
        "aliases": ["p-lcr", "plcr", "platelet large cell ratio"],
        "category": "Hematology / Platelet Indices (CBC)",
        "role": "The percentage of circulating platelets that are larger than 12 fL in volume, serving as a marker for newly synthesized reticulated platelets.",
        "normal_range": "15.0% – 35.0%",
        "high_causes": "Increased thrombopoiesis, immune thrombocytopenia (ITP), essential thrombocythemia, hypercoagulable states.",
        "low_causes": "Decreased production of large platelets, marrow suppression, chemotherapy effect.",
        "diet_lifestyle": "Balanced nutrition rich in vitamin B12, folate, and vitamin C.",
        "doctor_questions": "Does my P-LCR suggest accelerated marrow release of reticulated platelets?",
    },
    "Packed Cell Volume": {
        "aliases": ["pcv", "hematocrit", "hct", "haematocrit"],
        "category": "Hematology / CBC",
        "role": "The percentage of total blood volume made up of packed red blood cells.",
        "normal_range": "Men: 40% - 52% | Women: 36% - 48%",
        "high_causes": "Dehydration (hemoconcentration), hypoxia, smoking, polycythemia.",
        "low_causes": "Anemia, overhydration (fluid overload), chronic blood loss, hemolysis.",
        "diet_lifestyle": "Keep well-hydrated. Address underlying iron or B-vitamin deficiencies.",
        "doctor_questions": "Does the low PCV correlate with MCV and MCH indices?",
    },
    "MCV": {
        "aliases": ["mean corpuscular volume"],
        "category": "Hematology / CBC (RBC Indices)",
        "role": "Measures the average physical size of your red blood cells.",
        "normal_range": "80 - 100 fL",
        "high_causes": "Macrocytosis: Vitamin B12 deficiency, folate deficiency, chronic liver disease, alcoholism, hypothyroidism, myelodysplastic syndrome.",
        "low_causes": "Microcytosis: Iron deficiency anemia, thalassemia trait/minor, anemia of chronic disease, lead poisoning.",
        "diet_lifestyle": "• **If MCV < 80 (Microcytic):** Iron-rich nutrition, vitamin C, investigate source of blood loss.\n• **If MCV > 100 (Macrocytic):** B12 supplements, fortified nutritional yeast, green leafy vegetables (folate).",
        "doctor_questions": "Would a serum ferritin or hemoglobin electrophoresis (Hb HPLC) be appropriate to differentiate iron deficiency from thalassemia trait?",
    },
    "MCH": {
        "aliases": ["mean corpuscular hemoglobin"],
        "category": "Hematology / CBC (RBC Indices)",
        "role": "The average mass of hemoglobin inside a single red blood cell.",
        "normal_range": "27 - 33 pg",
        "high_causes": "Macrocytic anemias (B12/folate deficiency).",
        "low_causes": "Microcytic hypochromic anemias (iron deficiency, thalassemia).",
        "diet_lifestyle": "Follow targeted nutritional guidance based on whether MCV is high or low.",
        "doctor_questions": "How does this compare with the overall red cell morphology?",
    },
    "MCHC": {
        "aliases": ["mean corpuscular hemoglobin concentration"],
        "category": "Hematology / CBC (RBC Indices)",
        "role": "The concentration of hemoglobin in a given volume of packed red blood cells.",
        "normal_range": "32 - 36 g/dL",
        "high_causes": "Hereditary spherocytosis, cold agglutinin disease, severe burns.",
        "low_causes": "Iron deficiency anemia, sideroblastic anemia.",
        "diet_lifestyle": "Diet rich in iron, zinc, and lean proteins.",
        "doctor_questions": "Does the low MCHC confirm hypochromic red blood cells on smear?",
    },
    "RDW-CV": {
        "aliases": ["rdw", "red cell distribution width", "rdw cv", "rdw sd"],
        "category": "Hematology / CBC",
        "role": "Measures the variation in size among your circulating red blood cells (anisocytosis).",
        "normal_range": "11.5% - 14.5%",
        "high_causes": "Mixed nutritional deficiency (iron + B12), early iron deficiency anemia, recent blood transfusion, hemolytic anemia.",
        "low_causes": "Very uniform cell sizes (usually clinically insignificant or pure thalassemia trait).",
        "diet_lifestyle": "Investigate both iron and B-complex vitamin statuses.",
        "doctor_questions": "Does the high RDW help differentiate iron deficiency (high RDW) from thalassemia trait (often normal RDW)?",
    },
    "Neutrophils": {
        "aliases": ["neutrophil", "neutrophils %", "anc", "absolute neutrophil count"],
        "category": "Hematology / Differential WBC",
        "role": "First-responder white blood cells specialized in engulfing and destroying pyogenic bacteria.",
        "normal_range": "40% - 75% (or 2,000 - 7,000 /mcL)",
        "high_causes": "Acute bacterial infection, tissue injury, burns, physical stress, myocardial infarction, steroid medications.",
        "low_causes": "Viral infections, severe sepsis (neutropenia), drug toxicity, autoimmune destruction.",
        "diet_lifestyle": "Eat well-cooked foods and maintain clean hygiene if neutropenic.",
        "doctor_questions": "Do I show any active signs of bacterial infection or tissue inflammation?",
    },
    "Lymphocyte %": {
        "aliases": ["lymphocytes", "lymphocyte", "absolute lymphocyte count", "alc"],
        "category": "Hematology / Differential WBC",
        "role": "Specialized immune cells (T-cells, B-cells, NK-cells) vital for viral defense and antibody production.",
        "normal_range": "20% - 45% (or 1,000 - 4,000 /mcL)",
        "high_causes": "Viral infections (mononucleosis, cytomegalovirus, hepatitis), chronic lymphocytic leukemia (CLL), tuberculosis.",
        "low_causes": "Immunodeficiency states, acute corticosteroid therapy, advanced HIV, malnutrition, lupus.",
        "diet_lifestyle": "Adequate protein, zinc, selenium, and vitamin D support healthy lymphocyte proliferation.",
        "doctor_questions": "Is this relative or absolute lymphocytosis, and does it indicate recovering viral infection?",
    },
    "Monocytes": {
        "aliases": ["monocyte", "monocytes %"],
        "category": "Hematology / Differential WBC",
        "role": "Large phagocytic white blood cells that clear cellular debris and present antigens to lymphocytes.",
        "normal_range": "2% - 10% (or 200 - 800 /mcL)",
        "high_causes": "Chronic bacterial infections (tuberculosis, endocarditis), inflammatory bowel disease, recovery phase of acute infection.",
        "low_causes": "Aplastic anemia, bone marrow disease, severe stress.",
        "diet_lifestyle": "Anti-inflammatory Mediterranean diet.",
        "doctor_questions": "Is this related to any persistent or chronic low-grade infection?",
    },
    "Eosinophils": {
        "aliases": ["eosinophil", "eosinophils %", "aec", "absolute eosinophil count"],
        "category": "Hematology / Differential WBC",
        "role": "White blood cells combating parasitic infections and modulating allergic and asthmatic responses.",
        "normal_range": "1% - 6% (or 50 - 500 /mcL)",
        "high_causes": "Allergic conditions (asthma, eczema, hay fever, drug allergies), parasitic helminth infections, autoimmune disorders, eosinophilic esophagitis.",
        "low_causes": "Acute stress, elevated cortisol/Cushing's syndrome, acute bacterial infections.",
        "diet_lifestyle": "Identify and eliminate known food allergens (e.g. dairy, gluten, nuts) if atopic.",
        "doctor_questions": "Should we test for IgE levels or perform stool ova & parasite examination?",
    },
    "Basophils": {
        "aliases": ["basophil", "basophils %"],
        "category": "Hematology / Differential WBC",
        "role": "Release histamine and heparin during severe hypersensitivity reactions and inflammation.",
        "normal_range": "0% - 2% (or 0 - 100 /mcL)",
        "high_causes": "Myeloproliferative disorders (CML, polycythemia vera), severe allergic reactions, chronic inflammation.",
        "low_causes": "Acute allergic reactions, hyperthyroidism, prolonged corticosteroid use.",
        "diet_lifestyle": "Maintain antihistamine allergy management if indicated.",
        "doctor_questions": "Are there any secondary inflammatory signs accompanying this?",
    },
    "Absolute Eosinophil Count": {
        "aliases": ["aec", "absolute eosinophil count", "absolute eosinophils", "absolute eosinophil count aec"],
        "category": "Hematology / Differential WBC",
        "role": "The exact absolute number of eosinophils circulating per microliter of blood, specialized white blood cells that defend against parasites and mediate allergic/asthmatic responses.",
        "normal_range": "20 – 500 cells/mcL",
        "high_causes": "Allergic conditions (bronchial asthma, allergic rhinitis/hay fever, atopic dermatitis/eczema), drug hypersensitivity reactions, parasitic worm infections, eosinophilic esophagitis, hypereosinophilic syndrome.",
        "low_causes": "Acute severe stress, systemic corticosteroid administration, Cushing's syndrome.",
        "diet_lifestyle": "Identify and avoid known environmental allergens or trigger foods. Consume anti-inflammatory omega-3 fatty acids and quercetin-rich foods (apples, onions).",
        "doctor_questions": "Could my elevated AEC be driven by seasonal allergies, medication reaction, or gastrointestinal parasites?",
    },
    "Absolute Neutrophil Count": {
        "aliases": ["anc", "absolute neutrophil count", "absolute neutrophils"],
        "category": "Hematology / Differential WBC",
        "role": "The absolute number of circulating neutrophils, the immune system's primary and most abundant cellular frontline defense against bacterial and fungal infections.",
        "normal_range": "1,500 – 8,000 cells/mcL",
        "high_causes": "Neutrophilia: Acute bacterial infections (pneumonia, appendicitis), tissue necrosis (burns, heart attack), intense physical stress/exercise, corticosteroid use, inflammatory bowel disease.",
        "low_causes": "Neutropenia: Chemotherapy/radiation, severe viral infections (influenza, hepatitis, COVID-19), autoimmune destruction, drug-induced agranulocytosis, aplastic anemia.",
        "diet_lifestyle": "• **If ANC < 1,000 (Neutropenia):** Practice strict neutropenic precautions: cook all meats and vegetables thoroughly, avoid raw foods/unpasteurized cheeses, wash hands frequently, and seek immediate emergency care if fever develops (> 38°C / 100.4°F).",
        "doctor_questions": "Is my neutrophil count reactive to an infection, or is closer hematological follow-up required?",
    },
    "Absolute Lymphocyte Count": {
        "aliases": ["alc", "absolute lymphocyte count", "absolute lymphocytes"],
        "category": "Hematology / Differential WBC",
        "role": "The absolute concentration of circulating lymphocytes (T-cells, B-cells, Natural Killer cells) responsible for adaptive immunity, antibody production, and viral defense.",
        "normal_range": "1,000 – 4,800 cells/mcL",
        "high_causes": "Lymphocytosis: Acute viral infections (mononucleosis, cytomegalovirus, hepatitis), chronic bacterial infections (tuberculosis, pertussis/whooping cough), chronic lymphocytic leukemia (CLL).",
        "low_causes": "Lymphopenia: Corticosteroid therapy, acute systemic viral illness, malnutrition, severe COVID-19, autoimmune disease (lupus), immunodeficiency states.",
        "diet_lifestyle": "Support adaptive immunity with adequate dietary zinc, selenium, vitamin D3, and quality amino acids.",
        "doctor_questions": "Does my lymphocyte count suggest a recent or resolving viral illness?",
    },
    "Absolute Monocyte Count": {
        "aliases": ["amc", "absolute monocyte count", "absolute monocytes"],
        "category": "Hematology / Differential WBC",
        "role": "The absolute concentration of monocytes, large circulating phagocytes that migrate into tissues to become macrophages and dendritic cells, clearing cellular debris and presenting antigens.",
        "normal_range": "200 – 800 cells/mcL",
        "high_causes": "Monocytosis: Subacute or chronic infections (tuberculosis, subacute bacterial endocarditis), recovery phase of acute infections, inflammatory bowel disease, rheumatoid arthritis, myelomonocytic leukemia.",
        "low_causes": "Bone marrow failure, acute overwhelming endotoxemia, hairy cell leukemia.",
        "diet_lifestyle": "Focus on an anti-inflammatory diet and adequate rest during convalescence.",
        "doctor_questions": "Is this monocyte elevation indicative of resolving infection or ongoing chronic inflammation?",
    },
    "Absolute Basophil Count": {
        "aliases": ["abc", "absolute basophil count", "absolute basophils"],
        "category": "Hematology / Differential WBC",
        "role": "The absolute concentration of basophils, rare granulocytes containing histamine and heparin that participate in severe allergic reactions and hypersensitivity responses.",
        "normal_range": "0 – 200 cells/mcL (or 0 – 0.2 × 10^3/mcL)",
        "high_causes": "Basophilia: Myeloproliferative disorders (chronic myeloid leukemia - CML, polycythemia vera), allergic inflammation, hypothyroidism, ulcerative colitis.",
        "low_causes": "Acute allergic reactions (degranulation), acute infection, hyperthyroidism, steroid use.",
        "diet_lifestyle": "Manage known histamine triggers and seek clinical review if persistently elevated.",
        "doctor_questions": "Does persistent basophilia warrant evaluation for myeloproliferative conditions?",
    },
    "ESR": {
        "aliases": ["erythrocyte sedimentation rate", "sed rate"],
        "category": "Inflammatory Marker",
        "role": "Measures the rate at which red blood cells settle at the bottom of a test tube, indicating systemic inflammation.",
        "normal_range": "Men: 0 - 15 mm/hr | Women: 0 - 20 mm/hr (higher in older adults)",
        "high_causes": "Active bacterial/viral infections, autoimmune diseases (rheumatoid arthritis, lupus), temporal arteritis, malignancies, anemia, pregnancy.",
        "low_causes": "Polycythemia, sickle cell disease, severe heart failure, hypofibrinogenemia.",
        "diet_lifestyle": "Anti-inflammatory foods: berries, fatty fish (salmon), olive oil, walnuts, green leafy vegetables.",
        "doctor_questions": "Should we check C-Reactive Protein (CRP) for a more specific acute-phase measurement?",
    },

    # --- Biochemistry & Diabetes ---
    "Glucose - Fasting": {
        "aliases": ["fasting blood sugar", "fbs", "fasting glucose", "glucose fasting", "blood sugar fasting"],
        "category": "Biochemistry / Diabetes",
        "role": "Measures blood glucose concentration after a strict overnight fast of at least 8 to 12 hours.",
        "normal_range": "Normal: 70 - 99 mg/dL | Prediabetes: 100 - 125 mg/dL | Diabetes: ≥ 126 mg/dL",
        "high_causes": "Type 1 or Type 2 Diabetes, impaired fasting glucose (prediabetes), acute stress/illness, Cushing's syndrome, pancreatitis, steroid therapy.",
        "low_causes": "Hypoglycemia (too much insulin, skipped meals, prolonged intense exercise, insulinoma, Addison's disease, severe liver disease).",
        "diet_lifestyle": "• **Diet:** Low glycemic index foods (oats, legumes, non-starchy vegetables, whole grains). Limit refined sugars, sweetened drinks, and white flour.\n• **Lifestyle:** 30 minutes of daily aerobic exercise (brisk walking, cycling) and strength training to enhance insulin sensitivity.",
        "doctor_questions": "Do I need an HbA1c test or an Oral Glucose Tolerance Test (OGTT) to confirm diabetes?",
    },
    "HbA1c": {
        "aliases": ["glycated hemoglobin", "glycosylated hemoglobin", "a1c"],
        "category": "Biochemistry / Diabetes",
        "role": "Measures the percentage of hemoglobin bounded to glucose, reflecting the average blood sugar level over the past 2 to 3 months.",
        "normal_range": "Normal: < 5.7% | Prediabetes: 5.7% - 6.4% | Diabetes: ≥ 6.5%",
        "high_causes": "Uncontrolled or newly diagnosed diabetes, chronic hyperglycemia, metabolic syndrome.",
        "low_causes": "Hemolytic anemia, recent blood transfusion, pregnancy, chronic blood loss.",
        "diet_lifestyle": "Consistent portion control, Mediterranean diet pattern, regular exercise, weight management, and continuous/periodic glucose monitoring.",
        "doctor_questions": "What is my personalized target HbA1c based on my age and medical history?",
    },
    "Blood Glucose Random": {
        "aliases": ["rbs", "blood glucose random", "random blood sugar", "random glucose", "blood glucose random-rbs", "random blood glucose", "glucose random"],
        "category": "Metabolic & Diabetes Screening",
        "role": "Measures the circulating concentration of glucose in the bloodstream at an unscheduled time regardless of when the last meal was consumed.",
        "normal_range": "70 – 140 mg/dL (< 7.8 mmol/L)",
        "high_causes": "Prediabetes (140 – 199 mg/dL), Diabetes mellitus (≥ 200 mg/dL with symptoms), acute stress/illness, corticosteroid therapy, high-carbohydrate recent meal.",
        "low_causes": "Hypoglycemia (< 70 mg/dL): Excess insulin or diabetic medication, prolonged fasting, intense exercise without food, reactive hypoglycemia, insulinoma.",
        "diet_lifestyle": "• **If High:** Adopt a low-glycemic Mediterranean or whole-food diet, eliminate sugary sodas and refined carbohydrates, increase dietary fiber (chia seeds, oats, beans), and walk 15–20 minutes after meals.\n• **If Low (< 70 mg/dL):** Consume 15 grams of fast-acting carbohydrate (half cup juice, 3-4 glucose tablets) and recheck in 15 minutes (Rule of 15).",
        "doctor_questions": "Should we schedule a formal Fasting Blood Sugar (FBS) and HbA1c test to evaluate average 3-month glycemic control?",
    },
    "Post Prandial Blood Sugar": {
        "aliases": ["ppbs", "post prandial glucose", "2 hour postprandial glucose", "pp glucose"],
        "category": "Metabolic & Diabetes Screening",
        "role": "Measures the circulating concentration of blood sugar exactly 2 hours after the start of a meal, assessing your body's post-meal insulin response.",
        "normal_range": "< 140 mg/dL (in non-diabetics)",
        "high_causes": "Impaired glucose tolerance / prediabetes (140 – 199 mg/dL), Type 2 or Type 1 diabetes (≥ 200 mg/dL), insulin resistance.",
        "low_causes": "Late reactive hypoglycemia, excessive diabetic medication, gastric bypass surgery.",
        "diet_lifestyle": "Reduce simple refined starches; pair carbohydrates with healthy fats and proteins to slow glycemic absorption.",
        "doctor_questions": "Does my post-prandial spike warrant dietary carbohydrate modification or metformin therapy?",
    },

    # --- Renal & Kidney Panel ---
    "Creatinine": {
        "aliases": ["serum creatinine", "creat"],
        "category": "Renal / Kidney Function",
        "role": "A normal waste breakdown product of muscle metabolism filtered out of the blood exclusively by the kidneys.",
        "normal_range": "Men: 0.7 - 1.3 mg/dL | Women: 0.6 - 1.1 mg/dL",
        "high_causes": "Acute kidney injury (AKI), chronic kidney disease (CKD), severe dehydration, urinary tract obstruction (kidney stones, enlarged prostate), nephrotoxic drugs (NSAIDs, certain antibiotics), excessive creatine supplementation.",
        "low_causes": "Severe muscle atrophy, malnutrition, pregnancy, advanced liver disease.",
        "diet_lifestyle": "• **Care:** Maintain consistent daily hydration (unless fluid-restricted by a nephrologist).\n• **Avoid:** Overuse of over-the-counter painkillers (ibuprofen, naproxen, diclofenac) which stress renal arterioles.\n• **Diet:** Moderate protein intake; avoid high-sodium processed foods.",
        "doctor_questions": "What is my estimated Glomerular Filtration Rate (eGFR), and should we test urine for microalbumin?",
    },
    "Blood Urea": {
        "aliases": ["urea", "blood urea nitrogen", "bun"],
        "category": "Renal / Kidney Function",
        "role": "Waste product formed in the liver from dietary protein breakdown and excreted by the kidneys.",
        "normal_range": "15 - 40 mg/dL (BUN: 7 - 20 mg/dL)",
        "high_causes": "Kidney failure/dysfunction, dehydration, high-protein diet, upper gastrointestinal bleeding, congestive heart failure.",
        "low_causes": "Severe liver failure, low-protein malnutrition, overhydration, pregnancy.",
        "diet_lifestyle": "Stay hydrated with clean water throughout the day. Balance dietary protein intake.",
        "doctor_questions": "Does the ratio of BUN to Creatinine point towards dehydration or intrinsic kidney injury?",
    },
    "Uric Acid": {
        "aliases": ["serum uric acid"],
        "category": "Renal & Metabolic",
        "role": "Waste product produced when your body breaks down chemicals called purines found in certain foods and cells.",
        "normal_range": "Men: 3.5 - 7.2 mg/dL | Women: 2.4 - 6.0 mg/dL",
        "high_causes": "Hyperuricemia, gout (crystalline arthritis), kidney stones, high-purine diet, excessive alcohol/beer, kidney dysfunction, metabolic syndrome.",
        "low_causes": "Wilson's disease, Fanconi syndrome, low purine intake, certain medications.",
        "diet_lifestyle": "• **Avoid:** Red meat, organ meats, shellfish, beer, high-fructose corn syrup.\n• **Add:** Low-fat dairy, tart cherries/sour cherry juice, lemons, abundant water (2.5 - 3 L/day).",
        "doctor_questions": "Do I need prophylactic uric acid lowering therapy (allopurinol, febuxostat) if I have joint pain?",
    },

    # --- Liver Function Panel ---
    "Total Bilirubin": {
        "aliases": ["bilirubin", "bilirubin total", "serum bilirubin"],
        "category": "Liver Function Panel",
        "role": "Yellow-orange pigment formed during the normal physiological breakdown of red blood cells, processed by the liver.",
        "normal_range": "0.2 - 1.2 mg/dL",
        "high_causes": "Jaundice: Liver inflammation/hepatitis, bile duct obstruction (gallstones), Gilbert's syndrome, hemolysis (rapid RBC destruction).",
        "low_causes": "Generally not clinically concerning.",
        "diet_lifestyle": "Avoid all alcohol, greasy fried foods, and untested herbal supplements. Drink plenty of water and eat antioxidant-rich foods.",
        "doctor_questions": "Is the elevation conjugated (direct) or unconjugated (indirect), and is an abdominal ultrasound indicated?",
    },
    "SGPT": {
        "aliases": ["alt", "alanine aminotransferase", "alanine transaminase", "sgpt/alt"],
        "category": "Liver Function Panel",
        "role": "An enzyme found predominantly inside liver cells. Highly sensitive and specific marker for hepatocellular injury.",
        "normal_range": "Men: 10 - 40 U/L | Women: 7 - 35 U/L",
        "high_causes": "Non-alcoholic fatty liver disease (NAFLD), viral hepatitis (A, B, C), alcohol-related liver injury, drug toxicity (paracetamol/acetaminophen), obesity.",
        "low_causes": "Normal; severe vitamin B6 deficiency in rare cases.",
        "diet_lifestyle": "• **Diet:** Whole food Mediterranean diet, black coffee (shown to protect hepatocytes), cruciferous vegetables (broccoli, cabbage).\n• **Avoid:** Alcohol, sugary sodas, high saturated fats.\n• **Weight:** Gradual weight loss (5-10%) dramatically reduces liver fat and ALT levels.",
        "doctor_questions": "Do we need an ultrasound of the abdomen or viral hepatitis serology markers?",
    },
    "SGOT": {
        "aliases": ["ast", "aspartate aminotransferase", "aspartate transaminase", "sgot/ast"],
        "category": "Liver & Cardiac Function",
        "role": "An enzyme found in the liver, heart muscle, skeletal muscle, and kidneys.",
        "normal_range": "10 - 35 U/L",
        "high_causes": "Liver damage, alcoholic hepatitis (AST/ALT ratio > 2), intense muscle injury or strenuous workout, myocardial infarction, heart failure.",
        "low_causes": "Generally considered normal.",
        "diet_lifestyle": "Limit alcohol strictly; avoid strenuous weightlifting within 48 hours before testing if checking liver enzymes.",
        "doctor_questions": "Does the AST/ALT ratio suggest alcohol-related injury or metabolic fatty liver?",
    },
    "Alkaline Phosphatase": {
        "aliases": ["alp", "alk phos"],
        "category": "Liver & Bone Marker",
        "role": "An enzyme concentrated in the bile ducts of the liver and in bone-forming osteoblasts.",
        "normal_range": "44 - 147 U/L (higher in growing children and pregnant women)",
        "high_causes": "Biliary obstruction (gallstones, biliary stricture), hepatitis, bone conditions (Paget's disease, healing fractures, rickets/osteomalacia), hyperparathyroidism.",
        "low_causes": "Malnutrition, zinc deficiency, magnesium deficiency, hypothyroidism.",
        "diet_lifestyle": "Adequate calcium and Vitamin D for bone health; low-fat diet if biliary congestion.",
        "doctor_questions": "Should we test GGT (Gamma-GT) to clarify if the ALP origin is liver or bone?",
    },
    "Albumin": {
        "aliases": ["serum albumin"],
        "category": "Liver & Nutritional Status",
        "role": "The primary protein produced by the liver that keeps fluid from leaking out of blood vessels (oncotic pressure) and transports hormones.",
        "normal_range": "3.5 - 5.0 g/dL",
        "high_causes": "Dehydration (relative concentration).",
        "low_causes": "Advanced liver disease (cirrhosis), nephrotic syndrome (protein loss in urine), severe malnutrition, protein-losing enteropathy, severe burns.",
        "diet_lifestyle": "Quality protein foods: eggs, fish, tofu, legumes, dairy. Maintain liver health.",
        "doctor_questions": "Is the low albumin accompanied by peripheral edema (swelling in ankles) or proteinuria?",
    },
    "Total Protein": {
        "aliases": ["total protein", "serum protein", "protein total"],
        "category": "Liver & Nutritional Panel",
        "role": "Measures the combined total quantity of albumin and globulins in the liquid portion of the blood.",
        "normal_range": "6.0 – 8.3 g/dL",
        "high_causes": "Dehydration (hemoconcentration), chronic inflammatory disorders, multiple myeloma or monoclonal gammopathies, chronic viral hepatitis.",
        "low_causes": "Malnutrition/protein deficiency, liver cirrhosis (impaired synthesis), nephrotic syndrome (renal protein loss), malabsorption (celiac/Crohn's).",
        "diet_lifestyle": "Ensure adequate, high-biological-value protein in your daily diet (lentils, eggs, fish, tofu, beans).",
        "doctor_questions": "Is the protein change primarily due to albumin or globulin fraction?",
    },
    "Globulin": {
        "aliases": ["globulin", "serum globulin"],
        "category": "Liver & Immune Status",
        "role": "A group of proteins including immunoglobulins (antibodies) produced by the immune system and transport proteins produced by the liver.",
        "normal_range": "2.0 – 3.5 g/dL",
        "high_causes": "Chronic autoimmune diseases (lupus, rheumatoid arthritis), chronic infections (HIV, hepatitis), multiple myeloma, lymphoma.",
        "low_causes": "Genetic immunodeficiency, severe malnutrition, kidney protein leakage.",
        "diet_lifestyle": "Anti-inflammatory whole food nutrition; manage underlying immune health.",
        "doctor_questions": "Would a serum protein electrophoresis (SPEP) be helpful to evaluate this globulin level?",
    },
    "A/G Ratio": {
        "aliases": ["a/g ratio", "albumin globulin ratio", "ag ratio"],
        "category": "Liver & Kidney Panel",
        "role": "The mathematical ratio of albumin to globulin in serum, serving as a sensitive indicator for liver, kidney, or plasma cell disorders.",
        "normal_range": "1.2 – 2.2",
        "high_causes": "Underproduction of immunoglobulins (leukemia, genetic hypogammaglobulinemia).",
        "low_causes": "Overproduction of globulins (multiple myeloma, autoimmune disease, chronic infection) or underproduction/loss of albumin (cirrhosis, nephrotic syndrome).",
        "diet_lifestyle": "Support liver and kidney health with balanced hydration and adequate dietary protein.",
        "doctor_questions": "Does a low A/G ratio suggest liver dysfunction, protein loss, or an active immune response?",
    },
    "Direct Bilirubin": {
        "aliases": ["direct bilirubin", "conjugated bilirubin", "bilirubin direct"],
        "category": "Liver Function Panel",
        "role": "Water-soluble bilirubin that has been processed and conjugated by the liver cells with glucuronic acid for excretion into bile.",
        "normal_range": "0.0 – 0.3 mg/dL",
        "high_causes": "Biliary obstruction (gallstones, biliary stricture, pancreatic tumor), acute hepatitis, drug-induced cholestasis, Dubin-Johnson syndrome.",
        "low_causes": "Physiologically normal.",
        "diet_lifestyle": "Strictly avoid alcohol and heavy fried/greasy foods. Drink plenty of water and seek prompt medical evaluation if skin/eyes are yellow.",
        "doctor_questions": "Is an abdominal ultrasound or MRCP recommended to check for bile duct obstruction?",
    },

    # --- Lipid Profile ---
    "Total Cholesterol": {
        "aliases": ["cholesterol", "serum cholesterol"],
        "category": "Lipid Profile / Cardiovascular",
        "role": "A sterol lipid essential for cell membrane structure and steroid hormone synthesis; high circulating levels cause vascular plaque.",
        "normal_range": "Desirable: < 200 mg/dL | Borderline High: 200 - 239 mg/dL | High: ≥ 240 mg/dL",
        "high_causes": "Familial hypercholesterolemia, diet rich in saturated/trans fats, hypothyroidism, nephrotic syndrome, sedentary lifestyle.",
        "low_causes": "Severe malnutrition, hyperthyroidism, malabsorption, advanced liver disease.",
        "diet_lifestyle": "• **Diet:** Soluble fiber (oats, barley, psyllium husk, apples), walnuts, extra virgin olive oil. Eliminate trans fats and commercial fried snacks.\n• **Exercise:** 150+ minutes of weekly moderate cardiovascular exercise.",
        "doctor_questions": "What is my 10-year Atherosclerotic Cardiovascular Disease (ASCVD) risk score?",
    },
    "HDL Cholesterol": {
        "aliases": ["hdl", "good cholesterol", "high density lipoprotein"],
        "category": "Lipid Profile / Cardiovascular",
        "role": "'Good' cholesterol that scours blood vessels, picking up excess cholesterol and returning it to the liver for excretion (reverse transport).",
        "normal_range": "Optimal: ≥ 60 mg/dL | Men: > 40 mg/dL | Women: > 50 mg/dL (higher is better)",
        "high_causes": "Regular aerobic exercise, genetic factors, estrogen, moderate olive oil intake.",
        "low_causes": "Sedentary lifestyle, cigarette smoking, obesity, metabolic syndrome, Type 2 diabetes.",
        "diet_lifestyle": "Brisk walking, quit smoking, healthy fats (avocados, nuts, fatty fish).",
        "doctor_questions": "How can I raise my HDL level safely?",
    },
    "LDL Cholesterol": {
        "aliases": ["ldl", "bad cholesterol", "low density lipoprotein"],
        "category": "Lipid Profile / Cardiovascular",
        "role": "'Bad' cholesterol that carries cholesterol from the liver to tissues; excess oxidizes and deposits as arterial plaque (atherosclerosis).",
        "normal_range": "Optimal: < 100 mg/dL | Near Optimal: 100 - 129 mg/dL | High: ≥ 160 mg/dL (For heart patients: < 70 or < 55 mg/dL)",
        "high_causes": "High saturated/trans fat intake, genetic hypercholesterolemia, diabetes, hypothyroidism, smoking.",
        "low_causes": "Hypobetalipoproteinemia, hyperthyroidism, severe malnutrition.",
        "diet_lifestyle": "Reduce red meat, full-fat butter/cream, palm oil. Increase plant sterols, legumes, chia seeds, and oats.",
        "doctor_questions": "Would a statin medication be recommended based on my overall cardiovascular risk profile?",
    },
    "Triglycerides": {
        "aliases": ["triglyceride", "tg", "serum triglycerides"],
        "category": "Lipid Profile / Cardiovascular",
        "role": "The most common form of stored fat in your body, derived directly from unused dietary calories and simple sugars.",
        "normal_range": "Normal: < 150 mg/dL | Borderline High: 150 - 199 mg/dL | High: 200 - 499 mg/dL | Very High: ≥ 500 mg/dL",
        "high_causes": "High sugar and refined carbohydrate intake, alcohol consumption, obesity, uncontrolled diabetes, hypothyroidism, pancreatitis risk when > 500.",
        "low_causes": "Malnutrition, very low-fat diet, hyperthyroidism.",
        "diet_lifestyle": "• **Key:** Drastically cut sugar, sweetened beverages, white rice/bread, and alcohol.\n• **Add:** Omega-3 fatty acids (flaxseed, salmon, walnuts).",
        "doctor_questions": "Is my fasting glucose or insulin elevated along with triglycerides?",
    },

    # --- Thyroid Panel ---
    "TSH": {
        "aliases": ["thyroid stimulating hormone", "thyrotropin"],
        "category": "Thyroid Profile",
        "role": "Hormone produced by the pituitary gland that instructs the thyroid gland how much T3 and T4 hormones to manufacture.",
        "normal_range": "0.4 - 4.5 mIU/L (varies slightly by lab and pregnancy trimester)",
        "high_causes": "Primary Hypothyroidism (underactive thyroid), Hashimoto's thyroiditis, recovery phase from severe illness, inadequate thyroid medication dosage.",
        "low_causes": "Hyperthyroidism (overactive thyroid, Graves' disease), toxic nodular goiter, excessive thyroid medication, secondary central hypothyroidism.",
        "diet_lifestyle": "• **If High (Hypo):** Adequate dietary iodine and selenium (brazil nuts, eggs). Take thyroid medications (Levothyroxine) on an empty stomach with water 30-60 min before breakfast.\n• **Note:** Avoid Biotin supplements 48h before thyroid blood tests as it interferes with assays.",
        "doctor_questions": "Should we test Free T3, Free T4, and anti-TPO antibodies to determine the etiology?",
    },
    "Free T4": {
        "aliases": ["ft4", "free thyroxine"],
        "category": "Thyroid Profile",
        "role": "The biologically active, unbound form of thyroxine hormone that regulates your body's metabolic rate.",
        "normal_range": "0.8 - 1.8 ng/dL",
        "high_causes": "Hyperthyroidism, Graves' disease, subacute thyroiditis.",
        "low_causes": "Hypothyroidism, severe iodine deficiency, pituitary insufficiency.",
        "diet_lifestyle": "Consult physician for medication titration.",
        "doctor_questions": "Does my FT4 correlate with my TSH level?",
    },

    # --- Vitamins & Minerals ---
    "25-OH Vitamin D": {
        "aliases": ["vitamin d", "vitamin d3", "25 hydroxy vitamin d", "vit d"],
        "category": "Vitamins & Bone Health",
        "role": "Steroid pre-hormone critical for calcium absorption, bone mineralization, neuromuscular function, and immune defense.",
        "normal_range": "Deficiency: < 20 ng/mL | Insufficiency: 20 - 29 ng/mL | Sufficiency: 30 - 100 ng/mL",
        "high_causes": "Excessive supplementation toxicity (hypervitaminosis D), hypercalcemia.",
        "low_causes": "Lack of direct sun exposure, dark skin pigmentation, malabsorption (celiac, Crohn's), obesity, inadequate dietary intake.",
        "diet_lifestyle": "• **Sunlight:** 15-20 minutes of midday sunlight on arms and legs without sunscreen 3-4 times a week.\n• **Diet:** Fortified milk, fatty fish, egg yolks, mushrooms exposed to UV.\n• **Supplementation:** Discuss oral Vitamin D3 (cholecalciferol) with your doctor.",
        "doctor_questions": "Do I need a weekly therapeutic booster dose (e.g. 60,000 IU) followed by maintenance?",
    },
    "Vitamin B12": {
        "aliases": ["b12", "cobalamin", "cyanocobalamin"],
        "category": "Vitamins & Neurological Health",
        "role": "Essential cofactor for myelin sheath synthesis in nerve cells and DNA synthesis in developing red blood cells.",
        "normal_range": "200 - 900 pg/mL (Borderline: 200 - 300 pg/mL)",
        "high_causes": "Excessive vitamin supplementation, liver disease, myeloproliferative disorders.",
        "low_causes": "Strict vegetarian/vegan diet without supplements, pernicious anemia (lack of intrinsic factor), metformin long-term use, PPI antacid overuse, celiac disease.",
        "diet_lifestyle": "• **Diet:** Dairy products, eggs, fish, meat, fortified plant milks/cereals.\n• **Symptoms:** Fatigue, tingling/numbness in fingers/toes, memory fog, sore tongue.",
        "doctor_questions": "Would sublingual or intramuscular B12 injections be better than oral tablets for me?",
    },
    "Ferritin": {
        "aliases": ["serum ferritin", "iron stores"],
        "category": "Iron Studies & Acute Phase",
        "role": "The primary intracellular protein that stores iron and releases it in a controlled fashion.",
        "normal_range": "Men: 20 - 250 ng/mL | Women: 15 - 150 ng/mL",
        "high_causes": "Hemochromatosis (iron overload), systemic inflammation/infection, liver injury, adult-onset Still's disease, malignancies.",
        "low_causes": "Iron deficiency anemia, depleted body iron stores (even before hemoglobin drops).",
        "diet_lifestyle": "• **If Low:** Iron-rich nutrition, vitamin C, oral iron supplements.\n• **If High:** Avoid iron supplements and iron-fortified cereals; explore liver health.",
        "doctor_questions": "Is low ferritin the root cause of my fatigue and hair thinning?",
    },
    "Serum Calcium": {
        "aliases": ["calcium", "total calcium"],
        "category": "Minerals & Bone Metabolism",
        "role": "Essential mineral for bone structural integrity, cardiac muscle contraction, nerve impulse transmission, and blood coagulation.",
        "normal_range": "8.5 - 10.5 mg/dL",
        "high_causes": "Primary hyperparathyroidism, malignancy, excess vitamin D, thiazide diuretics.",
        "low_causes": "Hypoparathyroidism, vitamin D deficiency, chronic kidney disease, hypomagnesemia, low albumin (pseudohypocalcemia).",
        "diet_lifestyle": "Dairy, fortified plant milk, tofu, leafy greens, sesame seeds.",
        "doctor_questions": "Was this calcium level corrected for serum albumin?",
    },
    "C-Reactive Protein": {
        "aliases": ["crp", "c reactive protein", "hs-crp"],
        "category": "Inflammatory Marker",
        "role": "An acute-phase protein produced by the liver whose blood levels spike dramatically in response to systemic inflammation or bacterial infection.",
        "normal_range": "Normal: < 5 mg/L (hs-CRP for cardiac risk: < 1 mg/L Low, 1-3 Average, > 3 High)",
        "high_causes": "Bacterial infections, flare-ups of autoimmune disease (rheumatoid arthritis, IBD), cardiovascular inflammation, post-operative trauma.",
        "low_causes": "Normal physiological state.",
        "diet_lifestyle": "Adopt anti-inflammatory lifestyle: Mediterranean diet, regular exercise, stress reduction, restful sleep.",
        "doctor_questions": "Does this indicate an acute infection or an ongoing chronic inflammatory condition?",
    },
    "Serum Iron": {
        "aliases": ["serum iron", "iron", "fe"],
        "category": "Iron Studies & Hematology",
        "role": "Measures the circulating ferric iron bound to transferrin in blood serum, indicating immediate iron availability.",
        "normal_range": "60 – 170 mcg/dL",
        "high_causes": "Hemochromatosis (iron overload), repeated blood transfusions, acute iron poisoning, hemolytic anemia.",
        "low_causes": "Iron deficiency anemia, chronic occult blood loss, poor dietary intake, pregnancy, malabsorption.",
        "diet_lifestyle": "Pair dietary iron with vitamin C; avoid tea/coffee near meals.",
        "doctor_questions": "What is my Transferrin Saturation percentage?",
    },
    "Total Iron Binding Capacity": {
        "aliases": ["tibc", "total iron binding capacity", "iron binding capacity"],
        "category": "Iron Studies & Hematology",
        "role": "Measures the maximum amount of iron that transferrin proteins in the blood can bind, reflecting the body's iron transport capacity.",
        "normal_range": "240 – 450 mcg/dL",
        "high_causes": "Iron deficiency anemia (the liver synthesizes more transferrin to capture available iron), pregnancy, oral contraceptives.",
        "low_causes": "Iron overload/hemochromatosis, chronic systemic disease, liver cirrhosis, malnutrition.",
        "diet_lifestyle": "Focus on bioavailable iron and optimal protein nutrition.",
        "doctor_questions": "Does elevated TIBC confirm iron deficiency anemia?",
    },
    "Sodium": {
        "aliases": ["serum sodium", "sodium", "na+"],
        "category": "Serum Electrolytes & Renal",
        "role": "The primary extracellular electrolyte regulating fluid volume, osmotic balance, nerve transmission, and blood pressure.",
        "normal_range": "135 – 145 mEq/L (or mmol/L)",
        "high_causes": "Hypernatremia: Dehydration, excessive fluid loss (sweating, diarrhea, vomiting), diabetes insipidus, high sodium intake.",
        "low_causes": "Hyponatremia: Diuretic medications, heart failure, liver cirrhosis, SIADH, excessive free water intake, adrenal insufficiency.",
        "diet_lifestyle": "Drink fluids in moderation according to thirst; consult doctor before changing dietary salt intake if taking cardiac medications.",
        "doctor_questions": "Does my sodium level reflect my hydration status or medication effects?",
    },
    "Potassium": {
        "aliases": ["serum potassium", "potassium", "k+"],
        "category": "Serum Electrolytes & Cardiac",
        "role": "The primary intracellular cation essential for cardiac electrical conduction, neuromuscular function, and cellular membrane potential.",
        "normal_range": "3.5 – 5.0 mEq/L (or mmol/L)",
        "high_causes": "Hyperkalemia: Chronic kidney disease, ACE inhibitors / ARBs, potassium-sparing diuretics, severe tissue trauma/hemolysis, Addison's disease.",
        "low_causes": "Hypokalemia: Loop or thiazide diuretics, prolonged vomiting/diarrhea, laxative abuse, low dietary intake, hyperaldosteronism.",
        "diet_lifestyle": "• **If High:** Restrict high-potassium foods (bananas, potatoes, tomatoes, oranges) and salt substitutes (potassium chloride).\n• **If Low:** Eat potassium-rich foods (bananas, coconut water, sweet potatoes, spinach).",
        "doctor_questions": "Is an ECG indicated to check cardiac conduction if potassium is out of range?",
    },
    "Chloride": {
        "aliases": ["serum chloride", "chloride", "cl-"],
        "category": "Serum Electrolytes & Acid-Base",
        "role": "Major extracellular anion maintaining proper hydration, osmotic pressure, and electrical neutrality alongside sodium.",
        "normal_range": "96 – 106 mEq/L (or mmol/L)",
        "high_causes": "Hyperchloremia: Dehydration, renal tubular acidosis, metabolic acidosis, excessive saline infusion.",
        "low_causes": "Hypochloremia: Prolonged vomiting (loss of stomach HCl), metabolic alkalosis, chronic respiratory acidosis, loop diuretics.",
        "diet_lifestyle": "Stay properly hydrated with balanced electrolyte fluids.",
        "doctor_questions": "Does my chloride level correlate with my bicarbonate and anion gap?",
    },
    "Procalcitonin": {
        "aliases": ["procalcitonin", "pct procalcitonin", "serum procalcitonin"],
        "category": "Clinical Biochemistry / Sepsis & Infection Biomarker",
        "role": "A peptide precursor of calcitonin. In healthy individuals, levels are negligible. During systemic bacterial infections, tissue cells release large amounts, making it a critical biomarker to distinguish bacterial infection/sepsis from viral illness.",
        "normal_range": "< 0.05 – 0.10 ng/mL",
        "high_causes": "Systemic bacterial infection, severe sepsis, septic shock, bacterial pneumonia, bacterial meningitis, severe trauma/burns, major surgery.",
        "low_causes": "Normal; viral infections, localized minor infections, or non-infectious inflammatory states.",
        "diet_lifestyle": "Procalcitonin is an acute-care biomarker requiring immediate physician assessment.",
        "doctor_questions": "Does this procalcitonin level indicate systemic bacterial infection requiring antibiotic therapy?",
    },
}


# ---------------------------------------------------------------------------
# 2. CONDITIONS & DISEASES KNOWLEDGE
# ---------------------------------------------------------------------------

CONDITIONS_DB: Dict[str, Dict[str, Any]] = {
    "anemia": {
        "title": "Anemia (Reduced Oxygen Carrying Capacity)",
        "summary": "Anemia occurs when your blood has a lower than normal concentration of hemoglobin or red blood cells, impairing oxygen delivery to vital tissues.",
        "common_tests": ["Hemoglobin", "RBC Count", "Packed Cell Volume", "MCV", "MCH", "Ferritin", "Vitamin B12"],
        "subtypes": [
            "**Microcytic Hypochromic Anemia (MCV < 80):** Most commonly caused by Iron Deficiency Anemia or Thalassemia trait.",
            "**Macrocytic Anemia (MCV > 100):** Often caused by Vitamin B12 or Folate deficiency, alcoholism, or thyroid dysfunction.",
            "**Normocytic Normochromic Anemia (MCV 80-100):** Frequently seen in chronic kidney disease, acute blood loss, or anemia of chronic inflammation.",
        ],
        "symptoms": "Generalized fatigue, pale skin/nail beds, dizziness, shortness of breath on exertion, cold hands and feet, brittle nails, heart palpitations.",
        "action_plan": "Consult your doctor for iron studies (Ferritin, TIBC) or B12 testing. Never take high-dose iron pills without clinical confirmation, as iron overload can harm the liver.",
    },
    "diabetes": {
        "title": "Diabetes Mellitus & Impaired Glycaemic Control",
        "summary": "A metabolic disorder characterized by persistent hyperglycemia resulting from defects in insulin secretion, insulin action, or both.",
        "common_tests": ["Glucose - Fasting", "HbA1c", "Postprandial Glucose", "Urine Glucose", "Urine Microalbumin"],
        "diagnostic_thresholds": [
            "**Normal:** Fasting Glucose < 100 mg/dL | HbA1c < 5.7%",
            "**Prediabetes:** Fasting Glucose 100 - 125 mg/dL | HbA1c 5.7% - 6.4%",
            "**Diabetes:** Fasting Glucose ≥ 126 mg/dL | HbA1c ≥ 6.5%",
        ],
        "symptoms": "Frequent urination (polyuria), excessive thirst (polydipsia), unexplained weight loss, slow wound healing, blurred vision, recurrent infections.",
        "action_plan": "Implement dietary changes (low carb, high fiber, zero sugary drinks), start 150 min/week of exercise, and consult a physician or diabetologist for medical management.",
    },
    "kidney_disease": {
        "title": "Renal Impairment & Chronic Kidney Disease (CKD)",
        "summary": "Gradual loss of kidney filtering capacity, causing toxic metabolic waste products (urea, creatinine) to accumulate in the bloodstream.",
        "common_tests": ["Creatinine", "Blood Urea", "Uric Acid", "eGFR", "Electrolytes", "Urine Protein"],
        "symptoms": "Swelling in ankles, legs, or around the eyes (edema), fatigue, foamy urine, decreased urine output, nausea, muscle cramps.",
        "action_plan": "Avoid NSAID painkillers (ibuprofen, naproxen). Manage blood pressure and diabetes strictly. Follow up with a nephrologist.",
    },
    "liver_disease": {
        "title": "Hepatic Dysfunction & Fatty Liver (NAFLD)",
        "summary": "Injury to liver cells resulting from fat deposition (steatosis), viral hepatitis, alcohol, or medications, releasing transaminases into circulation.",
        "common_tests": ["SGPT (ALT)", "SGOT (AST)", "Total Bilirubin", "Alkaline Phosphatase", "Albumin", "GGT"],
        "symptoms": "Right upper quadrant abdominal discomfort, yellowing of skin/eyes (jaundice), dark urine, pale stools, easy bruising, chronic fatigue.",
        "action_plan": "Eliminate alcohol completely. Follow a low-fat, low-sugar diet. Consider an abdominal ultrasound to stage fatty liver infiltration.",
    },
    "thyroid": {
        "title": "Thyroid Gland Dysfunction (Hypo / Hyperthyroidism)",
        "summary": "Disorders where the thyroid gland produces too little hormone (Hypothyroidism - slow metabolism) or too much hormone (Hyperthyroidism - revved up metabolism).",
        "common_tests": ["TSH", "Free T3", "Free T4", "Anti-TPO"],
        "symptoms": [
            "**Hypothyroidism (High TSH, Low FT4):** Weight gain, fatigue, cold intolerance, dry skin, constipation, puffy face, depression.",
            "**Hyperthyroidism (Low TSH, High FT4):** Weight loss, rapid heartbeat/palpitations, heat intolerance, anxiety, tremors, frequent bowel movements.",
        ],
        "action_plan": "Consult an endocrinologist for personalized hormone replacement (Levothyroxine) or anti-thyroid medications.",
    },
    "gout": {
        "title": "Hyperuricemia & Gouty Arthritis",
        "summary": "Elevated uric acid levels crystallize into monosodium urate needle-like deposits inside joint cavities, triggering excruciating inflammatory flares.",
        "common_tests": ["Uric Acid", "ESR", "CRP", "Creatinine"],
        "symptoms": "Sudden severe pain, swelling, warmth, and redness in a joint (classically the big toe / podagra), usually starting at night.",
        "action_plan": "Strictly avoid high-purine foods (red meat, beer, anchovies, organ meats). Drink 3 liters of water daily. Discuss uric acid lowering drugs with your doctor.",
    },
    "dyslipidemia": {
        "title": "Dyslipidemia & High Cholesterol",
        "summary": "Unfavorable lipid concentrations (high LDL, high triglycerides, low HDL) that accelerate atherosclerotic arterial plaque formation.",
        "common_tests": ["Total Cholesterol", "LDL Cholesterol", "HDL Cholesterol", "Triglycerides"],
        "symptoms": "Often completely asymptomatic ('silent') until cardiovascular events like heart attack or stroke occur.",
        "action_plan": "Adopt heart-healthy Mediterranean diet, exercise daily, quit smoking, and discuss cardiovascular risk calculation with your physician.",
    },
    "thrombocytopenia": {
        "title": "Thrombocytopenia (Low Platelet Count)",
        "summary": "A clinical state characterized by a circulating platelet count below 150,000 /mcL, impairing primary hemostasis and increasing bleeding vulnerability.",
        "common_tests": ["Platelet Count", "Plateletcrit (PCT)", "Mean Platelet Volume (MPV)", "Peripheral Blood Smear", "Coagulation PT/INR"],
        "symptoms": "Petechiae (tiny pinpoint red/purple spots on lower legs), easy or unexplained bruising (purpura), prolonged bleeding from minor cuts, bleeding gums, frequent nosebleeds.",
        "action_plan": "Avoid antiplatelet/NSAID medications (aspirin, ibuprofen). Avoid high-impact contact sports. Seek prompt medical evaluation to distinguish immune-mediated destruction (ITP) from bone marrow underproduction.",
    },
    "thrombocytosis": {
        "title": "Thrombocytosis (Elevated Platelet Count)",
        "summary": "Circulating platelet count exceeding 450,000 /mcL. Categorized as primary (essential thrombocythemia - myeloproliferative) or secondary/reactive (inflammation, infection, iron deficiency).",
        "common_tests": ["Platelet Count", "Plateletcrit (PCT)", "MPV", "Ferritin", "ESR", "CRP", "JAK2 Mutation"],
        "symptoms": "Often asymptomatic; may present with headaches, visual disturbances, numbness/tingling in hands/feet, or paradoxically thrombosis or bleeding.",
        "action_plan": "Maintain generous daily hydration. Evaluate for iron deficiency or chronic occult infection/inflammation. Consult a hematologist for persistent unexplained counts > 600,000 /mcL.",
    },
    "leukocytosis": {
        "title": "Leukocytosis (Elevated White Blood Cell Count)",
        "summary": "Total white blood cell count exceeding 11,000 /mcL, indicating an activated immune, inflammatory, or proliferative response.",
        "common_tests": ["Total Leucocyte Count (TLC)", "Absolute Neutrophil Count (ANC)", "Absolute Lymphocyte Count (ALC)", "Peripheral Blood Smear"],
        "symptoms": "Fever, chills, localized pain, sweating, fatigue, or symptoms related to the specific primary infection.",
        "action_plan": "Review the WBC differential breakdown (neutrophilia suggests bacterial infection, lymphocytosis suggests viral infection) and consult your doctor for targeted management.",
    },
    "leukopenia": {
        "title": "Leukopenia (Low White Blood Cell Count)",
        "summary": "Total circulating white blood cell count below 4,000 /mcL, which reduces the body's defense capabilities against infectious pathogens.",
        "common_tests": ["Total Leucocyte Count", "Absolute Neutrophil Count (ANC)", "Peripheral Blood Smear", "Viral Serology"],
        "symptoms": "Increased susceptibility to recurrent infections, mouth ulcers, sore throat, fever.",
        "action_plan": "Practice vigilant food and personal hygiene. Avoid exposure to sick individuals. Seek immediate emergency evaluation if you develop a fever (> 38°C / 100.4°F).",
    },
    "anisocytosis": {
        "title": "Anisocytosis (Unequal Red Blood Cell Sizes)",
        "summary": "A high variation in the physical diameter and volume of red blood cells, reflected quantitatively by an elevated Red Cell Distribution Width (RDW > 14.5%).",
        "common_tests": ["RDW-CV", "RDW-SD", "MCV", "Hemoglobin", "Ferritin", "Vitamin B12", "Folate"],
        "symptoms": "Symptoms mirror underlying anemia: fatigue, weakness, pale complexion, shortness of breath, lightheadedness.",
        "action_plan": "Anisocytosis is one of the earliest signs of nutritional deficiency (iron, B12, or folate) before severe anemia manifests. Discuss an Iron Profile and B-vitamin screen with your physician.",
    },
    "microcytosis": {
        "title": "Microcytosis (Small Red Blood Cells)",
        "summary": "Circulating red blood cells that have an abnormally small mean volume (MCV < 80 fL), typically caused by impaired hemoglobin synthesis.",
        "common_tests": ["MCV", "MCH", "Hemoglobin", "Ferritin", "Serum Iron", "TIBC", "Hemoglobin HPLC"],
        "symptoms": "Fatigue, pale skin, exercise intolerance, brittle spoon-shaped nails (koilonychia).",
        "action_plan": "The top two causes are Iron Deficiency Anemia and Thalassemia Minor/Trait. Request a Serum Ferritin and Hemoglobin Electrophoresis to differentiate them.",
    },
    "macrocytosis": {
        "title": "Macrocytosis (Large Red Blood Cells)",
        "summary": "Red blood cells that have an abnormally large mean volume (MCV > 100 fL), frequently resulting from impaired nuclear DNA synthesis during erythropoiesis.",
        "common_tests": ["MCV", "Vitamin B12", "Folate", "TSH", "Liver Enzymes (SGPT/SGOT)", "Reticulocyte Count"],
        "symptoms": "Fatigue, tongue soreness (glossitis), neurological tingling or numbness in extremities, cognitive fog, mood changes.",
        "action_plan": "Test serum Vitamin B12 and Folate. Avoid alcohol. Check thyroid and liver function panels.",
    },
    "hypochromia": {
        "title": "Hypochromia (Pale Red Blood Cells)",
        "summary": "Red blood cells that contain insufficient hemoglobin, appearing pale with an enlarged central pallor under microscopic evaluation, reflected by low MCH and MCHC.",
        "common_tests": ["MCH", "MCHC", "Hemoglobin", "Iron Profile", "Ferritin"],
        "symptoms": "Fatigue, paleness of the inner eyelids, lightheadedness, cold sensitivity.",
        "action_plan": "Nutritional replenishment of iron and protein under physician guidance.",
    },
}


# ---------------------------------------------------------------------------
# 3. TEST PREPARATION PROTOCOLS & GUIDANCE
# ---------------------------------------------------------------------------

PREPARATION_DB: Dict[str, str] = {
    "fasting": (
        "### 🕒 Fasting Guidelines for Blood Tests\n\n"
        "| Test Panel | Fasting Duration | Rules |\n"
        "|---|---|---|\n"
        "| **Fasting Blood Glucose (FBS)** | 8 to 10 hours | Strictly overnight; plain water is allowed |\n"
        "| **Lipid Profile (Cholesterol/Triglycerides)** | 10 to 12 hours | Fasting required for accurate triglycerides |\n"
        "| **Complete Blood Count (CBC)** | No fasting required | Can be taken anytime, stay normally hydrated |\n"
        "| **Liver Function Tests (LFT)** | Preferred 8 hours | Fasting preferred, avoid heavy fatty dinner |\n"
        "| **Kidney Function Tests (KFT)** | Preferred 8 hours | Avoid high meat meal night before test |\n"
        "| **Thyroid Profile (TSH, FT4)** | No strict fasting | Take early morning before taking daily thyroid pill |\n"
        "| **HbA1c** | No fasting required | Measures 3-month average, unaffected by recent meals |\n\n"
        "💡 **Key Rule:** You **CAN and SHOULD drink plain water** during fasting unless specifically instructed otherwise by your doctor. Dehydration makes veins harder to access and artificially concentrates blood markers!"
    ),
    "medication": (
        "### 💊 Medication Instructions Before Lab Tests\n\n"
        "1. **Thyroid Medications (Levothyroxine/Thyronorm):** Take your blood sample *first* in the early morning, and take your thyroid pill *after* the sample is drawn.\n"
        "2. **Blood Pressure Medications:** Usually taken as prescribed with a small sip of water, unless your doctor says otherwise.\n"
        "3. **Diabetes Medications (Insulin / Metformin):** Do NOT take your morning diabetes pill or morning insulin *while fasting* unless instructed, as this can trigger dangerous hypoglycemia. Take it when you have your post-draw meal.\n"
        "4. **Biotin (Vitamin B7):** High-dose biotin can artificially distort thyroid, troponin, and hormone lab results. Discontinue biotin supplements 48-72 hours before testing.\n"
        "5. **Always notify the phlebotomist** of all prescription drugs, vitamins, and supplements you are taking."
    ),
}


# ---------------------------------------------------------------------------
# 4. SYMPTOM CORRELATION DATABASE
# ---------------------------------------------------------------------------

SYMPTOMS_DB: Dict[str, Dict[str, Any]] = {
    "fatigue": {
        "symptom": "Fatigue, Low Energy, or Chronic Tiredness",
        "key_biomarkers": ["Hemoglobin", "Ferritin", "TSH", "25-OH Vitamin D", "Vitamin B12", "Glucose - Fasting", "Creatinine"],
        "possible_causes": "Iron deficiency anemia, hypothyroidism, vitamin D or B12 deficiency, diabetes, chronic kidney disease, poor sleep, chronic stress.",
        "advice": "Ask your doctor for a Comprehensive Metabolic Panel, CBC, Iron Studies, Thyroid panel, and Vitamin D/B12 levels.",
    },
    "dizziness": {
        "symptom": "Dizziness, Lightheadedness, or Feeling Faint",
        "key_biomarkers": ["Hemoglobin", "Packed Cell Volume", "Glucose - Fasting", "Electrolytes", "Blood Pressure"],
        "possible_causes": "Low blood sugar (hypoglycemia), anemia (low oxygen to brain), dehydration, orthostatic hypotension, inner ear vestibular issues.",
        "advice": "Sit down immediately when dizzy. Drink water. If dizziness is accompanied by chest pain, weakness on one side of the face/arm, or slurred speech, seek emergency care immediately.",
    },
    "swelling": {
        "symptom": "Swelling in Legs, Feet, Ankles, or Puffy Eyes (Edema)",
        "key_biomarkers": ["Creatinine", "Blood Urea", "Albumin", "Urine Protein", "Total Protein", "BNP"],
        "possible_causes": "Kidney disease (protein leakage/nephrotic syndrome), heart failure (fluid retention), liver cirrhosis (low albumin), venous insufficiency.",
        "advice": "Elevate your legs when resting, restrict sodium/salt intake, and consult your doctor promptly for kidney and heart evaluation.",
    },
    "yellow_eyes": {
        "symptom": "Yellowing of Skin or Whites of the Eyes (Jaundice)",
        "key_biomarkers": ["Total Bilirubin", "Direct Bilirubin", "SGPT", "SGOT", "Alkaline Phosphatase"],
        "possible_causes": "Biliary duct obstruction (gallstones), viral hepatitis, acute drug-induced liver toxicity, hemolytic anemia.",
        "advice": "This is a significant medical finding requiring urgent medical assessment and liver imaging (ultrasound).",
    },
    "joint_pain": {
        "symptom": "Joint Pain, Stiffness, or Swelling",
        "key_biomarkers": ["Uric Acid", "ESR", "C-Reactive Protein", "Rheumatoid Factor (RF)", "Anti-CCP", "Serum Calcium"],
        "possible_causes": "Gout (hyperuricemia), rheumatoid arthritis, osteoarthritis, autoimmune connective tissue diseases, vitamin D deficiency.",
        "advice": "Avoid high-purine foods and alcohol if uric acid is suspected. Consult a rheumatologist or orthopedic doctor.",
    },
}


# ---------------------------------------------------------------------------
# 5. RETRIEVAL & SYNTHESIS HELPERS
# ---------------------------------------------------------------------------

def match_biomarker(query: str) -> Optional[Dict[str, Any]]:
    """Match a user question to a biomarker entry via canonical name, aliases, or fuzzy typo tolerance."""
    q_lower = query.lower()
    
    # 1. Direct canonical substring match
    for name, data in BIOMARKERS_DB.items():
        if name.lower() in q_lower:
            return {"name": name, **data}

    # 2. Exact word-boundary alias match
    for name, data in BIOMARKERS_DB.items():
        for alias in data.get("aliases", []):
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, q_lower):
                return {"name": name, **data}

    # 3. Fuzzy match for common misspellings (e.g. "haemogloblin", "colestrol", "platlets", "creatine")
    try:
        from rapidfuzz import fuzz
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower)
        best_match = None
        best_score = 0.0

        for name, data in BIOMARKERS_DB.items():
            candidates = [name.lower()] + [a.lower() for a in data.get("aliases", [])]
            for w in words:
                if len(w) >= 4:
                    for cand in candidates:
                        # Skip if lengths differ drastically
                        if abs(len(w) - len(cand)) > 4:
                            continue
                        score = fuzz.ratio(w, cand)
                        if score > 78 and score > best_score:
                            best_score = score
                            best_match = {"name": name, **data}

        if best_match and best_score >= 78:
            return best_match
    except Exception:
        pass

    return None


def match_condition(query: str) -> Optional[Dict[str, Any]]:
    """Match a condition/disease query."""
    q_lower = query.lower()
    keyword_map = {
        "anemia": "anemia", "anaemia": "anemia", "low blood": "anemia", "pale": "anemia",
        "diabetes": "diabetes", "diabetic": "diabetes", "high sugar": "diabetes", "prediabetes": "diabetes",
        "kidney": "kidney_disease", "renal": "kidney_disease", "ckd": "kidney_disease", "creatinine high": "kidney_disease",
        "liver": "liver_disease", "fatty liver": "liver_disease", "hepatitis": "liver_disease", "jaundice": "liver_disease",
        "thyroid": "thyroid", "tsh": "thyroid", "hypothyroid": "thyroid", "hyperthyroid": "thyroid",
        "gout": "gout", "uric acid": "gout", "podagra": "gout",
        "cholesterol": "dyslipidemia", "lipid": "dyslipidemia", "triglyceride": "dyslipidemia", "atherosclerosis": "dyslipidemia",
        "thrombocytopenia": "thrombocytopenia", "low platelet": "thrombocytopenia",
        "thrombocytosis": "thrombocytosis", "high platelet": "thrombocytosis",
        "leukocytosis": "leukocytosis", "high wbc": "leukocytosis", "high leucocyte": "leukocytosis",
        "leukopenia": "leukopenia", "low wbc": "leukopenia", "low leucocyte": "leukopenia",
        "anisocytosis": "anisocytosis", "rdw high": "anisocytosis",
        "microcytosis": "microcytosis", "small rbc": "microcytosis", "mcv low": "microcytosis",
        "macrocytosis": "macrocytosis", "large rbc": "macrocytosis", "mcv high": "macrocytosis",
        "hypochromia": "hypochromia", "hypochromic": "hypochromia",
    }
    for kw, key in keyword_map.items():
        if kw in q_lower:
            return CONDITIONS_DB.get(key)
    return None


def match_symptom(query: str) -> Optional[Dict[str, Any]]:
    """Match a symptom query."""
    q_lower = query.lower()
    if any(w in q_lower for kw in ["fatigue", "tired", "exhaust", "weakness", "low energy"] for w in [kw]):
        return SYMPTOMS_DB["fatigue"]
    if any(w in q_lower for kw in ["dizzy", "dizziness", "lightheaded", "faint", "vertigo"] for w in [kw]):
        return SYMPTOMS_DB["dizziness"]
    if any(w in q_lower for kw in ["swelling", "swollen", "edema", "puffy", "feet swell"] for w in [kw]):
        return SYMPTOMS_DB["swelling"]
    if any(w in q_lower for kw in ["yellow eye", "yellow skin", "jaundice", "yellowish"] for w in [kw]):
        return SYMPTOMS_DB["yellow_eyes"]
    if any(w in q_lower for kw in ["joint pain", "joints hurt", "knee pain", "arthritis", "big toe"] for w in [kw]):
        return SYMPTOMS_DB["joint_pain"]
    return None


def match_preparation(query: str) -> Optional[str]:
    """Match test preparation or medication fasting guidance."""
    q_lower = query.lower()
    if any(kw in q_lower for kw in ["fasting", "fast", "hours fast", "drink water", "water before test", "eat before"]):
        return PREPARATION_DB["fasting"]
    if any(kw in q_lower for kw in ["medication before", "medicine before", "thyroid medicine", "take pills", "biotin"]):
        return PREPARATION_DB["medication"]
    return None


def format_biomarker_response(info: Dict[str, Any], patient_test: Optional[Dict[str, Any]] = None) -> str:
    """Format an encyclopedic clinical explanation for a biomarker."""
    name = info["name"]
    category = info.get("category", "Laboratory Medicine")
    role = info.get("role", "")
    normal_range = info.get("normal_range", "Refer to specific laboratory assay")
    high_causes = info.get("high_causes", "")
    low_causes = info.get("low_causes", "")
    diet_lifestyle = info.get("diet_lifestyle", "")
    doc_q = info.get("doctor_questions", "")

    lines = [f"## 🧬 Clinical Biomarker Profile: **{name}**\n"]
    lines.append(f"**Category:** {category} | **Biological Role:** {role}\n")

    if patient_test:
        val = patient_test.get("value_raw", "")
        unit = patient_test.get("unit_normalized") or patient_test.get("unit_raw", "")
        status = patient_test.get("status", "NORMAL")
        ref = patient_test.get("reference_range_display") or patient_test.get("reference_range_raw", normal_range)
        emoji = {"NORMAL": "✅", "HIGH": "🔴", "LOW": "🔵", "CRITICAL_HIGH": "🚨", "CRITICAL_LOW": "🚨"}.get(status, "🟡")

        lines.append(f"### 📊 Your Report Result")
        lines.append(f"| Metric | Finding |")
        lines.append(f"|---|---|")
        lines.append(f"| **Your Level** | **{val} {unit}** |")
        lines.append(f"| **Reference Interval** | {ref} |")
        lines.append(f"| **Evaluation** | {emoji} **{status.replace('_', ' ')}** |\n")
    else:
        lines.append(f"| Reference Standard | Values |")
        lines.append(f"|---|---|")
        lines.append(f"| **Standard Reference Interval** | {normal_range} |\n")

    lines.append("### 🔍 Etiology & Clinical Significance")
    if high_causes:
        lines.append(f"• **Potential Causes of High Levels (Elevation):** {high_causes}")
    if low_causes:
        lines.append(f"• **Potential Causes of Low Levels (Deficiency):** {low_causes}")

    if diet_lifestyle:
        lines.append(f"\n### 🥗 Dietary & Lifestyle Recommendations")
        lines.append(diet_lifestyle)

    if doc_q:
        lines.append(f"\n### 🩺 Suggested Doctor Consultation Focus")
        lines.append(f"*{doc_q}*")

    return "\n".join(lines)


def format_condition_response(cond: Dict[str, Any]) -> str:
    """Format an encyclopedic clinical condition monograph."""
    lines = [f"## 🩺 Medical Condition Guide: **{cond['title']}**\n"]
    lines.append(f"{cond['summary']}\n")

    if "common_tests" in cond:
        lines.append(f"**Key Diagnostic Biomarkers:** {', '.join(cond['common_tests'])}\n")

    if "diagnostic_thresholds" in cond:
        lines.append("### 📊 Diagnostic Thresholds")
        for th in cond["diagnostic_thresholds"]:
            lines.append(f"• {th}")
        lines.append("")

    if "subtypes" in cond:
        lines.append("### 🧬 Clinical Subtypes & Mechanisms")
        for st in cond["subtypes"]:
            lines.append(f"• {st}")
        lines.append("")

    if "symptoms" in cond:
        lines.append("### ⚠️ Common Signs & Symptoms")
        if isinstance(cond["symptoms"], list):
            for s in cond["symptoms"]:
                lines.append(f"• {s}")
        else:
            lines.append(cond["symptoms"])
        lines.append("")

    if "action_plan" in cond:
        lines.append("### 📋 Recommended Clinical Action Plan")
        lines.append(cond["action_plan"])

    return "\n".join(lines)


def format_symptom_response(sym: Dict[str, Any]) -> str:
    """Format a symptom correlation guidance."""
    lines = [f"## 🩺 Clinical Symptom Correlation: **{sym['symptom']}**\n"]
    lines.append(f"### 🔍 Potential Underlying Causes\n{sym['possible_causes']}\n")
    lines.append(f"### 🧪 Relevant Laboratory Biomarkers to Review\n{', '.join(sym['key_biomarkers'])}\n")
    lines.append(f"### 📋 Actionable Guidance\n{sym['advice']}")
    return "\n".join(lines)
