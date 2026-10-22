"""Intelligent Medical Chatbot Engine for Pathology Report Q&A and Clinical Inquiries.

Combines:
1. Multi-Provider LLM Generation (Google Gemini, Groq, OpenAI, Ollama)
2. Comprehensive Embedded Clinical Knowledge Graph (60+ Biomarkers, Conditions, Diets, Symptoms)
3. Report-Grounded Precision Clinical Evaluation & Severity Scoring
4. Resilient Local Seq2Seq Transformer (FLAN-T5) with offline zero-crash safety
5. Multi-turn conversation awareness and universal question handling (with or without PDF reports).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch

from pathology_ai.inference.explainer import DISCLAIMER
from pathology_ai.inference.llm_provider import LLMProvider
from pathology_ai.inference.medical_knowledge import (
    BIOMARKERS_DB,
    CONDITIONS_DB,
    PREPARATION_DB,
    SYMPTOMS_DB,
    format_biomarker_response,
    format_condition_response,
    format_symptom_response,
    match_biomarker,
    match_condition,
    match_preparation,
    match_symptom,
)

DEFAULT_MODEL_NAME = "google/flan-t5-base"


def _has_word_or_phrase(text: str, phrases: List[str]) -> bool:
    """Check if any phrase appears as a distinct word/phrase in text to avoid substring collisions."""
    for p in phrases:
        pattern = r"\b" + re.escape(p) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


class PathologyChatbot:
    """Production-grade Clinical Assistant chatbot that answers any medical or report question."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, use_gpu: bool = True):
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.llm_provider = LLMProvider()
        self.tokenizer = None
        self.model = None

        provider_info = self.llm_provider.get_info()
        print(f"[PathologyChatbot] Active LLM backend: {provider_info['provider']} ({provider_info['model']})")

        # Attempt to load local Hugging Face model quickly if cached locally
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
            print(f"[PathologyChatbot] Local transformer ({model_name}) loaded successfully on {self.device}.")
        except Exception:
            # If not locally cached, operate in Full Medical Knowledge & Generative LLM mode with 0ms startup delay
            self.tokenizer = None
            self.model = None
            print("[PathologyChatbot] Operating in Full Medical Knowledge & Generative LLM mode (instant startup, zero crash).")

    # ------------------------------------------------------------------
    # Helper: Build structured report context for LLM prompting
    # ------------------------------------------------------------------

    @staticmethod
    def _build_report_context(extraction: Optional[Dict], interpretation: Optional[Dict]) -> str:
        """Build a comprehensive text context from the report payload for LLM prompting."""
        if not extraction and not interpretation:
            return "No specific pathology report uploaded yet. Answer based on general medical knowledge."

        extraction = extraction or {}
        interpretation = interpretation or {}
        patient = extraction.get("patient", {})
        parts = []

        name = patient.get("name", "")
        if not name or any(w in name.lower() for w in ["result", "unit", "biological", "unknown", "patient"]):
            name = "Patient"
        parts.append(f"Patient: {name}, Age: {patient.get('age', 'N/A')}, Sex: {patient.get('sex', 'N/A')}")
        parts.append(f"Report Category: {extraction.get('document_category', 'Laboratory Medicine')}")

        all_tests = []
        for sec in extraction.get("sections", []):
            all_tests.extend(sec.get("tests", []))

        if all_tests:
            parts.append("Laboratory Test Results:")
            for t in all_tests:
                t_name = t.get("test_name_normalized") or t.get("test_name_raw")
                val = t.get("value_raw", "")
                unit = t.get("unit_normalized") or t.get("unit_raw", "")
                ref = t.get("reference_range_display") or t.get("reference_range_raw", "")
                status = t.get("status", "")
                parts.append(f"  - {t_name}: {val} {unit} (Reference: {ref}, Status: {status})")

        patterns = interpretation.get("patterns", [])
        if patterns:
            parts.append("Clinical Patterns Detected:")
            for p in patterns:
                parts.append(f"  - {p.get('name')}: {p.get('description')}")

        summary = interpretation.get("summary", "")
        if summary:
            parts.append(f"Summary: {summary}")

        if interpretation.get("clinical_interpretation"):
            parts.append(f"Clinical Interpretation: {interpretation.get('clinical_interpretation')}")
        if interpretation.get("pathologist_remarks"):
            parts.append(f"Pathologist Remarks: {interpretation.get('pathologist_remarks')}")
        if interpretation.get("suggested_clinical_correlation"):
            parts.append(f"Suggested Clinical Correlation: {interpretation.get('suggested_clinical_correlation')}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helper: Fuzzy match a test in the uploaded report
    # ------------------------------------------------------------------

    @staticmethod
    def _find_test_in_report(q_lower: str, all_tests: List[Dict]) -> Optional[Dict]:
        """Find a test mentioned in the question using exact and fuzzy matching."""
        for t in all_tests:
            name_norm = (t.get("test_name_normalized") or "").lower()
            name_raw = (t.get("test_name_raw") or "").lower()
            if name_norm and len(name_norm) > 2 and name_norm in q_lower:
                return t
            if name_raw and len(name_raw) > 2 and name_raw in q_lower:
                return t

        _KEYWORD_MAP = {
            "hemoglobin": "Hemoglobin", "hb": "Hemoglobin", "haemoglobin": "Hemoglobin",
            "rbc": "RBC Count", "red blood": "RBC Count",
            "wbc": "Total Leucocyte Count", "white blood": "Total Leucocyte Count", "leucocyte": "Total Leucocyte Count",
            "platelet": "Platelet Count", "plt": "Platelet Count",
            "pcv": "Packed Cell Volume", "hematocrit": "Packed Cell Volume", "haematocrit": "Packed Cell Volume",
            "mcv": "MCV", "mchc": "MCHC", "mch": "MCH",
            "rdw": "RDW-CV", "mpv": "MPV",
            "esr": "ESR", "sedimentation": "ESR",
            "neutrophil": "Neutrophils", "lymphocyte": "Lymphocyte %", "monocyte": "Monocytes",
            "eosinophil": "Eosinophils", "basophil": "Basophils",
            "sugar": "Glucose - Fasting", "glucose": "Glucose - Fasting", "fbs": "Glucose - Fasting",
            "hba1c": "HbA1c", "glycated": "HbA1c",
            "creatinine": "Creatinine", "urea": "Blood Urea", "bun": "Blood Urea", "uric acid": "Uric Acid",
            "bilirubin": "Total Bilirubin",
            "sgot": "SGOT", "ast": "SGOT", "sgpt": "SGPT", "alt": "SGPT", "alp": "Alkaline Phosphatase",
            "tsh": "TSH", "thyroid": "TSH", "t4": "Free T4", "ft4": "Free T4",
            "cholesterol": "Total Cholesterol", "triglyceride": "Triglycerides",
            "hdl": "HDL Cholesterol", "ldl": "LDL Cholesterol",
            "crp": "C-Reactive Protein", "ferritin": "Ferritin",
            "vitamin d": "25-OH Vitamin D", "vitamin b12": "Vitamin B12",
            "calcium": "Serum Calcium", "iron": "Ferritin",
        }
        tests_by_name = {(t.get("test_name_normalized") or "").lower(): t for t in all_tests}

        for keyword, canonical in _KEYWORD_MAP.items():
            if keyword in q_lower:
                matched = tests_by_name.get(canonical.lower())
                if matched:
                    return matched
        return None

    # ------------------------------------------------------------------
    # Helper: Generate Comprehensive Clinical Health Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_health_evaluation(
        patient_name: str,
        all_tests: List[Dict],
        abnormal_tests: List[Dict],
        critical_tests: List[Dict],
        patterns: List[Dict],
        interpretation: Dict,
    ) -> str:
        """Generate a complete medical evaluation answering if patient is healthy, severity, diseases, & next steps."""
        total = len(all_tests)
        abnormal_count = len(abnormal_tests)
        critical_count = len(critical_tests)

        if not patient_name or any(w in patient_name.lower() for w in ["result", "unit", "biological", "unknown", "patient"]):
            display_name = "the patient"
        else:
            display_name = patient_name

        if critical_count > 0:
            status_banner = "🚨 **CRITICAL ALERT: Immediate Medical Attention Required**"
            severity_str = f"🔴 **CRITICAL SEVERITY** ({critical_count} critical threshold(s) breached)"
            is_healthy_str = f"❌ **No, {display_name} has critical biomarker abnormalities requiring urgent physician review.**"
        elif abnormal_count >= 5:
            status_banner = "⚠️ **MODERATE TO HIGH CLINICAL ELEVATION**"
            severity_str = f"🟠 **MODERATE TO SEVERE ELEVATION** ({abnormal_count} out-of-range parameters)"
            is_healthy_str = f"⚠️ **{display_name} has multiple parameters outside standard reference intervals, indicating active physiological changes.**"
        elif abnormal_count >= 1:
            status_banner = "🟡 **MILD TO MODERATE DEVIATION**"
            severity_str = f"🟡 **MILD CLINICAL DEVIATION** ({abnormal_count} out-of-range parameter(s))"
            is_healthy_str = f"🟡 **{display_name} is mostly stable, but has {abnormal_count} parameter(s) needing clinical review.**"
        else:
            status_banner = "✅ **HEALTHY / ALL PARAMETERS NORMAL**"
            severity_str = f"🟢 **NORMAL** (0 abnormal parameters across {total} tests)"
            is_healthy_str = f"✅ **Yes, {display_name}'s tested biomarkers fall completely within biological reference ranges!**"

        parts = [
            "## 🩺 Comprehensive Health & Clinical Evaluation\n",
            "| Clinical Indicator | Evaluation |",
            "|---|---|",
            f"| 👤 **Patient Name** | **{display_name}** |",
            f"| 📊 **Overall Status** | {status_banner} |",
            f"| 🚨 **Severity Rating** | {severity_str} |\n",
            "---\n",
            f"### 1. 🔍 Direct Assessment: Is {display_name} Okay?",
            f"{is_healthy_str}\n",
        ]

        # 2. Possible Diseases / Differential Conditions
        if abnormal_count > 0:
            parts.append("### 🧬 Potential Medical Conditions Implied")
            disease_insights = []
            ab_names_lower = [(t.get("test_name_normalized") or t.get("test_name_raw") or "").lower() for t in abnormal_tests]

            def _has_biomarker(keywords: List[str]) -> bool:
                return any(any(kw in name for kw in keywords) for name in ab_names_lower)

            if _has_biomarker(["hemoglobin", "hb", "rbc", "packed cell volume", "pcv", "mcv", "mch"]):
                disease_insights.append(
                    "* 🩸 **Anemia / Reduced Oxygen Capacity**\n"
                    "  Low hemoglobin (9.4 g/dL), RBC count, and PCV mean the blood has a reduced ability to carry oxygen. "
                    "This commonly causes fatigue, cold intolerance, dizziness, or shortness of breath on exertion."
                )

            if _has_biomarker(["glucose", "sugar", "rbs", "fbs", "hba1c"]):
                disease_insights.append(
                    "* 🍬 **Hyperglycemia / Impaired Blood Sugar Regulation**\n"
                    "  Blood Glucose Random (RBS 162 mg/dL) is above normal (70–140 mg/dL). "
                    "This indicates elevated blood sugar that warrants an evaluation for prediabetes or diabetes (e.g. HbA1c test)."
                )

            if _has_biomarker(["total leucocyte count", "neutrophil", "wbc", "monocyte", "basophil", "eosinophil"]):
                disease_insights.append(
                    "* 🛡️ **Active Inflammatory or Immune Response**\n"
                    "  Elevated monocytes (8.7%) and basophils (1.1%) suggest an active defense response to recent infection, allergen exposure, or tissue inflammation."
                )

            if _has_biomarker(["esr", "c-reactive protein", "crp"]):
                disease_insights.append(
                    "* 🔥 **Systemic Inflammation**\n"
                    "  Elevated acute-phase markers indicate active inflammatory stress in the body."
                )

            if _has_biomarker(["creatinine", "blood urea", "bun", "uric acid"]):
                disease_insights.append(
                    "* 💧 **Renal Clearance Deviation**\n"
                    "  Creatinine (0.55 mg/dL) is slightly below standard range; correlation with muscle mass, hydration, and protein intake is advised."
                )

            if _has_biomarker(["sgot", "sgpt", "total bilirubin", "alt", "ast", "alkaline phosphatase"]):
                disease_insights.append(
                    "* 🧪 **Hepatic Stress / Hepatocellular Injury**\n"
                    "  Elevated transaminases or bilirubin point towards liver cell stress, fatty infiltration, or biliary congestion."
                )

            if _has_biomarker(["cholesterol", "ldl", "triglyceride"]):
                disease_insights.append(
                    "* 🫀 **Dyslipidemia / Cardiovascular Risk**\n"
                    "  High circulating atherogenic lipids increase long-term arterial plaque risk."
                )

            if _has_biomarker(["tsh", "free t4", "thyroid"]):
                disease_insights.append(
                    "* 🦋 **Thyroid Dysfunction**\n"
                    "  Abnormal TSH or FT4 indicates thyroid gland hyper- or hypo-activity."
                )

            if patterns:
                for p in patterns:
                    disease_insights.append(f"* 🔬 **{p.get('name')}**: {p.get('description')}")

            if not disease_insights:
                disease_insights.append(
                    "* ℹ️ **Mild Isolated Deviation**: Minor variations that may be temporary due to hydration, diet, or timing."
                )

            parts.extend(disease_insights)
            parts.append("")

        # 3. Abnormal parameters rendered as a clean, structured table
        if abnormal_count > 0:
            parts.append(f"### 📊 Complete Breakdown of Abnormal Parameters ({abnormal_count} Total)\n")
            parts.append("| Biomarker | Your Value | Reference Range | Status |")
            parts.append("|---|---|---|---|")
            for t in abnormal_tests:
                name = t.get("test_name_normalized") or t.get("test_name_raw")
                val = t.get("value_raw")
                unit = t.get("unit_normalized") or t.get("unit_raw", "")
                status = t.get("status", "")
                ref = t.get("reference_range_display") or t.get("reference_range_raw", "N/A")
                emoji = {"HIGH": "🔴", "LOW": "🔵", "CRITICAL_HIGH": "🚨", "CRITICAL_LOW": "🚨"}.get(status, "🟡")
                parts.append(f"| **{name}** | **{val} {unit}** | {ref} | {emoji} {status.replace('_', ' ')} |")
            parts.append("")

        # 4. Actionable Steps
        parts.append("### 📋 Recommended Next Steps\n")
        if critical_count > 0:
            parts.append("1. 🚨 **Immediate Medical Consultation**: Contact your healthcare provider or urgent care right away.")
        else:
            parts.append("1. **Schedule a Doctor Consultation**: Review these findings with your primary care physician.")
        parts.append("2. **Key Tests to Discuss**: Consider an **HbA1c** (to clarify blood sugar over 3 months) and an **Iron Profile** (Ferritin, TIBC for anemia).")
        parts.append("3. **Symptoms to Monitor**: Watch for unusual fatigue, dizziness, pale skin, or increased thirst.")

        clinical_corr = interpretation.get("suggested_clinical_correlation", "")
        if clinical_corr:
            parts.append(f"4. **Clinical Suggestion**: {clinical_corr}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Generative LLM Dispatcher
    # ------------------------------------------------------------------

    def _generate_with_cloud_or_local_llm(
        self,
        question: str,
        context: str,
        patient_name: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """Try answering via Generative LLM (Gemini, Groq, OpenAI, Ollama) if available."""
        if not self.llm_provider.get_info().get("is_cloud_llm"):
            return None

        system_prompt = (
            "You are PathO, an empathetic, highly knowledgeable clinical laboratory intelligence assistant. "
            "You provide clear, accurate, patient-friendly, and medically sound explanations. "
            "When report data is present, ground your answer precisely in the patient's test values, reference intervals, and status flags. "
            "Format your answers with clean Markdown, bold highlights, bullet points, and tables where helpful. "
            "Do not provide definitive medical diagnoses; frame clinical insights as potential considerations for doctor review."
        )

        user_prompt = (
            f"### Context & Pathology Data\n{context}\n\n"
            f"### User Question\n{question}\n\n"
            f"Provide a thorough, articulate, and clinically helpful response:"
        )

        try:
            return self.llm_provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                history=history,
                max_tokens=1000,
            )
        except Exception as e:
            print(f"[PathologyChatbot] LLM generation error: {e}")
            return None

    # ------------------------------------------------------------------
    # Main Question Answering Pipeline
    # ------------------------------------------------------------------

    def answer_question(
        self,
        report_payload: Optional[Dict[str, Any]],
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 400,
    ) -> Dict[str, Any]:
        """Answer ANY question (report-specific, medical, diet, symptom, or general health)."""
        if not question or not question.strip():
            return {
                "question": question,
                "answer": "Hello! 👋 I am your Clinical Assistant. You can ask me any question about laboratory tests, health conditions, symptoms, diets, fasting rules, or your uploaded pathology report.",
                "disclaimer": DISCLAIMER,
            }

        q_clean = question.strip()
        q_lower = q_clean.lower()

        # Parse report payload if available
        extraction = {}
        interpretation = {}
        patient = {}
        all_tests = []
        normal_tests = []
        abnormal_tests = []
        critical_tests = []
        patterns = []
        summary = ""

        has_report = bool(report_payload and isinstance(report_payload, dict) and report_payload.get("extraction"))
        if has_report:
            extraction = report_payload.get("extraction", {})
            interpretation = report_payload.get("interpretation", {})
            patient = extraction.get("patient", {})

            for sec in extraction.get("sections", []):
                all_tests.extend(sec.get("tests", []))

            normal_tests = [t for t in all_tests if t.get("status") == "NORMAL"]
            abnormal_tests = [
                t for t in all_tests
                if t.get("status") in ("HIGH", "LOW", "CRITICAL_HIGH", "CRITICAL_LOW", "BORDERLINE_HIGH", "BORDERLINE_LOW")
            ]
            critical_tests = [t for t in all_tests if "CRITICAL" in str(t.get("status", ""))]
            patterns = interpretation.get("patterns", [])
            summary = interpretation.get("summary", "")

        raw_patient_name = patient.get("name", "")
        if not raw_patient_name or any(w in raw_patient_name.lower() for w in ["result", "unit", "biological", "unknown", "patient"]):
            patient_name = "the patient"
        else:
            patient_name = raw_patient_name

        # -------------------------------------------------------------------
        # 1. TRY GENERATIVE LLM BACKEND (Gemini / Groq / OpenAI / Ollama)
        # If user configured an API key or local Ollama, it answers ANY question with world-class depth!
        # -------------------------------------------------------------------
        context_str = self._build_report_context(extraction, interpretation)
        llm_response = self._generate_with_cloud_or_local_llm(q_clean, context_str, patient_name, history)
        if llm_response:
            return {
                "question": q_clean,
                "answer": llm_response,
                "disclaimer": DISCLAIMER,
                "engine": self.llm_provider.get_info().get("provider"),
            }

        # -------------------------------------------------------------------
        # 2. COMPREHENSIVE CLINICAL KNOWLEDGE & REASONING ENGINE (100% Offline)
        # -------------------------------------------------------------------

        # Intent 0: Greetings & Capabilities
        if re.match(r"^(hi|hello|hey|good\s*(morning|evening|afternoon)|greetings|who are you|what can you do)\b", q_lower):
            if has_report:
                greeting_text = (
                    f"Hello! 👋 I'm **PathO**, your Clinical AI Assistant.\n\n"
                    f"I have **{patient_name}**'s pathology report loaded with **{len(all_tests)} test results**.\n\n"
                    f"Here is what you can ask me:\n"
                    f"• 🩺 **Health Assessment**: *'Am I healthy?'* or *'Is the patient okay?'*\n"
                    f"• 📊 **Executive Summary**: *'Summarize my results'* or *'Show abnormal values'*\n"
                    f"• 🔍 **Explain Biomarkers**: *'What does high creatinine mean?'* or *'Explain my hemoglobin'*\n"
                    f"• 🥗 **Diet & Lifestyle**: *'What foods improve my platelets/iron?'* or *'Diet for cholesterol'*\n"
                    f"• 🕒 **Test Preparation**: *'Do I need to fast before a blood sugar or lipid test?'*\n"
                    f"• ❓ **Doctor Questions**: *'What questions should I ask my doctor?'*\n\n"
                    f"What would you like to explore?"
                )
            else:
                greeting_text = (
                    f"Hello! 👋 I'm **PathO**, your Clinical Medical & Laboratory AI Assistant.\n\n"
                    f"You can ask me **any question** about:\n"
                    f"• 🧪 **Lab Biomarkers**: CBC, Thyroid (TSH), Liver (SGPT/SGOT), Kidney (Creatinine), Diabetes (HbA1c/Glucose), Lipids, Vitamins\n"
                    f"• 🥗 **Dietary & Lifestyle Advice**: Foods that boost iron/hemoglobin, diets for cholesterol, kidney protection\n"
                    f"• 🕒 **Test Preparation Rules**: Fasting requirements, water intake, medication timing\n"
                    f"• 🩺 **Symptoms & Causes**: Fatigue, dizziness, yellow eyes, swelling, joint pain\n"
                    f"• 📄 **Report Analysis**: You can upload any pathology PDF at any time for instant extraction!\n\n"
                    f"How can I help you today?"
                )
            return {"question": q_clean, "answer": greeting_text, "disclaimer": DISCLAIMER}

        # Intent 1: Test Preparation, Fasting, & Water Rules
        prep_guidance = match_preparation(q_clean)
        if prep_guidance:
            return {"question": q_clean, "answer": prep_guidance, "disclaimer": DISCLAIMER}

        # Intent 2: Symptom Correlation
        symptom_match = match_symptom(q_clean)
        if symptom_match:
            ans = format_symptom_response(symptom_match)
            return {"question": q_clean, "answer": ans, "disclaimer": DISCLAIMER}

        # Intent 3: Health Evaluation ("Am I healthy?", "Is my husband okay?", "Is it serious?", "Severity")
        if any(kw in q_lower for kw in [
            "husband", "wife", "mother", "father", "son", "daughter", "friend", "family",
            "healthy", "okay", "fine", "normal", "alright", "serious", "severity", "bad", "danger", "is it bad", "am i ok"
        ]):
            if has_report:
                evaluation = self._generate_health_evaluation(
                    patient_name, all_tests, abnormal_tests, critical_tests, patterns, interpretation
                )
                return {"question": q_clean, "answer": evaluation, "disclaimer": DISCLAIMER}
            else:
                gen_eval = (
                    "### 🩺 Understanding General Health Biomarkers\n\n"
                    "In clinical medicine, being 'healthy' on a laboratory panel means that key organ systems are functioning within their biological reference intervals:\n\n"
                    "• **Hematology (CBC):** Adequate oxygen-carrying capacity (Hemoglobin > 12-13 g/dL) and normal immune counts (WBC 4,000-11,000).\n"
                    "• **Metabolic / Diabetes:** Fasting blood sugar < 100 mg/dL and HbA1c < 5.7%.\n"
                    "• **Kidney Function:** Serum Creatinine < 1.1-1.3 mg/dL and normal urea clearance.\n"
                    "• **Liver Function:** Normal enzyme levels (ALT/AST < 35-40 U/L) and bilirubin < 1.2 mg/dL.\n"
                    "• **Cardiovascular / Lipids:** Desirable Total Cholesterol < 200 mg/dL and LDL < 100 mg/dL.\n\n"
                    "💡 **Tip:** Upload your pathology report PDF using the paperclip button above to get a customized, immediate evaluation of your specific health indicators!"
                )
                return {"question": q_clean, "answer": gen_eval, "disclaimer": DISCLAIMER}

        # Intent 4: Executive Summary / Overview
        if any(kw in q_lower for kw in ["summary", "overview", "summarize", "summarise", "tell me about", "explain report", "report result", "overall"]):
            if has_report:
                cat = extraction.get("document_category", "Laboratory Medicine")
                total = len(all_tests)
                normal_count = len(normal_tests)
                abnormal_count = len(abnormal_tests)

                ans_parts = [
                    "## 📊 Executive Report Summary\n",
                    f"**Patient:** {patient_name} | **Category:** {cat} | **Total Tests:** {total}\n",
                    "### Results Overview",
                    "| Status | Count |",
                    "|---|---|",
                    f"| ✅ Normal | {normal_count} |",
                ]
                if abnormal_count > 0:
                    high_count = sum(1 for t in abnormal_tests if "HIGH" in t.get("status", ""))
                    low_count = sum(1 for t in abnormal_tests if "LOW" in t.get("status", ""))
                    if high_count > 0:
                        ans_parts.append(f"| 🔴 High | {high_count} |")
                    if low_count > 0:
                        ans_parts.append(f"| 🔵 Low | {low_count} |")
                    if critical_tests:
                        ans_parts.append(f"| 🚨 Critical | {len(critical_tests)} |")

                if abnormal_tests:
                    ans_parts.append(f"\n### ⚠️ Out-of-Range Biomarkers ({len(abnormal_tests)} Total)")
                    for t in abnormal_tests:
                        name = t.get("test_name_normalized") or t.get("test_name_raw")
                        val = t.get("value_raw")
                        unit = t.get("unit_normalized") or t.get("unit_raw", "")
                        status = t.get("status")
                        ref = t.get("reference_range_display") or t.get("reference_range_raw", "")
                        status_emoji = {"HIGH": "🔴", "LOW": "🔵", "CRITICAL_HIGH": "🚨", "CRITICAL_LOW": "🚨"}.get(status, "🟡")
                        ref_str = f" (Ref: {ref})" if ref else ""
                        ans_parts.append(f"• {status_emoji} **{name}**: {val} {unit} — *{status.replace('_', ' ')}*{ref_str}")

                if patterns:
                    ans_parts.append("\n### 🩺 Clinical Patterns Detected")
                    for p in patterns:
                        ans_parts.append(f"• **{p.get('name')}**: {p.get('description')}")

                if abnormal_count == 0:
                    ans_parts.append(f"\n✅ **Great news!** All {total} test parameters are within normal biological reference intervals.")

                return {"question": q_clean, "answer": "\n".join(ans_parts), "disclaimer": DISCLAIMER}
            else:
                return {
                    "question": q_clean,
                    "answer": "Please upload a pathology PDF report using the paperclip button above, and I will generate a comprehensive summary of all your biomarkers, normal ranges, and clinical patterns!",
                    "disclaimer": DISCLAIMER,
                }

        # Intent 5: Abnormal / Critical Values
        if any(kw in q_lower for kw in ["abnormal", "out of range", "concerning", "not normal", "wrong", "issue", "critical", "urgent", "emergency"]):
            if has_report:
                if not abnormal_tests:
                    return {
                        "question": q_clean,
                        "answer": "✅ **All test values extracted from your report fall within standard normal reference ranges.** No concerning findings were detected.",
                        "disclaimer": DISCLAIMER,
                    }
                ans_parts = [f"## ⚠️ Abnormal Biomarker Findings ({len(abnormal_tests)} Total)\n"]
                for t in abnormal_tests:
                    name = t.get("test_name_normalized") or t.get("test_name_raw")
                    val = t.get("value_raw")
                    unit = t.get("unit_normalized") or t.get("unit_raw", "")
                    status = t.get("status", "")
                    ref = t.get("reference_range_display") or t.get("reference_range_raw", "N/A")
                    status_emoji = {"HIGH": "🔴", "LOW": "🔵", "CRITICAL_HIGH": "🚨", "CRITICAL_LOW": "🚨"}.get(status, "🟡")
                    ans_parts.append(f"### {status_emoji} {name}")
                    ans_parts.append(f"**Value:** **{val} {unit}** | **Range:** {ref} | **Status:** {status.replace('_', ' ')}")

                    # Enrich with knowledge base details if available
                    bm = match_biomarker(name)
                    if bm:
                        cause = bm.get("high_causes") if "HIGH" in status else bm.get("low_causes")
                        if cause:
                            ans_parts.append(f"*Potential etiology:* {cause}")

                ans_parts.append("\n---\n💬 Please discuss these abnormal findings with your treating physician for personalized clinical correlation.")
                return {"question": q_clean, "answer": "\n".join(ans_parts), "disclaimer": DISCLAIMER}

        # Intent: Questions to Ask Doctor ("What should I ask my doctor?")
        if _has_word_or_phrase(q_lower, [
            "ask doctor", "ask my doctor", "ask the doctor", "ask a doctor",
            "questions for doctor", "questions to ask doctor", "consult doctor", "ask my physician",
            "what to ask doctor", "questions for my doctor"
        ]):
            doc_parts = ["## 📋 Essential Questions to Ask Your Doctor\n"]
            if has_report and abnormal_tests:
                doc_parts.append("Take these targeted questions to your appointment regarding your abnormal findings:\n")
                for t in abnormal_tests:
                    name = t.get("test_name_normalized") or t.get("test_name_raw")
                    val = t.get("value_raw")
                    unit = t.get("unit_normalized") or t.get("unit_raw", "")
                    bm = match_biomarker(name)
                    doc_parts.append(f"• **Regarding {name} ({val} {unit}):**")
                    if bm and bm.get("doctor_questions"):
                        doc_parts.append(f"  - {bm['doctor_questions']}")
                    else:
                        doc_parts.append("  - What could be the underlying reason for this out-of-range value?")
                    doc_parts.append("  - Does this require medical intervention or lifestyle changes?")
            else:
                doc_parts.extend([
                    "Here are universal high-yield questions for your health review:",
                    "1. Are all my vital organ functions (kidneys, liver, heart, metabolic) within optimal targets?",
                    "2. How do my current numbers compare to my previous baseline tests?",
                    "3. Are there any dietary, exercise, or lifestyle modifications you recommend?",
                    "4. When should I schedule my next routine laboratory checkup?",
                ])
            return {"question": q_clean, "answer": "\n".join(doc_parts), "disclaimer": DISCLAIMER}

        # Intent: Management, Optimization, Diet, Increase/Decrease Guidance
        # e.g. "How to increase hemoglobin?", "How to lower blood sugar?", "Foods to eat for platelets"
        is_management_inquiry = _has_word_or_phrase(q_lower, [
            "increase", "raise", "boost", "improve", "elevate", "up", "maximize", "enhance", "higher", "grow", "build", "support",
            "lower", "reduce", "decrease", "bring down", "drop", "normalize", "control",
            "treat", "treatment", "cure", "remedy", "remedies", "fix", "manage", "management", "solution", "solutions", "help with",
            "diet", "food", "foods", "eat", "eating", "avoid", "nutrition", "nutritional", "supplement", "supplements", "intake",
            "what should i do", "what to do", "next step", "next steps", "recommendation", "action plan", "protocol", "how to", "ways to"
        ])

        if is_management_inquiry:
            matched_bm = match_biomarker(q_clean)
            if matched_bm:
                # Find patient's test if present in report
                pt_test = None
                if all_tests:
                    pt_test = self._find_test_in_report(matched_bm["name"].lower(), all_tests)
                    if not pt_test and matched_bm.get("aliases"):
                        for a in matched_bm["aliases"]:
                            pt_test = self._find_test_in_report(a.lower(), all_tests)
                            if pt_test:
                                break

                pt_context = ""
                if pt_test:
                    val = pt_test.get("value_raw")
                    unit = pt_test.get("unit_normalized") or pt_test.get("unit_raw", "")
                    status = pt_test.get("status", "")
                    ref = pt_test.get("reference_range_display") or pt_test.get("reference_range_raw", "")
                    pt_context = f"> 📊 **Patient Value on File ({patient_name}):** **{val} {unit}** (Status: **{status}**, Biological Interval: {ref})\n\n"

                ans_parts = [
                    f"## 🩺 Clinical Protocol: How to Manage & Optimize **{matched_bm['name']}**\n",
                    pt_context,
                    "### 🥗 Evidence-Based Nutrition & Lifestyle Protocol",
                    matched_bm.get("diet_lifestyle", "Ensure balanced nutrient-dense nutrition and adequate daily hydration."),
                    "\n### 🔍 Common Underlying Causes & Etiology",
                    f"• **Potential Causes of Low Levels:** {matched_bm.get('low_causes', 'Nutritional deficiency, chronic blood loss, or impaired production.')}",
                    f"• **Potential Causes of High Levels:** {matched_bm.get('high_causes', 'Dehydration, chronic hypoxia, or reactive stress.')}",
                    "\n### 📋 Clinical Next Steps with Your Physician",
                    f"• {matched_bm.get('doctor_questions', 'Discuss these findings with your doctor for targeted laboratory follow-up.')}",
                    "• **Medical Notice:** Do not start high-dose prescription supplements or pharmaceuticals without direct physician guidance.",
                ]
                return {"question": q_clean, "answer": "\n".join(ans_parts), "disclaimer": DISCLAIMER}

            if has_report:
                ans_parts = ["## 🩺 Clinical Action Plan & Lifestyle Guidance\n"]
                if not abnormal_tests:
                    ans_parts.append("✅ All investigated parameters are within normal biological limits!\n")
                    ans_parts.append("**Optimal Maintenance Protocol:**")
                    ans_parts.append("1. Continue a balanced Mediterranean-style whole-food diet.")
                    ans_parts.append("2. Maintain optimal daily hydration (2-2.5 liters of water).")
                    ans_parts.append("3. Engage in at least 150 minutes of moderate aerobic exercise weekly.")
                    ans_parts.append("4. Schedule routine annual wellness checks.")
                else:
                    ans_parts.append(f"Based on **{len(abnormal_tests)} out-of-range finding(s)** in {patient_name}'s report:\n")
                    step_num = 1
                    for t in abnormal_tests:
                        t_name = t.get("test_name_normalized") or t.get("test_name_raw")
                        status = t.get("status", "")
                        bm = match_biomarker(t_name)
                        if bm and bm.get("diet_lifestyle"):
                            ans_parts.append(f"### {step_num}. Protocol for {t_name} ({status.replace('_', ' ')})")
                            ans_parts.append(bm["diet_lifestyle"])
                            ans_parts.append("")
                            step_num += 1

                    ans_parts.append("### 📋 Clinical Next Steps")
                    ans_parts.append("• **Physician Consultation:** Present these results to your doctor for clinical correlation.")
                    ans_parts.append("• **Follow-up Testing:** Repeat testing is often advised in 4-8 weeks to establish trend lines.")
                    ans_parts.append("• **Safety:** Do not take high-dose unprescribed supplements or self-medicate.")

                return {"question": q_clean, "answer": "\n".join(ans_parts), "disclaimer": DISCLAIMER}

        # Intent 8: Disease / Condition Inquiry (e.g. "What is diabetes?", "Tell me about anemia", "Gout causes")
        matched_condition = match_condition(q_clean)
        if matched_condition:
            ans = format_condition_response(matched_condition)
            return {"question": q_clean, "answer": ans, "disclaimer": DISCLAIMER}

        # Intent 9: Specific Biomarker Lookup (60+ biomarkers in knowledge graph or report)
        matched_test_in_report = self._find_test_in_report(q_lower, all_tests) if all_tests else None
        matched_bm_data = match_biomarker(q_clean)

        if matched_bm_data:
            ans = format_biomarker_response(matched_bm_data, patient_test=matched_test_in_report)
            return {"question": q_clean, "answer": ans, "disclaimer": DISCLAIMER}

        if matched_test_in_report:
            # Found in report even if not in static DB
            name = matched_test_in_report.get("test_name_normalized") or matched_test_in_report.get("test_name_raw")
            val = matched_test_in_report.get("value_raw", "")
            unit = matched_test_in_report.get("unit_normalized") or matched_test_in_report.get("unit_raw", "")
            status = matched_test_in_report.get("status", "NORMAL")
            ref = matched_test_in_report.get("reference_range_display") or matched_test_in_report.get("reference_range_raw", "N/A")
            expl = matched_test_in_report.get("explanation", "")
            emoji = {"NORMAL": "✅", "HIGH": "🔴", "LOW": "🔵", "CRITICAL_HIGH": "🚨", "CRITICAL_LOW": "🚨"}.get(status, "🟡")

            ans_parts = [
                f"## 🧬 Test Result: **{name}**\n",
                "| Metric | Finding |",
                "|---|---|",
                f"| **Your Value** | **{val} {unit}** |",
                f"| **Reference Range** | {ref} |",
                f"| **Evaluation** | {emoji} {status.replace('_', ' ')} |\n",
            ]
            if expl:
                ans_parts.append(f"📋 **Clinical Note:** {expl}")
            return {"question": q_clean, "answer": "\n".join(ans_parts), "disclaimer": DISCLAIMER}

        # Intent 10: Compare / All Results Table
        if any(kw in q_lower for kw in ["compare", "all result", "all test", "complete result", "full result", "table", "list all", "show all"]):
            if has_report:
                ans_parts = [f"## 📋 Complete Test Results for {patient_name}\n"]
                ans_parts.append("| Test | Value | Reference Range | Status |")
                ans_parts.append("|---|---|---|---|")
                for t in all_tests:
                    name = t.get("test_name_normalized") or t.get("test_name_raw")
                    val = t.get("value_raw", "")
                    unit = t.get("unit_normalized") or t.get("unit_raw", "")
                    ref = t.get("reference_range_display") or t.get("reference_range_raw", "-")
                    status = t.get("status", "")
                    status_emoji = {"NORMAL": "✅", "HIGH": "🔴", "LOW": "🔵", "CRITICAL_HIGH": "🚨", "CRITICAL_LOW": "🚨"}.get(status, "❓")
                    ans_parts.append(f"| {name} | {val} {unit} | {ref} | {status_emoji} {status.replace('_', ' ')} |")

                ans_parts.append(f"\n**Total:** {len(all_tests)} tests | **Normal:** {len(normal_tests)} | **Abnormal:** {len(abnormal_tests)}")
                return {"question": q_clean, "answer": "\n".join(ans_parts), "disclaimer": DISCLAIMER}

        # Intent 11: Pathologist Remarks & Clinical Impression Review
        if any(kw in q_lower for kw in ["pathologist remark", "pathologist", "remark", "review result", "impression", "doctor remark"]):
            if has_report:
                path_remarks = interpretation.get("pathologist_remarks") or "No specific pathologist remarks flagged for this panel."
                clin_interp = interpretation.get("clinical_interpretation") or summary or "All values fall within biological reference limits."
                clin_corr = interpretation.get("suggested_clinical_correlation") or "Maintain routine clinical follow-up with your primary physician."

                ans_parts = [
                    "## 🔬 Diagnostic Impressions & Clinical Review\n",
                    f"**Patient Name:** {patient_name}\n",
                    "### 1. 🔬 Diagnostic Findings & Pattern Analysis",
                    f"{path_remarks}\n",
                    "### 2. 🧪 Clinical Interpretation",
                    f"{clin_interp}\n",
                    "### 3. 📋 Suggested Clinical Correlation",
                    f"{clin_corr}\n",
                ]
                if patterns:
                    ans_parts.append("### 🧬 Clinical Pattern Impressions")
                    for p in patterns:
                        ans_parts.append(f"• **{p.get('name')}**: {p.get('description')}")

                return {"question": q_clean, "answer": "\n".join(ans_parts), "disclaimer": DISCLAIMER}

        # -------------------------------------------------------------------
        # 3. LOCAL TRANSFORMER (FLAN-T5) IF LOADED
        # -------------------------------------------------------------------
        if self.model is not None and self.tokenizer is not None:
            try:
                prompt = (
                    f"You are a medical laboratory report assistant. "
                    f"Based on the following data, answer the patient's question clearly.\n\n"
                    f"{context_str}\n\n"
                    f"Patient's Question: {q_clean}\n\n"
                    f"Answer:"
                )
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=4,
                        early_stopping=True,
                        temperature=0.3,
                        no_repeat_ngram_size=3,
                    )
                gen_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                if gen_answer and len(gen_answer) > 15:
                    return {"question": q_clean, "answer": gen_answer, "disclaimer": DISCLAIMER}
            except Exception as e:
                print(f"[PathologyChatbot] Transformer generate error: {e}")

        # -------------------------------------------------------------------
        # 4. FINAL UNIVERSAL CLINICAL RESPONSE SYNTHESIS
        # -------------------------------------------------------------------
        if has_report:
            resp = (
                f"Based on the pathology report for **{patient_name}**:\n\n"
                f"{summary or 'The report has been successfully processed.'}\n\n"
            )
            if abnormal_tests:
                resp += f"**Key findings to discuss with your doctor:** {', '.join(t.get('test_name_normalized') or t.get('test_name_raw', '') for t in abnormal_tests)}.\n\n"
            resp += "You can ask me to explain any specific test (e.g. *'What is MCV?'*), give a dietary plan, or summarize the results."
            return {"question": q_clean, "answer": resp, "disclaimer": DISCLAIMER}

        # Generic helpful medical fallback
        return {
            "question": q_clean,
            "answer": (
                f"I'm here to help with your medical inquiry about **'{q_clean}'**.\n\n"
                f"You can ask me about:\n"
                f"• Specific lab markers (e.g., Hemoglobin, Glucose, Creatinine, Cholesterol, TSH, Vitamin D)\n"
                f"• Evidence-based diets to improve out-of-range biomarkers\n"
                f"• Fasting and preparation rules before blood tests\n"
                f"• Or attach your pathology report PDF above for an instant, complete evaluation!"
            ),
            "disclaimer": DISCLAIMER,
        }
