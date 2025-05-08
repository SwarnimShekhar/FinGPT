# 🚨 FinGPT-Audit: LLM-Powered Financial Red Flag Auditor

> A lightweight, open-source system that uses a fine-tuned language model to **detect and explain financial red flags** in company disclosures, built entirely with free-tier tools and zero-cost inference.

---

## 🔍 Project Summary

**FinGPT-Audit** is a domain-specific QA system that uses an LLM to perform **red-flag audits** on financial disclosures such as 10-K filings, annual reports, and management commentary.

This project is ideal for:
- ✅ Portfolio showcases for ML / Data Science roles
- ✅ Audit / Fintech use-cases
- ✅ LLM evaluation in regulated industries

Built with **Gradio**, **LangChain** and **free-tier tooling**, the app provides both **detection** and **explanation** of red flags in real-time using natural language prompts.

---

## 📁 Project Structure
```bash
FinGPT-Audit/
├── app.py # Fast API app
├── gradio_app.py # Full Gradio UI interface
├── test_app.py # Testing API
├── test_fingpt_audit.py # For unit testing
├── fingpt-audit-model/ # Finetuned LLM Model
```

---

## ⚙️ Tech Stack

| Layer         | Tools / Libraries                        |
|---------------|------------------------------------------|
| LLM Backend   | google/flan-t5-small     |
| Prompting     | Custom domain-specific prompt templates  |
| Interface     | Gradio                                   |
| Backend       | Python 3.10+                             |
| Deployment    | Colab / Localhost                        |
| Use Case      | Financial Red Flag Detection (NLP QA)    |

---

## 🚀 Key Features

- ✅ **Plug-and-play LLM app** for financial document QA
- 🔍 Detects **red flags** using domain-optimized prompts
- 💬 Provides **explanations** and rationale per detection
- 🧠 Modular & editable for fine-tuning / customization
- 🖥️ Web app (Gradio) + CLI app available
- 🆓 Built entirely using **free-tier APIs** and **open tooling**

---

## 🧠 Example Use Cases

- Forensic analysis of **10-K filings**
- Auditing **Annual Reports** for misleading trends
- Screening investor call transcripts for **red-flag signals**
- Plug into internal audit dashboards / RAG systems

---

## 📸 Sample Output

Prompt:

> `"Scan this 10-K excerpt and tell me if there are any financial red flags."`

Response:

> **⚠️ Detected Red Flag:**
> - Unusually high goodwill impairment noted without rationale.
>
> **🧠 Explanation:**
> - Indicates potential overvaluation of acquired assets. This may reflect aggressive M&A accounting.

---

## 🛠️ How to Run

### Option 1: Terminal (CLI)

```bash
python app.py
Option 2: Gradio Web App
```
```bash
python gradio_app.py
A browser window will open with a clean Gradio UI.
```

## 📌 Future Work

✅ Extend to support multiple documents / PDF upload

⏳ Add vectorstore (FAISS / Chroma) for retrieval augmentation

📊 Integrate financial metric anomaly detection

🔒 Add explainability guardrails for enterprise settings

## 👨‍💻 Author
Swarnim Shekhar
Generative AI • NLP • ML Ops • Research

## 🏷️ Tags
LLM • LangChain • OpenAI • FinTech • Red Flag Detection • Audit AI • Gradio App
