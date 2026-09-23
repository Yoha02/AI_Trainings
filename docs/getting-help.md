# Ask a question and share what you learned

Use [Q&A Discussions](https://github.com/Yoha02/AI_Trainings/discussions/categories/q-a) for lesson questions, setup help, and results you want to understand. The question form asks for the lab, environment, expected result, and observed behavior so someone else can follow along.

If you have isolated a reproducible bug, [open an issue](https://github.com/Yoha02/AI_Trainings/issues/new). If you are unsure whether something is a bug, start in Q&A.

## Before posting

- Confirm which notebook and cell or app step you ran, and whether you changed it.
- Record your Python version, operating system, and chosen Google backend. The [lab companion](lab-companion.md) can identify your source revision and its automated check status.
- Share a small example using the included fictional handbook or sample products. Remove credentials and private document text.

## Common setup questions

**I created a .env file, but the lab cannot see my settings.** These labs do not load .env files automatically. Set the variables in the terminal that starts Jupyter or Streamlit, then restart the notebook kernel or application. Follow [Choose one Google backend](../README.md#choose-one-google-backend).

**I signed in with gcloud, but Vertex AI asks me to authenticate.** The gcloud CLI sign-in and Application Default Credentials are separate. Follow the README's ADC setup. A permission error needs a permissions/API check; repeated sign-ins alone may not resolve it.

**The RAG answer is wrong or missing information.** Check the extracted text and retrieved passages first. With the sample handbook, the return window is on page 1; international shipping is unspecified. Include your question, retrieved page numbers, and the answer when asking for help. An answer with a citation can still be wrong.

**The answer changes each time.** Generated wording can vary. Compare the facts and cited evidence, and keep model, prompt, and retrieval settings recorded when comparing experiments. Change one part at a time.

## Keep the answer useful

If a suggestion resolves your question, explain what worked and mark the helpful reply as the answer. If it does not, add the new result and the smallest example that still fails. A clear resolution helps the next learner with the same question.

These are maintainer-written setup notes. Community answers and examples can improve them over time.
