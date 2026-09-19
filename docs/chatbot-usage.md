# Chatbot usage guide — for the marketing team

The churn copilot answers plain-language questions about subscriber churn risk.
It is a single web page served by the same service that runs the model; the page
talks only to that service (no external sites, no tracking, nothing stored).

## 1. Start the demo stack (once per session)

From the project root:

```sh
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml ps     # wait for api "Up (healthy)"
```

The default configuration uses the Ollama model already running on this machine
(no multi-GB downloads). The first answer after a pause can take 20–60 seconds
while the model warms up; later answers are faster.

## 2. Open the page

Browser: http://127.0.0.1:8000/demo

## 3. Enter the access token

The token is the bearer secret configured for this deployment; locally it lives
in the repo-root `.env` file (`TELCO_API_BEARER_TOKEN=...`). Paste it into the
"Access token" field. It is kept in the page's memory only — refreshing the page
forgets it. Never share it in screenshots.

## 4. Ask questions

Type in the message box (Enter sends, Shift+Enter adds a line). Examples:

- Will a fiber-optic customer on a month-to-month contract churn?
- What about a DSL customer with no tech support who has been with us 3 months?
- Would a two-year contract customer who pays by mailed check leave us?

Follow-up questions in the same session build on the profile you have described
(e.g. "and what if they also have no online backup?").

## 5. Read the answer

- **Chat answer** — a plain-language sentence with the churn probability and the
  operating threshold used for the "Churner / Not churner" verdict.
- **Payload card** — churn probability, label, and the top-3 drivers (SHAP
  weights) that pushed the risk up or down.
- **Model identity** — the header shows which model and backend produced the
  answer (e.g. qwen3:4b-instruct-2507-q4_K_M · ollama). If the model is
  unreachable the header says so and no answer is fabricated.

## 6. Common errors

| What you see | What it means | Do this |
|---|---|---|
| `unauthorized — missing or invalid bearer token` | token empty or wrong | check the token from `.env` |
| `rate_limited — too many requests` | more than 60 turns/hour | wait, then retry |
| model chip "unreachable" | the LLM backend is down or not started | ask an operator to restart the stack |
| first answer slow | model cold start | nothing — next answers are fast |

The numbers are produced by the trained churn classifier; the language model
only translates your question into profile facts (it never invents numbers).
