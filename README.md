## Multi-Agent AI Researcher

A Streamlit app where a team of AI agents researches a topic on Hacker News, reads the articles, adds context from the web and writes the result as a summary, a blog post, a report or a LinkedIn post.

Based on the [Multi-Agent AI Researcher](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/advanced_ai_agents/multi_agent_apps/multi_agent_researcher) by Shubham Saboo (Apache 2.0).

### Features

- Real topic search on Hacker News (Algolia API)
- Three providers to choose from: OpenAI, OpenRouter (any model) or Ollama (local, free)
- A separate model for the team leader and for the agents
- Output format: summary, blog post, report or LinkedIn post, in English or French
- Structured output: for every story, you can see if the article was really read, what it says and what the web adds
- Live view of what each agent is doing during the research
- Dashboard with key numbers, tokens and cost of every search, and Report / Sources / Export tabs
- Lower input cost: articles are cut to 6,000 characters, 3 web results per search, short answers from the agents
- Export to Markdown or JSON
- Search history saved locally
- Guardrails: prompt injection and personal data detection, moderation (OpenAI), tool call limits, personal data removed from the answer, daily search limit

### Getting started

1. Clone the repository and go to the project folder

2. Create a virtual environment and install the dependencies

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

3. Add your API key

Copy `.env.example` to `.env` and fill in the key you want to use:

```
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
```

You can also type the key in the sidebar. For Ollama no key is needed: install [Ollama](https://ollama.com) and run `ollama pull llama3.2`.

4. Run the app

```bash
streamlit run research_agent.py
```

Set `DEBUG=1` in `.env` to see the agents' logs in the terminal.

### How it works

The app uses [Agno](https://github.com/agno-agi/agno) to build a team of three agents, coordinated by a team leader:

| Agent | Tool | Job |
|---|---|---|
| HackerNews Researcher | `search_hackernews` (Algolia API), `HackerNewsTools` | Finds stories about the topic |
| Article Reader | `Newspaper4kTools` | Reads the content of each article |
| Web Searcher | `DuckDuckGoTools` (Google backend) | Adds context from the web |

1. The guardrails check the query (prompt injection, personal data).
2. The leader asks the HackerNews Researcher for stories.
3. It sends each story url to the Article Reader.
4. It asks the Web Searcher for more context on each story.
5. It returns a `Report` object (Pydantic): one `Story` per article and the final text in the requested format and language.

### Safety

- Queries containing prompt injection patterns or personal data (emails, phone numbers, credit cards) are blocked before reaching the model.
- With OpenAI, queries are also checked with the OpenAI moderation API.
- Article and web content is treated as untrusted data: the agents are told never to follow instructions found inside it.
- Each agent has a tool call limit, to avoid loops and unexpected costs.
- Emails, phone numbers and card numbers found in the answer are replaced with `[removed]` before it is shown, exported or saved.
- With OpenAI and OpenRouter, searches are limited per day (10 by default, `MAX_SEARCHES_PER_DAY` in `.env`). Queries blocked by a guardrail are not counted.
- The agents are told to only use facts from their sources and to say when something could not be found.
- Every result shows a warning that it is AI-generated and should be checked.

### Known limitations

- The free web search is not always reliable: search engines often block automated requests. A search API like Tavily would be more stable.
- Hacker News search is sorted by relevance, so results are not always recent.
- The prompt injection guardrail is pattern based: a reworded attack can get through.
- Results change from one run to another, and small models sometimes miss details.
