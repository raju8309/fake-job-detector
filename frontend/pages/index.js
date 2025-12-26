import { useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [company, setCompany] = useState("");
  const [location, setLocation] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [isWarmingUp, setIsWarmingUp] = useState(false);

  function renderHighlightedText(text, explainability) {
    if (!text || !explainability || !Array.isArray(explainability.tokens)) {
      return <span>{text}</span>;
    }

    const tokenMap = new Map();
    for (const t of explainability.tokens) {
      if (!t || !t.token) continue;
      const key = String(t.token).trim().toLowerCase();
      if (!key) continue;
      tokenMap.set(key, t);
    }

    const parts = text.split(/(\s+)/);
    return parts.map((part, idx) => {
      if (/^\s+$/.test(part)) {
        return <span key={idx}>{part}</span>;
      }

      const normalized = part
        .toLowerCase()
        .replace(/^[^a-z0-9]+|[^a-z0-9]+$/gi, "");

      const tokenInfo = tokenMap.get(normalized);
      if (!tokenInfo) {
        return <span key={idx}>{part}</span>;
      }

      const intensity = Math.max(0, Math.min(1, tokenInfo.impact_abs ?? 0));
      const bg = `rgba(239, 68, 68, ${0.15 + 0.35 * intensity})`;

      return (
        <mark
          key={idx}
          className="why-highlight"
          style={{ backgroundColor: bg }}
          title={`Impact: ${Number(tokenInfo.impact).toFixed(4)}`}
        >
          {part}
        </mark>
      );
    });
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setResult(null);
    setIsWarmingUp(false);

    if (!title.trim() || !description.trim()) {
      setError("Please enter both Job Title and Job Description.");
      return;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, 60000); // 60 second timeout for Render cold starts

    try {
      setLoading(true);
      
      // Show warming up message after 2 seconds
      const warmupTimer = setTimeout(() => {
        setIsWarmingUp(true);
      }, 2000);

      // FIXED: Proper template literal syntax
      const res = await fetch(`${API_BASE}/analyze-job`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title,
          description,
          company,
          location,
        }),
        signal: controller.signal,
      });

      clearTimeout(warmupTimer);
      clearTimeout(timeoutId);

      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      clearTimeout(timeoutId);
      
      if (err.name === "AbortError") {
        setError(
          "Request timed out. The server might be waking up from sleep. Please try again in a moment."
        );
      } else if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
        setError(
          "Cannot connect to the server. Please check if the backend is running or try again later."
        );
      } else {
        setError("Something went wrong while analyzing the job. Please try again.");
      }
    } finally {
      setLoading(false);
      setIsWarmingUp(false);
    }
  }

  return (
    <div className="page-root">
      {/* Hero */}
      <header className="hero">
        <h1 className="hero-title">
          Detect Fake Jobs
          <br />
          Protect Your Career
        </h1>
        <p className="hero-sub">
          AI-powered analysis to identify fraudulent job postings in seconds.
          Stay safe from scams and focus on real opportunities.
        </p>
        <p className="hero-bullets">
          • Instant Analysis &nbsp;&nbsp;• AI-Powered Detection &nbsp;&nbsp;• Free
          to Use
        </p>
      </header>

      {/* Main card */}
      <main className="main">
        <section className="card card-main">
          <form onSubmit={handleSubmit}>
            <label className="field-label">Job Title</label>
            <input
              className="field-input"
              placeholder="e.g., Senior Software Engineer"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />

            <label className="field-label">Job Description</label>
            <textarea
              className="field-textarea"
              placeholder="Paste the complete job posting here..."
              rows={6}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />

            <div className="field-row">
              <div className="field-col">
                <label className="field-label">Company (optional)</label>
                <input
                  className="field-input"
                  placeholder="e.g., Acme Corp"
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                />
              </div>
              <div className="field-col">
                <label className="field-label">Location (optional)</label>
                <input
                  className="field-input"
                  placeholder="e.g., New York, NY"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                />
              </div>
            </div>

            {error && <p className="error-text">{error}</p>}

            {/* Loading state with helpful messages */}
            {loading && (
              <div className="loading-state">
                <div className="spinner"></div>
                <p>
                  {isWarmingUp
                    ? "🔥 Warming up the server... This happens after inactivity. Thanks for your patience!"
                    : "🔎 Analyzing job posting..."}
                </p>
              </div>
            )}

            <button className="primary-btn" type="submit" disabled={loading}>
              {loading ? "Analyzing..." : "🔎 Analyze Job Posting"}
            </button>
          </form>
        </section>

        {/* Result */}
        {result && (
          <section className="card card-result">
            <div
              className={`verdict ${
                result.verdict === "fake" ? "verdict-fake" : "verdict-real"
              }`}
            >
              {result.verdict === "fake"
                ? "⚠️ This job looks FAKE (High Risk)"
                : "✅ This job appears REAL (Low Risk)"}
            </div>

            <div className="metrics">
              <div className="metric">
                <small>Real (Final Score)</small>
                <h3 className="metric-real">{result.real_pct}%</h3>
              </div>
              <div className="metric">
                <small>Fake (Final Score)</small>
                <h3 className="metric-fake">{result.fake_pct}%</h3>
              </div>
            </div>

            <div
              className="conf-bar"
              style={{
                "--realPct": `${result.real_pct}%`,
              }}
            />

            <p className="model-caption">
              <strong>Model-only score (for transparency):</strong> Real{" "}
              {result.model_real_pct}% · Fake {result.model_fake_pct}%
            </p>

            {result.explainability &&
              result.explainability.tokens &&
              result.explainability.tokens.length > 0 && (
                <>
                  <div className="insights-header">
                    <h2>🧠 Why the AI is Suspicious</h2>
                    <p>
                      Highlighted words below contributed the most to the AI's
                      fake-risk score.
                    </p>
                  </div>

                  <div className="why-box">
                    <div className="why-text">
                      {renderHighlightedText(description, result.explainability)}
                    </div>
                    <div className="why-top">
                      <strong>Top signals:</strong>{" "}
                      {result.explainability.tokens
                        .slice(0, 8)
                        .map((t) => t.token)
                        .join(", ")}
                    </div>
                  </div>
                </>
              )}

            {result.rag && (
              <>
                <div className="insights-header">
                  <h2>🗃️ Memory Bank (RAG)</h2>
                  <p>
                    Similarity search against previously seen scam job postings.
                  </p>
                </div>

                <div className="why-box">
                  <p className="small-text">
                    <strong>Max similarity:</strong>{" "}
                    {Math.round((result.rag.max_similarity || 0) * 100)}%
                  </p>

                  {result.rag.matches && result.rag.matches.length > 0 ? (
                    <div className="rag-list">
                      {result.rag.matches.slice(0, 3).map((m, idx) => (
                        <div key={idx} className="rag-item">
                          <p className="small-text">
                            <strong>
                              {Math.round((m.similarity || 0) * 100)}%
                            </strong>{" "}
                            similar — {m.item?.title}
                            {m.item?.location ? ` (${m.item.location})` : ""}
                          </p>
                          {m.item?.snippet && (
                            <p className="small-text">{m.item.snippet}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="small-text">No similar scams found.</p>
                  )}
                </div>
              </>
            )}

            {/* Verification insights */}
            <div className="insights-header">
              <h2>🔍 Verification Insights</h2>
              <p>
                Cross-checking public job indices, email safety, and risky scam
                keywords.
              </p>
            </div>

            <div className="info-grid">
              {/* Adzuna card */}
              <div className="info-card insight-card">
                <h4 data-icon="🌐">Public Index (Adzuna)</h4>
                {result.verification.api.found ? (
                  <>
                    <p>
                      ✅ Found{" "}
                      <strong>{result.verification.api.matches}</strong> similar
                      job(s) in the public index.
                    </p>
                    {result.verification.api.sample && (
                      <p className="small-text">
                        {result.verification.api.sample.title} at{" "}
                        {result.verification.api.sample.company}
                        {result.verification.api.sample.url && (
                          <>
                            {" "}
                            ·{" "}
                            <a
                              href={result.verification.api.sample.url}
                              target="_blank"
                              rel="noreferrer"
                            >
                              View sample job ↗
                            </a>
                          </>
                        )}
                      </p>
                    )}
                  </>
                ) : (
                  <p>❌ No matching results found for this title/company.</p>
                )}
              </div>

              {/* Email card */}
              <div className="info-card insight-card">
                <h4 data-icon="📧">Emails & Domains</h4>
                {result.verification.emails.length === 0 ? (
                  <p>No email address found in the job text.</p>
                ) : (
                  result.verification.emails.map((e, idx) => (
                    <p key={idx} className="small-text">
                      {e.signals && e.signals.length > 0 ? (
                        <>
                          ❗ <strong>{e.email}</strong> → {e.signals.join(", ")}
                        </>
                      ) : (
                        <>
                          ✅ <strong>{e.email}</strong> looks okay.
                        </>
                      )}
                    </p>
                  ))
                )}
              </div>

              {/* Keyword card */}
              <div className="info-card insight-card">
                <h4 data-icon="🚨">Risky Keywords</h4>
                {result.verification.kw_hits.length === 0 ? (
                  <p>✅ No known risky or scam-related phrases detected.</p>
                ) : (
                  <p>
                    ⚠️ Found{" "}
                    <strong>{result.verification.kw_hits.length}</strong> scam
                    phrases: {result.verification.kw_hits.slice(0, 6).join(", ")}
                  </p>
                )}
              </div>

              {/* Agentic verification card */}
              {result.verification.agentic && (
                <div className="info-card insight-card">
                  <h4 data-icon="🧭">Agentic Verification</h4>
                  <p className="small-text">
                    <strong>Investigator:</strong>{" "}
                    {result.verification.agentic.investigator?.official_domain
                      ? `Found likely official domain: ${result.verification.agentic.investigator.official_domain}`
                      : "Could not infer an official domain."}
                  </p>

                  {result.verification.agentic.auditor?.mismatches &&
                  result.verification.agentic.auditor.mismatches.length > 0 ? (
                    <p className="small-text">
                      ❗ Email/domain mismatch: {" "}
                      {result.verification.agentic.auditor.mismatches
                        .slice(0, 2)
                        .map((m) => `${m.email} vs ${m.expected_domain}`)
                        .join(" · ")}
                    </p>
                  ) : (
                    <p className="small-text">
                      ✅ No email/domain mismatches detected from available data.
                    </p>
                  )}

                  {result.verification.agentic.investigator?.top_results &&
                    result.verification.agentic.investigator.top_results.length > 0 && (
                      <p className="small-text">
                        Evidence: {" "}
                        <a
                          href={
                            result.verification.agentic.investigator.top_results[0].url
                          }
                          target="_blank"
                          rel="noreferrer"
                        >
                          {result.verification.agentic.investigator.top_results[0].title ||
                            "Top result"} ↗
                        </a>
                      </p>
                    )}
                </div>
              )}
            </div>

            {result.reasons && result.reasons.length > 0 && (
              <p className="reasons">
                <strong>Key factors:</strong> {result.reasons.join(" · ")}
              </p>
            )}
          </section>
        )}

        {/* How it works section */}
        <section className="how-it-works">
          <h2 className="section-title">How It Works</h2>
          <p className="section-sub">
            Cutting-edge AI technology to keep you safe from job scams.
          </p>
          <div className="info-grid">
            <div className="info-card">
              <h4 data-icon="🤖">AI-Powered Analysis</h4>
              <p>
                We clean the text and use TF-IDF plus a trained machine learning
                model to spot subtle red flags and suspicious patterns.
              </p>
            </div>
            <div className="info-card">
              <h4 data-icon="⚡">Instant Results</h4>
              <p>
                Paste a job post and get an immediate real vs fake confidence
                score in seconds — no signup required.
              </p>
            </div>
            <div className="info-card">
              <h4 data-icon="🛡️">Multi-Source Protection</h4>
              <p>
                We cross-check the job using the Adzuna public job index,
                email/domain safety checks, and known scam keyword detection.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}