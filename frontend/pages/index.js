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

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!title.trim() || !description.trim()) {
      setError("Please enter both Job Title and Job Description.");
      return;
    }

    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/analyze-job`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title,
          description,
          company,
          location
        })
      });

      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Something went wrong while analyzing the job.");
    } finally {
      setLoading(false);
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
                ? "This job looks FAKE (High Risk)"
                : "This job appears REAL (Low Risk)"}
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
                // Real percentage controls green/red split
                // CSS gradient is defined in globals.css
                "--realPct": `${result.real_pct}%`
              }}
            />

            <p className="model-caption">
              <strong>Model-only score (for transparency):</strong>{" "}
              Real {result.model_real_pct}% · Fake {result.model_fake_pct}%
            </p>

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
                      <strong>{result.verification.api.matches}</strong>{" "}
                      similar job(s) in the public index.
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
                  <p>No matching results found for this title/company.</p>
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
                  <p>No known risky or scam-related phrases detected.</p>
                ) : (
                  <p>
                    Found{" "}
                    <strong>{result.verification.kw_hits.length}</strong> scam
                    phrases:{" "}
                    {result.verification.kw_hits.slice(0, 6).join(", ")}
                  </p>
                )}
              </div>
            </div>

            {result.reasons && result.reasons.length > 0 && (
              <p className="reasons">
                <strong>Key factors:</strong>{" "}
                {result.reasons.join(" · ")}
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