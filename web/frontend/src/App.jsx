import { useState } from "react";

const defaultResult = null;
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "";
const progressStages = [
  { value: 12, label: "Preparing upload" },
  { value: 34, label: "Sending files to the server" },
  { value: 61, label: "Extracting audio features" },
  { value: 84, label: "Running ensemble prediction" }
];

// Formats backend processing duration for display.
function formatSeconds(value) {
  return `${value.toFixed(1)}s`;
}

// Renders the upload form, prediction request flow, and result summary.
function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [transcriptFile, setTranscriptFile] = useState(null);
  const [threshold, setThreshold] = useState(0.53);
  const [result, setResult] = useState(defaultResult);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [progressValue, setProgressValue] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");

  // Sends selected files and threshold to the Flask prediction endpoint.
  async function handleSubmit(event) {
    event.preventDefault();

    if (!audioFile) {
      setError("Please upload a WAV audio file before running the prediction.");
      return;
    }

    setIsSubmitting(true);
    setProgressValue(0);
    setProgressLabel("Starting analysis");
    setError("");
    setResult(defaultResult);

    try {
      let stageIndex = 0;
      setProgressValue(progressStages[0].value);
      setProgressLabel(progressStages[0].label);

      const progressTimer = window.setInterval(() => {
        stageIndex += 1;
        if (stageIndex >= progressStages.length) {
          window.clearInterval(progressTimer);
          return;
        }
        setProgressValue(progressStages[stageIndex].value);
        setProgressLabel(progressStages[stageIndex].label);
      }, 850);

      const formData = new FormData();
      formData.append("audio", audioFile);
      formData.append("threshold", threshold.toFixed(2));

      if (transcriptFile) {
        formData.append("transcript", transcriptFile);
      }

      const response = await fetch(`${apiBaseUrl}/api/predict`, {
        method: "POST",
        body: formData
      });

      const payload = await response.json();
      if (!response.ok) {
        window.clearInterval(progressTimer);
        throw new Error(payload.error || "Prediction failed.");
      }

      window.clearInterval(progressTimer);
      setProgressValue(100);
      setProgressLabel("Analysis complete");
      setResult(payload);
    } catch (requestError) {
      setProgressValue(0);
      setProgressLabel("");
      setError(requestError.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  const resultTone = result?.prediction ? "alert" : "calm";

  return (
    <div className="page-shell">
      <div className="page-accent page-accent-left" />
      <div className="page-accent page-accent-right" />

      <main className="page-content">
        <section className="hero-card">
          <h1>Depression Detection</h1>
        </section>

        <section className="workspace-grid">
          <form className="panel upload-panel" onSubmit={handleSubmit}>
            <div className="panel-header">
              <h2>Submit Recording</h2>
            </div>

            <label className="field">
              <span>Interview Audio (.wav)</span>
              <input
                type="file"
                accept=".wav,audio/wav"
                onChange={(event) => setAudioFile(event.target.files?.[0] ?? null)}
              />
            </label>

            <label className="field">
              <span>Transcript CSV (optional)</span>
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => setTranscriptFile(event.target.files?.[0] ?? null)}
              />
            </label>

            <label className="field">
              <span>Decision Threshold</span>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.01"
                value={threshold}
                onChange={(event) => setThreshold(Number(event.target.value))}
              />
              <div className="threshold-row">
                <span>Sensitive</span>
                <strong>{threshold.toFixed(2)}</strong>
                <span>Conservative</span>
              </div>
            </label>

            <button className="primary-button" type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Analyzing..." : "Run Prediction"}
            </button>

            {isSubmitting ? (
              <div className="progress-block" aria-live="polite">
                <div className="progress-copy">
                  <span>Progress</span>
                  <strong>{progressLabel}</strong>
                </div>
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{ width: `${progressValue}%` }}
                  />
                </div>
              </div>
            ) : null}

            <p className="helper-text">
              This site is intended for research demonstrations only and should not be
              used as a medical diagnostic tool.
            </p>
          </form>

          <section className="panel results-panel">
            <div className="panel-header">
              <h2>Result Summary</h2>
            </div>

            {error ? <div className="message error-message">{error}</div> : null}

            {!result && !error ? (
              <div className="empty-state">
                <p className="empty-title">Ready for evaluation</p>
                <p>
                  Upload a WAV file, optionally attach the transcript, and the system will
                  return a prediction score, confidence band, and processing metadata.
                </p>
              </div>
            ) : null}

            {result ? (
              <div className={`result-card ${resultTone}`}>
                <div className="result-banner">
                  <span className="result-label">Outcome</span>
                  <h3>{result.label}</h3>
                  {result.prediction ? (
                    <p className="result-note">
                      A small reminder: one difficult result does not define you, and support
                      is always a strong next step.
                    </p>
                  ) : (
                    <p className="result-note">
                      This screening result looks reassuring, but checking in with yourself still matters.
                    </p>
                  )}
                </div>

                <div className="metric-grid">
                  <article className="metric-card">
                    <span>Prediction Score</span>
                    <strong>{result.probability.toFixed(3)}</strong>
                  </article>
                </div>
              </div>
            ) : null}
          </section>
        </section>
      </main>
    </div>
  );
}

export default App;
