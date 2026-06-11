# Project Audit Report: Smart Traffic ANPR System

## 1. Issue Log

| Severity | Location | Description | Impact | Recommended Fix |
| :--- | :--- | :--- | :--- | :--- |
| **Critical** | `smart-traffic-anpr/streamlit_app.py` / `core/tracker.py` | **State Leakage:** `VehicleTracker` is cached as a singleton but not reset between video runs. | Incorrect vehicle IDs and potential track data overlap across different videos. | Call `tracker.reset()` at the start of `process_video`. |
| **High** | `smart-traffic-anpr/database/db_manager.py` | **Resource Leaks:** Database connections are opened but rarely closed explicitly. | Potential exhaustion of file handles or connection pools, leading to crashes. | Use context managers for all DB connections; ensure `.close()` is called. |
| **High** | `smart-traffic-anpr/requirements.txt` | **Missing Dependencies:** `psycopg2` is missing despite code support for PostgreSQL. | Application will crash immediately if switched to PostgreSQL mode. | Add `psycopg2-binary` to `requirements.txt`. |
| **Medium** | `smart-traffic-anpr/streamlit_app.py` | **UI Bug:** `st.rerun()` executes before error messages can be read by the user. | Users are left unaware of why processing failed (e.g., file too large). | Use session state to persist error messages across reruns. |
| **Medium** | `smart-traffic-anpr/streamlit_app.py` | **Ghost Feature:** `FrameAnnotator` is initialized but never used to generate output. | The "Visual" part of the ANPR system is missing from the final UI. | Integrate `annotator.draw()` and save/display the resulting video. |
| **Medium** | `smart-traffic-anpr/streamlit_app.py` | **Temp File Accumulation:** Uploaded videos in `/tmp` are not reliably deleted. | Server disk space exhaustion over time. | Use `atexit` or a more robust cleanup utility for temp files. |
| **Low** | `smart-traffic-anpr/database/db_manager.py` | **Schema Mismatch:** Runtime DB creation ignores constraints in `schema.sql`. | Risk of invalid data (e.g., negative confidence) entering the DB. | Use `schema.sql` to initialize the database tables. |
| **Low** | `smart-traffic-anpr/config.py` | **Windows Pathing:** Hardcoded `temp` folder in project root on Windows. | Potential permission issues or accidental file exposure. | Use `tempfile.gettempdir()` for platform-agnostic temp paths. |

---

## 2. Architecture Review
The project follows a modular "Service-Oriented" layout within a single process.
*   **Strengths:** Clear separation between detection, tracking, and storage. Excellent use of Streamlit's caching for heavy models.
*   **Weaknesses:** The "Process Video" function is too monolithic (doing detection, tracking, OCR, and DB writes in one loop). This makes it hard to test components in isolation.
*   **Scalability:** Limited. It is designed for single-user Streamlit Cloud. Moving to a multi-user environment would require a message queue (Celery/Redis) to handle video processing asynchronously.

## 3. Security Audit
*   **Data Injection:** Low risk. SQL queries use parameterized inputs (e.g., `?` for SQLite).
*   **File Uploads:** Medium risk. While there's a size limit, there is no validation of the actual file content (magic bytes), only the extension. A user could upload a malicious file renamed to `.mp4`.
*   **Dependencies:** The versions in `requirements.txt` are slightly dated (e.g., `ultralytics 8.2.0`). Regular updates are needed to patch vulnerabilities in the underlying ML libraries.

## 4. Code Quality Report
*   **Readability:** High. Code is well-commented and follows PEP 8 standards.
*   **Error Handling:** Inconsistent. Some modules (like `anpr.py`) have robust try-except blocks, while `db_manager.py` assumes connections will always succeed.
*   **Testing:** **Critical Missing Feature.** There are no unit or integration tests in the repository.

## 5. Performance Analysis
*   **OCR Bottleneck:** EasyOCR on CPU is slow. Processing every 3rd frame is a good compromise, but the system still struggles with 1080p footage.
*   **Memory Management:** Good. The use of generators and frame skipping prevents the app from crashing on Streamlit Cloud's 1GB RAM limit.
*   **DB Locking:** The global threading lock in `db_manager.py` is safe but unnecessary for simple SQLite reads.

## 6. Missing Documentation Report
*   **API Docs:** No docstrings for several key methods in `db_manager.py`.
*   **Developer Guide:** `README.md` is great for users, but lacks info on how to add new vehicle classes or swap the OCR engine.
*   **Deployment:** No `Dockerfile` provided for users who want to deploy outside of Streamlit Cloud.

---

## 7. Overall Project Score: 6.5 / 10
*A solid MVP for a portfolio project, but lacks the "production-readiness" required for a commercial traffic system.*

---

## 8. Prioritized Action Plan

1.  **Immediate (Fix Stability):**
    *   Add `tracker.reset()` to the processing pipeline.
    *   Implement proper DB connection closing.
    *   Add `psycopg2-binary` to dependencies.
2.  **Short-Term (Improve UI/UX):**
    *   Fix the Error Message/Rerun bug.
    *   Enable video annotation so users can see the "AI in action".
3.  **Long-Term (Robustness):**
    *   Implement unit tests for `core/` modules.
    *   Add a `Dockerfile`.
    *   Sync `db_manager.py` with `schema.sql` constraints.
