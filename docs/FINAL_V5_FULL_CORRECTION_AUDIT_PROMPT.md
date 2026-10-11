# FINAL V5 DOCUMENTATION — FULL CORRECTION, DEDUPLICATION & DIAGRAM AUDIT PROMPT

## ROLE

Edit the existing `Disaster_Insights_Final_Project_Documentation_Final_v5.docx`.

Do NOT rewrite the report from scratch. Perform a complete quality-control pass and produce a corrected final Word document.

The current report is already over 100 pages. Preserve valid depth. Remove errors and duplication instead of adding filler.

The goal is:

**same genuine project + same evidence + same college structure + corrected diagrams + zero accidental repetition + no fabricated implementation.**

---

## 1. SOURCE OF TRUTH

Use this priority:

1. Actual implementation/source evidence.
2. Existing v5 report.
3. Official college report format.
4. General technical knowledge only for wording/formatting.

Never invent project facts.

If two statements conflict, verify against implementation evidence.

---

## 2. PROJECT STATUS — ABSOLUTE

Application: **Disaster Insights**

Current implemented hazard: **FLOOD ONLY**

Future/planned:
- Drought
- Wildfire

Drought and Wildfire have NOT been implemented, trained, tested, or deployed.

Actual technologies include:
- React
- Vite
- Leaflet / React-Leaflet
- Axios
- FastAPI
- Uvicorn
- Python
- PyTorch
- EfficientNet-B0
- Grad-CAM
- WorldFloods v2
- Rasterio
- NumPy
- Pillow
- Matplotlib
- scikit-learn
- Copernicus CDSE / Sentinel Hub
- Open-Meteo
- OpenStreetMap
- Nominatim

There is NO persistent database in the current implementation.

Do not present MySQL, PostgreSQL, MongoDB, Supabase, Firebase, authentication, user accounts, persistent history, or cloud deployment as implemented.

---

# 3. CRITICAL CONFIRMED PROBLEM: DUPLICATE EXPLANATIONS

The current v5 document contains accidental duplicate explanation blocks.

This is confirmed in the existing document.

For example, the Gantt Chart section has two separate explanations describing the same chart.

Chapter 5 also contains multiple duplicate explanations.

Most seriously, the generic Context Diagram explanation beginning:

> “The Context Diagram illustrates the system boundary and external entities interacting with Disaster Insights...”

has been copied underneath unrelated diagrams, including sections such as Collaboration, Component, and Deployment.

REMOVE every misplaced copy.

Do not merely reword it.

Each figure must have its own correct explanation.

---

# 4. ONE FIGURE = ONE EXPLANATION

For every figure:

- exactly ONE explanation immediately after it;
- explanation must describe THAT figure;
- no second “Explanation:” block;
- no generic copied explanation;
- no explanation belonging to another figure.

A figure may have one focused paragraph, or two short paragraphs only if genuinely necessary.

Never produce:

`Explanation: ...`

followed by another:

`Explanation: ...`

for the same figure.

---

# 5. COMPLETE DOCUMENT-WIDE REPETITION AUDIT

Audit the entire document, not only Chapter 5.

Check Chapters 1–10, Future Scope, Appendices, captions, tables, code explanations, and figure explanations.

Find and remove accidental repetition of:

- paragraphs
- explanations
- conclusions
- definitions
- hardware constraints
- Sentinel-2 descriptions
- Grad-CAM descriptions
- EfficientNet descriptions
- dataset descriptions
- architecture descriptions
- training descriptions
- future-scope descriptions
- testing descriptions
- result interpretations
- code explanations

A fact can legitimately appear in multiple chapters when the purpose is different. The same paragraph must not be copied into multiple locations.

---

# 6. CHAPTER 3 — PROJECT PLANNING

Remove the duplicate Gantt explanation.

Keep ONE accurate Gantt explanation.

Keep ONE accurate Activity Network explanation.

Do not invent exact development dates. The report states that exact dates were not formally recorded and durations are estimates.

Do not describe a critical path that the actual diagram does not show.

---

# 7. CHAPTER 5 — FULL DIAGRAM AUDIT

The correct sequence is:

5.1 Context Diagram  
5.2 Level-0 Data Flow Diagram  
5.3 Level-1 DFD — Flood Assessment Process  
5.4 Conceptual Data Model  
5.5 Class Diagram  
5.6 State Transition Diagram  
5.7 Use Case Diagram  
5.8 System Architecture Diagram  
5.9 Activity Diagram — Flood Assessment  
5.10 Component Diagram  
5.11 Collaboration Diagram  
5.12 Deployment Diagram

Every heading, figure number, diagram, explanation, TOC entry, and List of Figures entry must agree.

---

# 8. FIGURE 5.1 — CONTEXT DIAGRAM

Represent Disaster Insights as one central system/process.

External entities/services:

- User
- Copernicus CDSE / Sentinel Hub
- Open-Meteo
- OpenStreetMap / Nominatim
- WorldFloods v2 as training/demo data source where appropriate

WorldFloods v2 is NOT a live API.

Do not show React, FastAPI, EfficientNet-B0, Grad-CAM, FloodInference, etc. as external entities in the context diagram.

Keep it genuinely high-level.

All data-flow arrows must point to the correct destination and have correct labels.

---

# 9. FIGURE 5.2 — LEVEL-0 DFD

Fix the current inconsistency.

Do NOT show the system simultaneously as a single Process 0 and as five internal processes.

Use a consistent Level-0 decomposition based on the actual system, such as:

1.0 Location / Request Processing  
2.0 Satellite Data Retrieval  
3.0 Flood Assessment Engine  
4.0 Result Assembly / Response  
5.0 Weather and AQI Retrieval

Use relevant external entities and labelled data flows.

The explanation must describe exactly what the final diagram shows.

---

# 10. FIGURE 5.3 — LEVEL-1 DFD

Decompose the relevant Flood Assessment process consistently.

Use actual steps such as:

3.1 Load Image  
3.2 Preprocess Image  
3.3 EfficientNet-B0 Inference  
3.4 Softmax Probability  
3.5 Risk Interpretation  
3.6 Grad-CAM Generation  
3.7 Result Preparation

Do not describe one decomposition and draw another.

---

# 11. FIGURE 5.4 — CONCEPTUAL DATA MODEL

The project has NO persistent database.

Therefore do not make this look like an implemented SQL ERD.

Clearly label it:

**Conceptual Data Model — No Persistent Database Implemented**

Logical/in-memory entities may include:

- Location
- Assessment
- Satellite Data
- Prediction
- Weather Data
- AQI Data
- Grad-CAM Result

Do not create fake SQL tables, foreign keys, database schemas, or database technologies.

The explanation must explicitly state that the model represents logical/in-memory data used during an assessment request.

---

# 12. FIGURE 5.5 — CLASS DIAGRAM

Use actual implementation classes.

Important actual classes include:

- FloodClassifier
- FloodInference
- GradCAM
- WorldFloodsDataset
- SentinelDataService

Verify every class against the implementation.

The current v5 text incorrectly discusses items such as FloodRoutes and AppState as classes. If implementation evidence shows these are modules/components rather than classes, remove that claim.

Do not invent DatabaseManager, UserManager, AuthenticationManager, etc.

The explanation must describe the classes actually shown.

---

# 13. FIGURE 5.6 — STATE TRANSITION DIAGRAM

Use actual application states:

- IDLE / Welcome
- LOCATION_SELECTED
- LOADING
- RESULTS_DISPLAYED
- ERROR_STATE

Show realistic transitions such as:

location selected  
assessment started  
loading completed  
result displayed  
request failed  
try again  
new location

Do not introduce login, database, or account states.

---

# 14. FIGURE 5.7 — USE CASE DIAGRAM

Use proper UML notation.

Actor:
- User

External systems:
- Copernicus CDSE / Sentinel Hub
- Open-Meteo

Actual functions:

- Select Location
- Run Flood Assessment
- View Flood Probability
- View Flood Risk
- View Satellite Image
- View Grad-CAM Heatmap
- View Weather Data
- View AQI Data
- Fetch Satellite Image
- Fetch Weather / AQI Data

Do not add login, registration, user history, admin panel, database management, etc.

Do not invent unnecessary `include`/`extend` relationships.

---

# 15. FIGURE 5.8 — SYSTEM ARCHITECTURE

Show the actual layers.

Frontend:
- React
- Vite
- Leaflet / React-Leaflet
- Assessment Dashboard
- Weather/AQI display

Backend:
- FastAPI
- Uvicorn
- Flood routes
- Sentinel service
- Inference engine

ML:
- PyTorch
- EfficientNet-B0
- Flood Classifier
- Flood Inference
- Grad-CAM

External:
- Copernicus CDSE / Sentinel Hub
- Open-Meteo
- OpenStreetMap / Nominatim

IMPORTANT IMPLEMENTATION DETAIL:

Weather and AQI are fetched directly by the React frontend from Open-Meteo.

Do not incorrectly route all weather/AQI traffic through FastAPI.

---

# 16. FIGURE 5.9 — ACTIVITY DIAGRAM

Use proper UML Activity notation.

Actual flow:

Start  
→ Open Disaster Insights  
→ Select Location  
→ Choose Demo or Live

Demo:
→ Select archived/demo WorldFloods test image  
→ preprocessing

Live:
→ Copernicus authentication  
→ bounding box  
→ Sentinel-2 request  
→ GeoTIFF response  
→ validation  
→ preprocessing

Then both paths converge:

→ Extract RGB  
→ Normalize  
→ Resize 224×224  
→ ImageNet normalization  
→ EfficientNet-B0 inference  
→ Softmax probability  
→ Risk interpretation  
→ Grad-CAM  
→ Prepare response  
→ Display results  
→ End

---

# 17. FIGURE 5.10 — COMPONENT DIAGRAM

Use actual project files/components:

Frontend:
- App.jsx
- index.css
- main.jsx
- Axios
- React-Leaflet

Backend:
- main.py
- flood_routes.py
- sentinel_service.py

ML:
- flood_classifier.py
- flood_dataset.py
- train_flood.py
- flood_inference.py
- gradcam.py

Do not add unsupported files/components.

---

# 18. FIGURE 5.11 — COLLABORATION DIAGRAM

Use a proper UML Communication/Collaboration Diagram.

A realistic sequence is:

1. User → React App: Run Assessment
2. React App → FastAPI: assessment request
3. FastAPI → Flood Routes: route request
4. Flood Routes → Satellite Service: live image request when Live mode
5. Flood Routes → Flood Inference: inference request
6. Flood Inference → Flood Classifier: model prediction
7. Flood Inference → GradCAM: explanation generation
8. Flood Routes → React App: JSON result
9. React App → User: display result

Only use method names actually supported by source evidence.

---

# 19. FIGURE 5.12 — DEPLOYMENT DIAGRAM

Show the actual local environment:

Local Developer Machine

Browser:
- Chrome / supported browser

Frontend:
- Node.js / Vite
- Port 5173

Backend:
- Python virtual environment
- FastAPI / Uvicorn
- Port 8000

ML:
- PyTorch
- CUDA GPU
- EfficientNet-B0
- best_flood_model.pt

Local data:
- WorldFloods v2 subset

External services:
- Copernicus CDSE / Sentinel Hub
- Open-Meteo
- OpenStreetMap / Nominatim

Do not show AWS, Azure, GCP, Kubernetes, Docker, or cloud deployment as current deployment.

---

# 20. DIAGRAM QUALITY

Regenerate/replace any incorrect diagram.

All diagrams must be:

- professional
- academic
- black/white or grayscale
- high resolution
- Word-ready
- properly aligned
- readable
- logically connected

No:

- ASCII
- Mermaid
- decorative infographic clutter
- random colours
- overlapping arrows
- overlapping labels
- clipped elements
- giant empty boxes
- unreadable text

Use the correct notation for each diagram type.

---

# 21. DIAGRAM-TO-TEXT CONSISTENCY

For every figure verify:

**Heading = Figure Number = Diagram = Explanation = Implementation**

If any differs, fix it.

Never allow a Context Diagram explanation to appear below a Component, Collaboration, Activity, Class, or Deployment Diagram.

---

# 22. CHAPTER 6 AUDIT

Check every module against implementation evidence.

Implemented modules include:

- Dataset and Data Pipeline
- Flood Classification Model
- Training Pipeline
- Evaluation
- Grad-CAM Explainability
- Inference Engine
- FastAPI Backend
- Satellite Service
- React Frontend
- Map and Location Service

Do not invent additional implemented modules.

Keep source-file names accurate.

---

# 23. HARDWARE/SOFTWARE CONSISTENCY

Use the actual documented environment:

- AMD Ryzen laptop processor
- 16 GB RAM
- NVIDIA RTX 4050 Laptop GPU
- 6 GB VRAM
- Windows 11
- approximately 32 GB dataset storage
- approximately 44 MB best model checkpoint

Do not silently replace verified project versions with random versions.

---

# 24. RESULTS — NO FABRICATION

Use the actual recorded evaluation:

- Accuracy: 0.9815
- Precision: 0.9811
- Recall: 1.0
- F1: 0.9905
- Specificity: 0.5
- TP = 52
- TN = 1
- FP = 1
- FN = 0
- Test samples = 54
- Flood = 52
- Non-flood = 2

Keep these consistent throughout the report.

Do not improve or alter the metrics.

Preserve the limitation that the test set is small and heavily imbalanced.

---

# 25. CLAIM STRENGTH

Remove unsupported wording such as:

- proves
- guarantees
- perfect
- flawless
- best
- optimal
- revolutionary
- real-time

The application is NOT an official real-time flood warning system.

Prefer:

- flood assessment
- model prediction
- satellite-based assessment
- latest available imagery
- prototype
- demonstration system

---

# 26. FUTURE SCOPE

Clearly separate CURRENT from FUTURE.

Future items can include:

- Drought model
- Wildfire model
- Sentinel-1 SAR
- larger balanced dataset
- semantic segmentation
- temporal change detection
- persistent database/history
- cloud deployment

Every such item must be clearly marked as future/planned.

---

# 27. APPENDICES / CODE

For every code snippet:

- correct file name
- genuine project code only
- explanation directly tied to that snippet
- no duplicated explanation
- no invented code

---

# 28. TOC / LIST OF FIGURES / LIST OF TABLES

After editing, update:

- Table of Contents
- List of Figures
- List of Tables
- page numbers
- figure numbering
- table numbering

They must match the final document exactly.

---

# 29. COLLEGE STRUCTURE

Do not create a second competing structure.

Preserve:

1. Introduction
2. Requirement Analysis
3. Project Planning
4. Software / Hardware Modules
5. System Analysis and Design
6. System Implementation and Methodology
7. System Testing
8. Results and Discussion
9. Cost and Benefit Analysis and Software Parameter Estimation
10. Conclusion and Future Scope

---

# 30. TABLE DESIGN

Because there is no persistent database:

State clearly:

**No persistent database tables are implemented in the current version.**

Future SQLite/PostgreSQL storage may be mentioned only as future work.

Do not create fake implemented tables.

---

# 31. FORMATTING

Preserve:

- Times New Roman
- black headings
- black body text
- justified body text
- professional spacing
- page numbers
- clean captions
- clean tables
- no decorative colours

---

# 32. DO NOT SHRINK THE REPORT

The existing report is already over 100 pages.

Do not delete valid technical depth.

Remove only:

- accidental duplicates
- misplaced explanations
- contradictions
- fabricated content
- irrelevant filler
- redundant paragraphs

Do not add repetition to preserve page count.

---

# 33. DO NOT ADD GENERIC FILLER

If a section is short, do NOT automatically add generic textbook material.

Useful additions are only project-specific:

- implementation decisions
- preprocessing details
- API workflows
- actual module interaction
- actual testing
- hardware constraints
- observed implementation issues
- design rationale

Do not add generic AI/ML history or repeat technology definitions.

---

# 34. FINAL PAGE-BY-PAGE QA

Before exporting the final file, inspect the complete document.

Check for:

- duplicate explanations
- duplicate paragraphs
- wrong explanations under figures
- blank pages
- broken diagrams
- clipped diagrams
- captions separated from figures
- overlapping text
- wrong figure numbers
- wrong page numbers
- inconsistent headings
- placeholder text
- malformed tables
- inconsistent formatting

---

# 35. FINAL OUTPUT

Create:

`Disaster_Insights_Final_Project_Documentation_Final_v6.docx`

Do NOT append corrections to the end.

EDIT THE EXISTING CONTENT IN PLACE.

If duplicate:
→ remove duplicate.

If misplaced:
→ remove from wrong section and keep/create correct version.

If diagram wrong:
→ replace diagram.

If figure number wrong:
→ correct it everywhere.

If contradiction:
→ reconcile using implementation evidence.

If valid:
→ preserve it.

The final document must look like ONE carefully reviewed academic report, not multiple generated drafts pasted together.

Before finalizing, perform one final full-document audit for duplication, consistency, diagram correctness, numbering, formatting, and implementation truth.
