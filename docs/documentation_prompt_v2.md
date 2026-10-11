# DOCUMENTATION PROMPT — DISASTER INSIGHTS
## FINAL YEAR PROJECT REPORT REBUILD / CORRECTION PROMPT

You are responsible for rebuilding the final project documentation for:

Project Title:
Deep Learning-Based Early Detection and Classification of Climate-Induced Natural Disasters Using Satellite Imagery

Application Name:
Disaster Insights

Degree:
B.Sc. Artificial Intelligence and Machine Learning

IMPORTANT:
The current implementation contains ONLY FLOOD detection/assessment.
Drought and Wildfire are FUTURE EXTENSIONS ONLY.
Never write as if drought or wildfire has already been implemented.

The previously generated documentation is NOT acceptable in its current form. Rebuild the Word document properly instead of merely editing a few sentences.

The current generated document is approximately 74 pages including preliminary pages, but the actual useful report is much shorter. The final report MUST be substantially expanded with genuine project-specific technical explanation and evidence. Do not inflate the page count using blank pages, oversized spacing, repeated text, or meaningless content.

The final deliverable MUST be a .DOCX Word document only.

============================================================
1. SOURCE MATERIAL — READ EVERYTHING FIRST
============================================================

Before writing anything, inspect all available project files.

You MUST use these as the primary sources:

1. The college project-report-format PDF.
2. The reference/final project report supplied with the project.
3. The current generated Disaster Insights documentation DOCX.
4. The complete project implementation directory.
5. The image/evidence directory that will be provided separately.
6. Actual source code, configuration files, model files, logs, reports, screenshots and outputs available in the project directory.

Do not rely only on the old documentation.

The implementation files and evidence are the highest authority for what was actually built.

If the report and implementation disagree:
- prefer the actual implementation;
- verify the implementation before writing;
- do not silently invent a correction;
- never present a planned feature as implemented.

Do not fabricate:
- experiments;
- dates;
- hardware;
- software versions;
- dataset counts;
- training results;
- performance;
- APIs;
- screenshots;
- code;
- database tables;
- project-management usage;
- deployment;
- features.

If something cannot be verified, clearly state that it is not verified or leave it out.

============================================================
2. COLLEGE FORMAT IS MANDATORY
============================================================

Follow the college-provided format as the structural authority.

Use the required chapter structure:

1. Introduction
   A. Background and Motivation of the Project
   B. Objective and Scope of the Project
   C. Problem Definition
   D. Limitations

2. Requirement Analysis
   A. Users Requirements / SRS / Functional Requirements
   B. Feasibility Study
   C. Details of Hardware and Software Used
   D. Proposed System and Advantages of Proposed System

3. Project Planning
   A. List of Activity / Task
   B. Timeline for Each Task
   C. Gantt Chart, Activity Network Diagram
   D. Resource Planning
   E. Proposed Software Development Model for the Project
   F. Using Project Management Tool Jira

4. Software / Hardware Modules
   Description of Module
   Individual Work Justification

5. System Analysis and Design
   A. Context Diagram
   B. DFD, ERD, Class Diagram, State Transition Diagram
   C. Use Case Diagram
   D. Architecture Design
   E. Activity Diagram
   F. Component Diagram
   G. Collaboration Diagram
   H. Table Design
   I. Architecture / Deployment Diagram

6. System Implementation and Methodology
   A. Development Methodology
   B. Technologies Used
   C. Algorithms / Techniques Used
   D. Module-wise Implementation
   E. Database Implementation
   F. User Interface Design
   G. Important Code Snippets

7. System Testing
   A. Testing Strategy
   B. Testable Unit
   C. Test Case Design with Reference to the Project
   D. Test Approach
   E. Test Result and Validation

8. Results and Discussion
   A. System Output
   B. Result Analysis
   C. Performance Evaluation

9. Cost and Benefit Analysis and Software Parameter Estimation
   A. Effort Estimation
   B. KSLOC Analysis
   C. FP Analysis

10. Conclusion and Future Scope
   A. Achievements of the Project
   B. Conclusion

Future Scope should be included inside Chapter 10 rather than creating random extra chapters.

Do NOT create a completely different chapter structure merely because the reference report uses another structure.

============================================================
3. NUMBERING — STRICTLY FIX THIS
============================================================

This is a major correction.

Do NOT use:
- 1.a
- 1.b
- 1.c
- 1.d
- 2.a
- 2.b
- 3.a
- 3.b
- or any other chapter-number + letter combination.

Use simple numbering exactly like:

Chapter 1 — Introduction

A. Background and Motivation of the Project
B. Objective and Scope of the Project
C. Problem Definition
D. Limitations

Chapter 2 — Requirement Analysis

A. Users Requirements / SRS / Functional Requirements
B. Feasibility Study
C. Details of Hardware and Software Used
D. Proposed System and Advantages of Proposed System

For numbered lists inside sections, ALWAYS restart numbering when a new list begins.

Example:

Motivation:
1. Increasing need for faster flood assessment.
2. Availability of satellite imagery.
3. Need for accessible tools.
4. Need for visual explanation.
5. Practical value of the application.

Objectives:
1. Train the flood classification model.
2. Implement Grad-CAM.
3. Develop the backend.
4. Develop the frontend.
5. Integrate satellite imagery.

The Objectives list MUST NOT continue from number 6.

Limitations:
1. Dataset limitations.
2. Cloud-cover limitations.
3. Image-level classification limitation.
4. Local deployment limitation.

Every independent numbered list starts from 1.

Do not use numbering inherited accidentally from previous sections.

============================================================
4. NO UNNECESSARY PRELIMINARY LISTS
============================================================

REMOVE these sections completely:

- List of Figures
- List of Tables
- List of Abbreviations

Do NOT create a separate page for them.

The report should contain:

Cover Page
Certificate
Declaration
Acknowledgement
Abstract
Table of Contents
Chapter 1 onward

Do not add unnecessary preliminary pages.

The user specifically does NOT want:
- list of figures;
- list of tables;
- list of abbreviations;
- references section;
- citation-heavy academic formatting.

Do not create a References chapter or References section at the end unless the college format absolutely requires it.

Do not add citation markers throughout the report.

Do not insert [1], [2], [3], (Author, Year), footnotes, endnotes, or reference numbers.

The project should be explained from the actual implementation and supplied project material.

If a third-party fact is genuinely necessary, explain it naturally without filling the document with citation markers.

============================================================
5. PAGE COUNT — THIS IS A HARD REQUIREMENT
============================================================

The final Word report MUST be at least 100 pages.

Target approximately 105–120 pages.

The page count must be REAL.

Do NOT reach 100 pages through:
- blank pages;
- huge empty spaces;
- unnecessary page breaks;
- oversized headings;
- oversized images;
- excessive paragraph spacing;
- repeated paragraphs;
- repeated tables;
- generic textbook material;
- fake appendices;
- repeated code screenshots;
- duplicate screenshots.

The additional pages must come from useful project documentation.

The final report should be detailed enough that an examiner can understand:
- what the project does;
- why it was built;
- what was actually implemented;
- how the data was prepared;
- how the model works;
- how training was performed;
- how inference works;
- how the backend works;
- how the frontend works;
- how satellite retrieval works;
- how Grad-CAM works;
- how the modules communicate;
- how the system was tested;
- what outputs were obtained;
- what the results mean;
- what the hardware constraints were;
- what the actual project cost was;
- what the limitations are;
- how drought and wildfire could be implemented later.

Every page must contain meaningful content.

============================================================
6. WRITING STYLE
============================================================

The writing must be:

- formal;
- natural;
- simple;
- student-level;
- technically correct;
- easy to understand;
- directly related to this project.

Imagine explaining the project to a technically interested 10-year-old first, then write it in formal academic language.

Avoid unnecessarily complicated words.

Do not use artificial phrases such as:
- "In today's rapidly evolving technological landscape"
- "cutting-edge"
- "revolutionary"
- "seamless"
- "robust and scalable solution"
- "leverages state-of-the-art"
- "plays a pivotal role"
- "it is worth noting"
- "furthermore, it is important to highlight"
- "in conclusion, it can be said"

Do not write generic AI-generated textbook paragraphs.

Do not intentionally insert mistakes to fool AI detectors.

The goal is genuinely original, project-specific writing based on the actual implementation.

Do not use a paraphrasing/humanizer strategy.

============================================================
7. FORMATTING — STRICT BLACK AND WHITE
============================================================

The entire Word document MUST use:

Font:
Times New Roman

Font sizes:
- Main chapter headings: 16 pt
- Section/subsection headings: 14 pt
- Normal body text: 12 pt

Everything MUST be pure black.

This includes:
- chapter headings;
- section headings;
- subsection headings;
- normal text;
- table text;
- captions;
- page numbers;
- diagrams;
- borders;
- labels.

NO coloured headings.

NO coloured text.

NO blue hyperlinks.

NO coloured table headers.

NO grey table headers.

NO blue backgrounds.

NO orange backgrounds.

NO green backgrounds.

NO decorative colour.

Tables must be simple black-and-white:
- white background;
- black text;
- black borders;
- no shading;
- no coloured cells;
- no gradients.

Diagrams must also be black and white unless a colour is absolutely required to represent actual application output.

Actual application screenshots are allowed to retain their original colours because they are evidence of the implemented UI/output. Do not recolour actual screenshots.

Body text must be JUSTIFIED.

Use clean academic spacing.

Do not use excessive paragraph spacing.

Do not use giant headings that waste page space.

============================================================
8. PAGE MANAGEMENT — NO BLANK PAGES
============================================================

There must be NO accidental blank pages.

Before finalising the document:
- inspect every page;
- remove unnecessary page breaks;
- prevent orphaned headings;
- prevent headings from appearing alone at the bottom of a page;
- keep captions with their images;
- keep tables with their headings;
- avoid large unused areas;
- ensure page numbers appear correctly;
- ensure content flows continuously.

Do not insert manual page breaks after every small subsection.

Only start a new page when academically appropriate, such as:
- cover;
- certificate;
- declaration;
- acknowledgement;
- abstract;
- Table of Contents;
- new chapter.

Do not create a separate page for a two-line subsection merely to increase page count.

============================================================
9. TABLE OF CONTENTS
============================================================

Include ONLY a normal Table of Contents.

Do not include:
- List of Figures;
- List of Tables;
- List of Abbreviations.

The Table of Contents must use the final actual headings and actual page numbers.

Do NOT leave placeholder text such as:
"[INSERT ACTUAL PAGE NUMBERS]"

The final document must be updated after all content and images are inserted.

============================================================
10. TABLES — SIMPLE BLACK AND WHITE
============================================================

Tables are required where they genuinely improve understanding.

However:

Every table MUST have:
- simple black border;
- white background;
- black text;
- Times New Roman;
- no coloured cells.

Examples of useful tables:

Hardware Requirements
Software Requirements
Functional Requirements
Non-Functional Requirements
Dataset Structure
Dataset Split
Training Hyperparameters
Model Configuration
API Endpoints
Module Summary
Test Cases
Testing Results
Cost Analysis
Effort Estimation
KSLOC Analysis
Function Point Analysis
Actual vs Future Features

Do not create decorative tables just to increase page count.

============================================================
11. IMAGES — MAJOR CORRECTION REQUIRED
============================================================

The previous document placed images badly.

Do NOT repeat that.

Every image must be placed immediately after the paragraph that introduces or explains it.

Never randomly place an image several pages away from the explanation.

Every image must have:

Figure X.X — Clear Figure Title

Immediately below the image, add a detailed explanation.

Example:

Figure 8.2 — Flood Assessment Result Dashboard

Explanation:
This screen shows the result produced after the user selects a location and runs the flood assessment. The application displays the selected location, the predicted flood class, the flood probability, the interpreted risk level, the satellite image and the Grad-CAM explanation. The result is returned by the FastAPI backend and rendered by the React frontend.

Every screenshot must be explained.

Do NOT place an image without explanation.

Do NOT use:
"Figure shown above."
"Output shown below."
"This is the result."

The explanation must tell the examiner:
1. What is visible.
2. What the user did before reaching this screen.
3. What the system did.
4. What the important elements mean.
5. Why the output matters.
6. How the output connects to the implementation.

============================================================
12. CODE SNIPPET IMAGES — MAJOR CORRECTION REQUIRED
============================================================

The provided image folder contains code snippets and output evidence.

These must be included where relevant.

Do NOT simply paste random code images.

For every code screenshot:

Figure X.X — Code Snippet: [Specific Purpose]

Then provide:

Purpose:
Explain what this code does.

Important lines:
Explain the important parts in simple language.

Connection to system:
Explain which module uses this code and what happens after it runs.

Why it is required:
Explain why the code exists in the project.

Expected/actual result:
Explain what the code produces.

The code screenshot must be:
- readable;
- large enough;
- not stretched;
- not cropped incorrectly;
- not blurry;
- not squeezed into a tiny space.

If the source image is too small to read:
- use the highest-resolution original available;
- crop only unnecessary empty areas;
- preserve the code;
- if necessary, recreate the code as a clean black-and-white code block from the verified source file rather than using an unreadable screenshot.

Never invent code.

Use the actual source code as the authority.

============================================================
13. OUTPUT SCREENSHOTS — MAJOR CORRECTION REQUIRED
============================================================

Every actual UI/output screenshot provided in the evidence folder must be reviewed.

Use the screenshots where they support the documentation.

Examples:
- Welcome screen
- Map/location selection
- Demo mode
- Live satellite assessment
- Flood result
- Risk display
- Grad-CAM
- Weather
- AQI
- Error state
- Training output
- Evaluation output
- Confusion matrix
- Training curves
- Terminal/backend output
- API response where useful

For every important output screenshot:

1. Give it a figure number.
2. Give it a proper title.
3. Place it near the relevant explanation.
4. Explain the complete screen.
5. Explain what action produced it.
6. Explain the technical flow behind it.
7. Explain what the examiner should observe.

Do not use screenshots just as decoration.

============================================================
14. EVIDENCE DIRECTORY — MANDATORY
============================================================

The image/evidence directory supplied with this prompt contains important implementation evidence.

You MUST inspect it completely.

Create an internal inventory:

- filename;
- type;
- whether it is code/output/diagram/UI;
- which module it belongs to;
- where it belongs in the report.

Do not ignore images because there are many.

Do not insert an image simply because it exists.

Each image must have a reason for inclusion.

If an image clearly belongs to a particular chapter, place it there.

Do not place all screenshots at the end.

============================================================
15. DIAGRAMS
============================================================

Generate or draw diagrams ONLY when they are genuinely required by the college format or needed to explain the system.

Required diagrams should include, where applicable to the actual implementation:

- Context Diagram
- Level-0 DFD
- Level-1 DFD
- Conceptual data model / ER-style representation, clearly labelled as conceptual because there is no implemented persistent database
- Class Diagram
- State Transition Diagram
- Use Case Diagram
- System Architecture Diagram
- Activity Diagram
- Component Diagram
- Collaboration Diagram if required by the college format
- Deployment Diagram
- ML training workflow
- ML inference workflow
- Satellite retrieval workflow
- Grad-CAM workflow
- Frontend-backend communication workflow
- Future multi-hazard architecture for flood + planned drought + planned wildfire

All newly generated academic diagrams should be:
- clean;
- black and white;
- readable;
- technically consistent;
- correctly labelled.

Do not generate decorative diagrams.

Do not generate a database schema as if a database exists.

The current implementation does NOT have a persistent database unless the actual project files prove otherwise.

============================================================
16. FLOOD ONLY — DO NOT MISREPRESENT FUTURE FEATURES
============================================================

The current implemented system is FLOOD ONLY.

Implemented functionality must be described as implemented only when verified.

Drought:
NOT IMPLEMENTED.

Wildfire:
NOT IMPLEMENTED.

In Chapter 10, explain how drought and wildfire could be added in the future.

Future Drought implementation can discuss:
- appropriate drought datasets;
- vegetation/water-related satellite indicators;
- suitable classification or severity approach;
- required preprocessing changes;
- additional model/data requirements;
- how a Drought module would fit into the current architecture.

Future Wildfire implementation can discuss:
- wildfire datasets;
- burned-area/fire-related imagery;
- suitable spectral information;
- classification/severity approach;
- preprocessing changes;
- how a Wildfire module would fit into the architecture.

Clearly label all of these as FUTURE WORK.

Never claim they were trained, tested, deployed, or demonstrated.

============================================================
17. HARDWARE — DOCUMENT ACTUAL CONSTRAINTS
============================================================

Document the actual hardware used and its constraints.

The current documentation reports:
- AMD Ryzen laptop processor;
- 16 GB RAM;
- NVIDIA RTX 4050 Laptop GPU;
- 6 GB VRAM;
- Windows 11;
- approximately 172 GB free disk space on D: at the documented point;
- approximately 32 GB dataset subset usage;
- approximately 44 MB trained checkpoint.

VERIFY THESE AGAINST THE ACTUAL PROJECT EVIDENCE BEFORE FINALISING.

Explain:
- why GPU was needed;
- VRAM limitation;
- RAM requirement;
- storage requirement;
- dataset storage;
- model storage;
- CPU fallback;
- internet requirement;
- satellite API dependency;
- weather/AQI API dependency;
- limitations caused by the available hardware.

Do not state hardware values as facts if the implementation evidence contradicts them.

============================================================
18. COST ANALYSIS
============================================================

Include actual and estimated costs honestly.

Separate:
- existing hardware cost;
- software cost;
- dataset cost;
- API cost;
- hosting/deployment cost;
- internet cost where appropriate;
- development effort.

If a cost was zero because a free/open-source service was used, state why.

Do not invent purchase prices.

If the hardware was already owned and its purchase price is unknown, say so.

Do not claim the project cost a specific amount unless supported.

============================================================
19. DATASET
============================================================

Explain the actual dataset used.

Current project documentation identifies:
WorldFloods v2.

Explain:
- what it contains;
- image structure;
- flood masks;
- permanent-water information if actually used;
- train/validation/test structure;
- spectral bands;
- image representation;
- label generation;
- preprocessing;
- crop size;
- resizing;
- normalization;
- augmentation;
- class imbalance;
- how the dataset becomes a binary classification dataset.

The report must distinguish:
Original dataset structure
from
the project's derived binary classification structure.

Do not invent dataset counts.

Verify all counts from the actual files/logs.

============================================================
20. MACHINE LEARNING IMPLEMENTATION
============================================================

Explain the actual implementation in simple but technically correct language.

Cover:

EfficientNet-B0
Transfer learning
ImageNet pretrained weights
Classifier modification
Input size
RGB bands
Preprocessing
Label derivation
Data augmentation
Train/validation/test split
Class imbalance
Class weights
Loss function
Optimizer
Learning rate
Scheduler
Batch size
Number of epochs
Frozen backbone phase
Fine-tuning phase
Checkpointing
GPU training
Evaluation metrics
Confusion matrix
Precision
Recall
F1
Accuracy
Specificity if actually calculated
Inference
Softmax
Risk interpretation
Grad-CAM

Do not include unnecessary mathematics.

Where mathematics is useful, explain it simply.

============================================================
21. ACTUAL TRAINING RESULTS
============================================================

Use actual results only.

The existing document reports approximately:
- 98.1% accuracy;
- 98.1% precision;
- 100% recall;
- 99.0% F1.

These values MUST be verified against the actual evaluation output before being used.

If the real output differs, use the real output.

Do not round values incorrectly.

Explain what each metric means in this particular project.

Discuss limitations of the evaluation set, especially if the test set is small or imbalanced.

Do not present high accuracy as proof that the system will work perfectly in every real-world location.

============================================================
22. BACKEND IMPLEMENTATION
============================================================

Explain actual FastAPI implementation.

Cover, where verified:
- application entry point;
- routes;
- request validation;
- CORS;
- inference service;
- satellite service;
- error handling;
- JSON response;
- image encoding;
- model loading;
- connection with frontend.

Explain each major endpoint actually present in the project.

Do not invent endpoints.

============================================================
23. SATELLITE INTEGRATION
============================================================

Explain actual Sentinel-2 / Copernicus integration.

Where verified, document:
- coordinate input;
- bounding box;
- authentication;
- OAuth2;
- Processing API;
- evalscript;
- requested bands;
- GeoTIFF response;
- rasterio;
- validation;
- temporary file handling;
- connection to inference.

Clearly explain the difference between:
Demo mode
and
Live satellite mode.

Do not claim live data was used if a screenshot/evidence does not support it.

============================================================
24. FRONTEND IMPLEMENTATION
============================================================

Explain the actual React application.

Cover:
- application structure;
- main components;
- map;
- location selection;
- demo/live mode;
- API communication;
- result dashboard;
- risk display;
- satellite image;
- Grad-CAM;
- weather;
- AQI;
- error handling;
- responsive behaviour if actually implemented.

Explain screenshots in detail.

============================================================
25. DATABASE IMPLEMENTATION
============================================================

The current project documentation states that no persistent database is implemented.

If the actual source code confirms this:
- state clearly that no persistent database is currently used;
- explain that data is processed in memory during assessment;
- include the college-required Database Implementation section;
- explain why a database was not required for the current prototype;
- explain possible future database use.

Do NOT create fake database tables.

Do NOT claim PostgreSQL/SQLite/Firebase exists unless the source code proves it.

============================================================
26. PROJECT MANAGEMENT / JIRA
============================================================

The existing report states that Jira/Trello was NOT actually used and development was tracked through a development log and Git.

If verified:
- state that Jira was not used during actual development;
- do not pretend that Jira screenshots or Jira boards existed;
- explain what was actually used;
- if the college requires "Using Project Management Tool Jira", provide a short proposed Jira structure clearly labelled as a proposed academic representation, NOT actual usage.

Do not fabricate Jira activity.

============================================================
27. TESTING
============================================================

Create detailed project-specific testing.

Include:
- unit-level testing;
- dataset pipeline testing;
- model testing;
- inference testing;
- API testing;
- satellite integration testing;
- frontend testing;
- map/location testing;
- weather/AQI testing;
- error handling;
- integration testing;
- end-to-end testing.

For every test case use a simple structure:

Test Case ID
Test Description
Input
Expected Result
Actual Result
Status

Use actual evidence where available.

Do not mark something PASS unless there is a basis for it.

============================================================
28. RESULTS AND DISCUSSION
============================================================

This chapter must be detailed.

Include:
- application outputs;
- model outputs;
- training curves;
- confusion matrix;
- Grad-CAM examples;
- flood/non-flood predictions;
- risk-level interpretation;
- UI results;
- satellite retrieval result;
- weather/AQI result;
- inference time if actually measured;
- GPU memory if actually measured;
- limitations of results.

For every result image, explain what it proves and what it does NOT prove.

============================================================
29. KSLOC
============================================================

Perform KSLOC analysis from the actual project files.

Do not simply copy an approximate number from the previous report.

Count project-written source code where practical.

Exclude:
- node_modules;
- Python virtual environments;
- datasets;
- compiled files;
- build output;
- third-party packages.

Clearly state whether comments and blank lines are included or excluded.

If exact counting is possible, use exact counting.

============================================================
30. FUNCTION POINT ANALYSIS
============================================================

Include Function Point Analysis as required.

Base the function list on actual application functionality.

Do not invent functions.

Check all arithmetic carefully.

The previous report contained an obvious suspicious value in the FP table where a reverse-geocoding row showed a value that did not match its weight. Recalculate the entire FP analysis rather than copying it.

Explain the calculation in simple language.

============================================================
31. EFFORT ESTIMATION
============================================================

Provide a realistic effort estimation.

If actual time logs exist, use them.

If not, clearly label values as estimates.

Do not pretend estimated hours are actual recorded hours.

Explain how the estimate was derived.

============================================================
32. WHAT IS WRONG WITH THE CURRENT DOCUMENT — FIX ALL OF IT
============================================================

The current generated document has the following known problems and ALL must be corrected:

1. It is only around 74 pages including preliminary pages, so the useful report is nowhere near 100 pages.
2. It contains unnecessary List of Figures.
3. It contains unnecessary List of Tables.
4. It contains unnecessary List of Abbreviations.
5. It contains references/citation-style material that the user does not want.
6. It uses numbering such as 1.a and 1.b instead of simple A, B, C.
7. Numbered lists continue from previous sections instead of restarting at 1.
8. Some sections contain numbering such as 6, 7, 8 etc. when a new list should start at 1.
9. Some pages are blank or contain excessive unused space.
10. Tables contain coloured or shaded formatting.
11. The document is not consistently pure black and white.
12. Images are sometimes inserted far away from the text explaining them.
13. Code snippet images are not properly placed or readable.
14. Some images do not have sufficiently detailed explanations.
15. Some figures are effectively text diagrams rather than polished, readable academic diagrams.
16. The document contains unnecessary preliminary material.
17. Some content is too compressed and therefore fails to explain the actual implementation deeply.
18. Some sections rely too much on generic explanation instead of project evidence.
19. Some content is written as if future features are already implemented.
20. The final report needs stronger module-by-module implementation explanation.
21. The report needs stronger explanation of actual hardware limitations and resource constraints.
22. Cost analysis needs to be tied to the actual project.
23. Dataset explanation needs to be more detailed.
24. Training and evaluation need deeper explanation.
25. Every major UI/output/code image needs an explanation immediately beside/below it.
26. The final document must have professional page flow and no accidental blank pages.
27. The Table of Contents must be updated to the final page numbers.
28. The final Word document must be visually consistent from first page to last page.

============================================================
33. REQUIRED DEPTH TO REACH 100+ PAGES
============================================================

Do not simply enlarge existing paragraphs.

Expand the report through legitimate detail.

For example, Chapter 1 should explain:
- project background;
- motivation;
- need;
- problem;
- objectives;
- scope;
- limitations;
- why flood is the current focus.

Chapter 2 should deeply explain:
- user requirements;
- functional requirements;
- non-functional requirements;
- technical feasibility;
- economic feasibility;
- operational feasibility;
- schedule feasibility;
- hardware;
- software;
- proposed system;
- advantages.

Chapter 3 should explain:
- actual development activities;
- task dependencies;
- timeline;
- Gantt chart;
- activity network;
- resources;
- development model;
- actual project management approach;
- Jira section honestly.

Chapter 4 should explain each implemented module individually.

For each module include:
Purpose
Input
Processing
Output
Important files
How it connects to other modules
Implementation explanation
Evidence/screenshot where available
Current status

Chapter 5 should provide detailed system design and all required diagrams.

Chapter 6 should provide the deepest technical implementation explanation.

Chapter 7 should provide detailed testing and actual test evidence.

Chapter 8 should provide detailed results and discussion.

Chapter 9 should provide detailed cost, effort, KSLOC and FP analysis.

Chapter 10 should provide detailed achievements, conclusion, limitations recap, future drought, future wildfire, and future multi-hazard architecture.

============================================================
34. CODE AND SOURCE FILE CROSS-REFERENCE
============================================================

Whenever a code snippet is included, identify the real file from which it came.

Examples may include, only if actually present:

ml/datasets/flood_dataset.py
ml/models/flood_classifier.py
ml/training/train_flood.py
ml/inference/flood_inference.py
ml/explainability/gradcam.py
backend/app/main.py
backend/app/api/flood_routes.py
backend/app/services/sentinel_service.py
frontend/src/App.jsx
frontend/src/index.css

Do not invent filenames.

Use the actual project directory as authority.

============================================================
35. IMAGE CAPTIONING SYSTEM
============================================================

Use consistent captions.

Example:

Figure 6.7 — Flood Training Pipeline

Then immediately:

Explanation:
[Detailed project-specific explanation.]

For screenshots:

Figure 8.3 — Flood Assessment Result Screen

Explanation:
[Detailed explanation.]

For code:

Figure 6.10 — Code Snippet from flood_dataset.py Showing Flood Label Derivation

Explanation:
[Detailed explanation.]

Do not place captions at random locations.

============================================================
36. NO COLOUR EXCEPT REAL APPLICATION SCREENSHOTS
============================================================

Academic diagrams:
BLACK AND WHITE.

Tables:
BLACK AND WHITE.

Headings:
BLACK.

Text:
BLACK.

Code blocks:
BLACK text on WHITE background.

The only permitted colour content is:
- actual application screenshots;
- actual output screenshots;
- actual training graphs/confusion matrix/Grad-CAM outputs when those colours are part of the genuine project evidence.

Do not recolour genuine output evidence.

============================================================
37. QUALITY CONTROL BEFORE FINAL EXPORT
============================================================

Before producing the final DOCX, perform a complete QA pass.

Check:

PAGE COUNT
- At least 100 pages.
- Prefer 105–120.
- No fake blank pages.

NUMBERING
- Chapter numbers correct.
- Section letters restart correctly.
- Every numbered list starts at 1.
- No accidental continuation from previous section.
- No 1.a / 1.b style.

FORMATTING
- Times New Roman everywhere.
- Headings 16/14.
- Body 12.
- Black text.
- Justified body.
- Tables black and white.
- No coloured backgrounds.

CONTENT
- College format followed.
- Flood only implemented.
- Drought future.
- Wildfire future.
- No fabricated features.
- No fake database.
- No fake Jira usage.
- No fake experiments.
- No unsupported results.

IMAGES
- All important supplied evidence inspected.
- Code screenshots readable.
- UI/output screenshots readable.
- Every important image has caption.
- Every image has detailed explanation.
- Images placed beside the relevant discussion.
- No random image placement.

TABLES
- Correct numbering.
- Correct data.
- No colour.
- No calculation errors.

TECHNICAL ACCURACY
- Dataset information verified.
- Training configuration verified.
- Results verified.
- API information verified.
- Hardware verified.
- KSLOC recalculated.
- FP recalculated.
- Cost analysis verified.
- Testing results verified.

DOCUMENT FLOW
- No accidental blank pages.
- No huge empty areas.
- No headings stranded at page bottoms.
- Captions stay with images.
- Tables stay with relevant text.
- Table of Contents updated.
- Page numbers correct.

============================================================
38. FINAL OUTPUT
============================================================

Generate ONLY the final Word document:

Disaster_Insights_Final_Project_Documentation.docx

Do not produce a PDF as the final deliverable.

Do not produce Markdown as the final deliverable.

Do not provide a shortened version.

Do not stop at 60–70 useful pages.

The report must be a complete final-year project report of at least 100 pages, with approximately 105–120 pages preferred.

The report must look like a genuine college project report prepared by a student who actually built the system.

Most importantly:

DO NOT TRY TO MAKE THE DOCUMENT LONG BY FILLING SPACE.

MAKE THE DOCUMENT LONG BECAUSE THE PROJECT IS ACTUALLY DOCUMENTED IN DETAIL.

============================================================
39. FINAL INSTRUCTION
============================================================

Do not start writing immediately.

First:
1. Inspect the college format.
2. Inspect the current DOCX and identify every structural/formatting problem.
3. Inspect the reference report.
4. Inspect the complete project source.
5. Inspect the complete evidence/image folder.
6. Build an internal chapter-to-evidence mapping.
7. Verify actual implementation facts.
8. Plan the final page structure.
9. Generate the Word document.
10. Perform a page-by-page QA pass.
11. Correct formatting and numbering.
12. Verify that the final document is at least 100 pages.
13. Verify that there are no accidental blank pages.
14. Verify that every major screenshot/code image has a detailed explanation.
15. Verify that everything is black and white except genuine application/output evidence.
16. Verify that no unwanted references/citations/lists remain.
17. Only then export the final DOCX.

The final report must be accurate, detailed, simple, formal, readable, evidence-based and consistent with the actual Disaster Insights implementation.
