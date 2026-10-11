# DISASTER INSIGHTS — FINAL DOCUMENTATION POLISH + DIAGRAM REPLACEMENT PROMPT

You are editing the EXISTING final Word documentation:

`Disaster_Insights_Final_Project_Documentation_v3.docx`

This document is already close to the desired final version. DO NOT rebuild the report from scratch.

Your job is to perform a careful professional polishing pass, replace all poor text-based diagrams with proper visual diagrams, improve weak sections where genuinely useful, and increase the report by AT LEAST 6 meaningful pages.

The current document is approximately 97 pages. The final document MUST therefore be at least 103 pages. A target of approximately 105–110 pages is preferred.

The final deliverable must remain a Word document (.docx) only.

============================================================
1. MOST IMPORTANT RULE — PRESERVE THE GOOD DOCUMENT
============================================================

The current DOCX is the MASTER DOCUMENT.

DO NOT rewrite the report from scratch.
DO NOT shorten existing chapters.
DO NOT remove useful explanations, screenshots, code evidence, tables or diagrams unless a specific correction requires it.
DO NOT change technical facts merely for wording or style.
DO NOT restructure the report into a different college format.

Before editing:
1. Inspect the entire existing DOCX.
2. Identify every diagram, screenshot, table and code image.
3. Find every ASCII / text / keyboard-character diagram.
4. Build an internal correction checklist.
5. Preserve all good content.
6. Apply controlled edits only.
7. Perform a complete final QA pass.

If the existing report and source code disagree, use the actual implementation as the authority. Never invent a correction.

============================================================
2. PROJECT TRUTH
============================================================

Current implementation: FLOOD ONLY.

Implemented functionality must only be described as implemented when supported by the project source/evidence.

Drought: FUTURE ONLY.
Wildfire: FUTURE ONLY.

Never claim that Drought or Wildfire were implemented, trained, tested, demonstrated or deployed.

Do not invent:
- modules;
- APIs;
- datasets;
- experiments;
- results;
- database systems;
- user roles;
- cloud deployment;
- hardware;
- source files;
- exact measurements.

============================================================
3. MAIN CHANGE — REPLACE ALL ASCII / TEXT DIAGRAMS
============================================================

The current document contains diagrams rendered as text, for example:

    Common Web Application (React Frontend)
             |
             v
    Common Services (Location, Satellite, Weather, AQI)
             |
             v
     +-----------------------------+
     | Flood | Drought | Wildfire |
     | Model |  Model  |  Model    |
     |CURRENT| PLANNED | PLANNED   |
     +-----------------------------+
             |
             v
    Common Result / Dashboard Layer

This is NOT acceptable in the final report.

Find EVERY diagram in the report that is represented by:
- ASCII art;
- plus signs;
- pipes;
- keyboard arrows;
- monospace boxes;
- pseudo-UML;
- terminal-style flowcharts;
- text-based architecture drawings;
- text-only process diagrams.

Replace them with REAL diagram images.

Do not leave a text-based version beside the new diagram unless a tiny text description is genuinely needed in normal prose.

============================================================
4. PROFESSIONAL DIAGRAM STYLE
============================================================

The replacement diagrams must look like diagrams made by a real software development team using tools such as StarUML, draw.io, Visio, Lucidchart or professionally designed Canva diagrams.

Required visual style:
- minimal;
- clean;
- professional;
- academic;
- technically accurate;
- easy to read;
- balanced spacing;
- consistent alignment;
- consistent line thickness;
- consistent typography.

Absolutely NO:
- AI art;
- photos;
- futuristic graphics;
- glowing effects;
- 3D objects;
- gradients;
- cartoon graphics;
- decorative illustrations;
- marketing-style visuals;
- excessive colour;
- flashy effects;
- unnecessary icons.

Academic diagrams should be essentially black and white:
- white background;
- black text;
- black outlines;
- black arrows;
- simple professional boxes;
- no coloured backgrounds.

Actual application screenshots/output images may retain their genuine colours because they are real project evidence. Do not recolour genuine application screenshots.

============================================================
5. USE GEMINI IMAGE / DIAGRAM GENERATION
============================================================

Since the current editing model is Gemini, use its available image/diagram generation capability for replacement diagrams where available.

DO NOT make the document engine draw diagrams from keyboard characters.

Generate a real image asset, save it, then insert it into the DOCX.

Before inserting any generated diagram:
- verify all labels;
- verify spelling;
- verify arrows;
- verify the architecture matches the real project;
- verify no fake components were introduced;
- verify the image is readable at normal page size.

If image generation is unavailable for a particular figure, create a proper vector-like diagram using a document/drawing method rather than ASCII text.

============================================================
6. DIAGRAMS TO RECHECK
============================================================

Inspect the entire document, but at minimum recheck:

Chapter 3:
- Gantt Chart
- Activity Network Diagram

Chapter 5:
- Context Diagram
- Level-0 DFD
- Level-1 DFD
- Conceptual Data Model / ER-style representation
- Class Diagram
- State Transition Diagram
- Use Case Diagram
- Overall System Architecture
- Activity Diagram
- Component Diagram
- Collaboration Diagram if required/present
- Deployment Diagram

Implementation / Methodology:
- EfficientNet-B0 architecture overview, if currently poor
- Transfer Learning workflow, if currently poor
- Data preprocessing pipeline, if currently poor
- Training pipeline, if currently poor
- Inference pipeline, if currently poor
- Grad-CAM workflow, if currently poor
- Satellite retrieval workflow, if currently poor
- Frontend-backend communication workflow, if currently poor

Future Scope:
- Future multi-hazard architecture

Do not regenerate figures that are already professional and readable. Replace only bad, text-based, broken or low-quality figures.

============================================================
7. UML DIAGRAMS MUST LOOK LIKE REAL UML
============================================================

For Class Diagram, Use Case Diagram, State Transition Diagram, Component Diagram, Activity Diagram and Deployment Diagram, use proper software-engineering conventions.

Class Diagram:
- class name;
- attributes;
- methods;
- meaningful relationships;
- no invented classes.

Use Case Diagram:
- actor;
- use-case ellipses;
- system boundary;
- clean relationships.

State Diagram:
- states;
- transition arrows;
- event labels;
- start/end where appropriate.

Component Diagram:
- real project components;
- clear dependencies/interfaces where appropriate.

Deployment Diagram:
- local developer machine;
- browser;
- frontend/dev server;
- FastAPI/Uvicorn;
- PyTorch/CUDA/model where applicable;
- external services actually used;
- internet boundary.

Never add fake enterprise infrastructure just to make the diagram look sophisticated.

============================================================
8. CONTEXT DIAGRAM
============================================================

Create a proper context diagram with clean boxes and arrows.

Where supported by the implementation, show:
- User;
- Disaster Insights System;
- Copernicus / Sentinel Hub;
- Open-Meteo;
- OpenStreetMap / Nominatim;
- WorldFloods v2 as a training/demo data source if appropriate.

The user should appear to interact with the system, not directly with internal implementation classes.

============================================================
9. DFD DIAGRAMS
============================================================

The Level-0 DFD should clearly show the major processes and data flows.

Where supported, represent:
1. Location Processing
2. Satellite Retrieval
3. Flood Assessment Engine
4. Result Formatting
5. Weather / AQI Service

The Level-1 DFD should decompose the flood assessment process into real stages such as:
- image loading;
- preprocessing;
- model forward pass;
- softmax/classification;
- risk interpretation;
- Grad-CAM generation;
- result assembly.

Use clean labelled arrows and proper flow direction.

============================================================
10. MULTI-HAZARD ARCHITECTURE DIAGRAM
============================================================

The future multi-hazard architecture should communicate this structure visually:

React Frontend
↓
Common Services
(Location / Satellite / Weather / AQI)
↓
Hazard Model Layer
Flood — Implemented
Drought — Future
Wildfire — Future
↓
Common Result / Dashboard Layer

DO NOT reproduce this as text or ASCII.

Use a professional layered architecture diagram with simple black-and-white boxes.

Clearly write:
- Flood Model — Implemented
- Drought Model — Planned Future Extension
- Wildfire Model — Planned Future Extension

Do not use colour coding to distinguish them.

============================================================
11. GANTT CHART AND ACTIVITY NETWORK
============================================================

The existing text-based Gantt chart must become a real professional Gantt chart image.

Use the actual project phases and durations already documented.

Do not invent exact dates if exact dates were not recorded.

Label the timeline as approximate/estimated when appropriate.

The Activity Network must also be a real professional node-and-arrow diagram.

Maintain the actual task dependencies. Do not invent a critical path unless it is supported by the documented project plan.

============================================================
12. IMAGE PLACEMENT
============================================================

Every diagram must appear immediately after the paragraph that introduces it.

Use this order:

Paragraph introducing the figure.

Figure X.X — Figure Title

[REAL DIAGRAM IMAGE]

Explanation:
Detailed explanation of the figure.

Then continue with the next paragraph.

Never:
- place a diagram several pages away from its explanation;
- place a diagram in an unrelated section;
- separate caption and image;
- leave a huge empty region around an image;
- stretch an image beyond its natural aspect ratio.

Keep figure + caption + explanation together where practical.

============================================================
13. EVERY FIGURE NEEDS A REAL EXPLANATION
============================================================

Do not write:
- "The figure is shown above."
- "This diagram explains the system."
- "The above diagram shows the flow."

Instead explain:
1. What is visible.
2. What each major block represents.
3. How information moves.
4. Which components communicate.
5. What the arrows mean.
6. How the figure maps to the real implementation.
7. Why the figure is useful for understanding the project.

Keep the language simple and formal.

============================================================
14. CODE AND UI IMAGES
============================================================

Preserve existing good code/output images.

For code screenshots, ensure they remain readable and continue to include:
- figure title;
- actual source filename;
- purpose;
- explanation of important lines/parts;
- connection to the system;
- resulting behaviour/output where relevant.

For UI/output screenshots, ensure each major screenshot has:
- figure title;
- user action leading to the screen;
- description of visible elements;
- explanation of what the backend/model did;
- explanation of why the output matters.

Do not paste a screenshot merely for decoration.

============================================================
15. ADD AT LEAST 6 REAL PAGES
============================================================

The current document is approximately 97 pages.

The final document MUST be at least 103 pages.

Preferred final length: 105–110 pages.

Do NOT add pages using:
- blank pages;
- huge whitespace;
- oversized headings;
- oversized images;
- duplicate screenshots;
- repeated paragraphs;
- meaningless appendices;
- repetitive definitions.

Instead, expand existing sections with useful project-specific material.

Use the following areas first, wherever they are currently too brief:

A. COMPLETE END-TO-END WORKFLOW

Add a detailed step-by-step explanation of:
1. Application startup
2. Location selection
3. Demo/live selection
4. Satellite image retrieval
5. Image validation
6. Preprocessing
7. Model inference
8. Flood probability
9. Risk interpretation
10. Grad-CAM generation
11. Result assembly
12. JSON/API response
13. Frontend rendering

For each step explain input → processing → output.

B. DATA PREPROCESSING

Expand the real implementation of:
- GeoTIFF reading;
- centre crop;
- RGB band selection;
- reflectance scaling;
- resizing;
- ImageNet normalisation;
- label derivation;
- permanent-water handling if actually used;
- augmentation;
- why training and inference preprocessing must match.

C. TRAINING PROCESS

Expand:
- why transfer learning was selected;
- frozen backbone phase;
- fine-tuning phase;
- class imbalance;
- class weights;
- loss function;
- optimizer;
- learning-rate handling;
- scheduler;
- checkpoint selection;
- training log interpretation;
- why GPU hardware was used.

D. INFERENCE PROCESS

Expand:
- model loading;
- image preparation;
- forward pass;
- probabilities;
- risk level interpretation;
- Grad-CAM;
- JSON result creation;
- frontend display.

E. EXTERNAL SERVICE INTEGRATION

Add detailed, project-specific explanations for:
- Copernicus/Sentinel Hub;
- Open-Meteo weather;
- Open-Meteo AQI;
- OpenStreetMap;
- Nominatim.

For each, explain:
- why it is needed;
- what input it receives;
- what data it returns;
- where it enters the application workflow;
- what happens when the service is unavailable.

Do not invent behaviour.

F. ERROR HANDLING

Expand only the errors actually handled by the implementation, such as:
- invalid location;
- API failure;
- missing satellite data;
- invalid response;
- model loading failure;
- assessment failure;
- frontend error state.

For each explain how the system responds.

G. HARDWARE AND RESOURCE CONSTRAINTS

Expand:
- CPU;
- 16 GB RAM if verified;
- RTX 4050 6 GB VRAM if verified;
- storage requirement;
- dataset storage;
- checkpoint storage;
- GPU vs CPU inference;
- training constraints;
- internet dependency;
- local-only deployment.

Use actual values from the project evidence.

H. TESTING METHODOLOGY

Explain how the actual implementation was tested:
- dataset pipeline;
- smoke testing;
- model testing;
- inference;
- backend/API;
- satellite integration;
- frontend;
- map/location;
- weather/AQI;
- error handling;
- end-to-end integration.

Use actual evidence where available.

I. RESULTS INTERPRETATION

Expand the explanation of:
- accuracy;
- precision;
- recall;
- F1;
- specificity if actually calculated;
- confusion matrix;
- training curves;
- Grad-CAM;
- inference timing if actually measured.

Explain what each result means for THIS project and also what it does NOT prove.

J. DEVELOPMENT DECISIONS / PROBLEMS SOLVED

Where supported by the development log/source evidence, document real issues such as:
- band-order mismatch;
- cloud masking behaviour;
- API integration problems;
- preprocessing mismatches;
- frontend/backend integration issues.

Use the structure:
Problem → Investigation → Change → Result.

Do not fabricate problems.

K. FUTURE DROUGHT AND WILDFIRE DESIGN

Add useful detail about how the current architecture could later support:

Drought:
- possible dataset requirements;
- spectral/environmental information;
- suitable label strategy;
- model adaptation;
- new preprocessing needs;
- integration into the existing model layer.

Wildfire:
- required datasets;
- fire/burn-related spectral information;
- model adaptation;
- preprocessing needs;
- integration into the common architecture.

Clearly state that both are future plans only.

============================================================
16. DO NOT CREATE RANDOM NEW CHAPTERS
============================================================

All additional material must fit inside the existing college-required chapters and sections.

Do not create extra chapters merely to increase page count.

============================================================
17. BLACK AND WHITE FORMATTING
============================================================

Maintain:

Font: Times New Roman
Chapter heading: 16 pt
Section heading: 14 pt
Body: 12 pt
Body text: Justified

Academic text: pure black.

Tables:
- white background;
- black text;
- black borders;
- no shading;
- no coloured headers.

Academic diagrams:
- white background;
- black text;
- black outlines;
- black arrows;
- no colourful styling.

Actual application/output screenshots may retain their original colours.

============================================================
18. NO REFERENCES / CITATION CLUTTER
============================================================

Preserve the existing user's preference:
- no References section added;
- no citation markers;
- no numbered citations;
- no footnotes;
- no endnotes;
- no bibliography page.

Do not introduce citation clutter during this editing pass.

============================================================
19. DOCUMENT NUMBERING
============================================================

Preserve the corrected numbering style:

Chapter 1
A. Background and Motivation of the Project
B. Objective and Scope of the Project
C. Problem Definition
D. Limitations

Do not introduce 1.a, 1.b, 2.a, etc.

Every independent numbered list must restart at 1.

Do not allow numbering to continue from a previous list.

============================================================
20. TOC
============================================================

After all editing and image insertion is complete, update the Table of Contents.

Make sure all page numbers reflect the final layout.

Do not add:
- List of Figures;
- List of Tables;
- List of Abbreviations.

============================================================
21. TECHNICAL ACCURACY CHECK
============================================================

Before final export, cross-check important claims against the actual project files wherever possible.

Pay special attention to:
- dataset counts;
- training configuration;
- actual test metrics;
- hardware;
- software versions;
- model file size;
- API endpoints;
- source-file names;
- source-file line counts;
- performance timings;
- storage values.

Do not silently turn estimates into recorded facts.

============================================================
22. FINAL PROFESSIONAL-QUALITY TEST
============================================================

View every recreated diagram as if you were an examiner.

Ask:
"Could this reasonably have been produced by a student software-development team using StarUML, draw.io, Visio or professional Canva?"

The answer must be YES.

The diagram must NOT look like:
- AI artwork;
- an infographic advertisement;
- a marketing poster;
- a futuristic concept graphic;
- a screenshot of text characters.

============================================================
23. FINAL QA CHECKLIST
============================================================

PAGE COUNT
[ ] Minimum 103 pages.
[ ] Prefer 105–110 pages.
[ ] No blank pages.
[ ] No fake padding.

CONTENT
[ ] Existing useful content preserved.
[ ] At least 6 meaningful pages added.
[ ] Flood is the only implemented hazard.
[ ] Drought is future.
[ ] Wildfire is future.
[ ] No fabricated implementation.

DIAGRAMS
[ ] All bad ASCII diagrams replaced.
[ ] No text-box diagrams made from keyboard characters remain.
[ ] Gantt chart is a real graphic.
[ ] Activity network is a real graphic.
[ ] UML figures look like real UML.
[ ] Architecture figures are clean and minimal.
[ ] Diagrams are black and white.
[ ] No decorative AI artwork.

FIGURES
[ ] Correct figure placement.
[ ] Proper caption for every figure.
[ ] Detailed explanation after every important figure.
[ ] Code screenshots readable.
[ ] UI/output screenshots readable.
[ ] No stretched or distorted images.

FORMATTING
[ ] Times New Roman everywhere.
[ ] 16 pt chapter headings.
[ ] 14 pt section headings.
[ ] 12 pt body.
[ ] Body justified.
[ ] Pure black academic text.
[ ] Tables black and white.
[ ] White backgrounds.

NUMBERING
[ ] A/B/C/D sections.
[ ] No 1.a / 1.b style.
[ ] Every new numbered list starts at 1.
[ ] No accidental continued numbering.

PAGE FLOW
[ ] No unnecessary page breaks.
[ ] No huge empty areas.
[ ] No captions separated from images.
[ ] No headings stranded at page bottoms.
[ ] No random figure placement.

TOC
[ ] Updated after final edits.
[ ] Final page numbers correct.

============================================================
24. FINAL FILE
============================================================

Export ONLY:

`Disaster_Insights_Final_Project_Documentation_Final.docx`

Do not export PDF.
Do not create a shortened version.
Do not regenerate the report from scratch.

============================================================
25. FINAL OPERATING RULE
============================================================

EDIT, DO NOT RESTART.

PRESERVE, THEN IMPROVE.

REPLACE BAD ASCII DIAGRAMS WITH REAL PROFESSIONAL DIAGRAM IMAGES.

ADD AT LEAST SIX REAL PAGES OF USEFUL PROJECT-SPECIFIC DOCUMENTATION.

KEEP THE VISUAL STYLE MINIMAL, BLACK, WHITE, PROFESSIONAL AND ACADEMIC.

MAKE EVERY DIAGRAM LOOK LIKE A REAL SOFTWARE DEVELOPMENT TEAM CREATED IT.

MAKE EVERY FIGURE UNDERSTANDABLE FROM ITS IMAGE + EXPLANATION.

MAKE THE FINAL WORD DOCUMENT EXAMINER-READY.
