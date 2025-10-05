# AIRNOTE

## The World is YOUR Notebook

### Major League Hacking — Rutgers University Hackathon

* Redefining Note-Taking: Think Freely, Write Anywhere, Learn Intuitively
* AI-Powered Note Graph and Recognition System || Bringing Augmented Reality Note-Taking to Life

## Insperation

The idea for AirNote stemmed from the increasing popularity of AR glasses and spatial computing. We were inspired by the future of intelligent interfaces and how they can help people process information faster. As **Mark Zuckerberg** said,  
> “People without smart glasses may one day be at a significant cognitive disadvantage compared to those who do use the tech.”

We wanted to build something that captures this vision — a tool that can take handwritten notes, understand them, and visualize the relationships between ideas in real time.

## Description

AirNote combines **AI image recognition**, **real-time labeling**, and **graph-based visualization** to make note-taking more intelligent.  
Our system takes handwritten screenshots or AR overlays, merges them into a single composite image, and sends them to the **Gemini 2.5 Flash** model for interpretation. The model returns a **label** and **summary** that are stored and visualized in an interactive **Vault Graph**, similar to Obsidian’s graph view.

Every note is represented as a node connected by semantic similarity, helping users visually explore how their ideas relate. Users can label notes, view AI-generated summaries, and delete or update them dynamically — all through a modern, intuitive interface.

### Dependencies

* Python 3.12.7+  
* Node.js 18+  
* Firebase  
* FastAPI  
* React + Vite

### What it does

AirNote takes handwritten or AR-based note screenshots, analyzes them with Gemini, and generates structured labels and summaries.  
These notes are stored in Firebase Firestore and visualized in an interactive Vault Graph that dynamically connects related ideas based on embedding similarity.  
Users can explore, label, or delete notes directly from the dashboard while the backend ensures real-time synchronization.

## Collaborator

* [Vishal Nagamalla](https://github.com/vishal-nagamalla)  
* [Siddharth Paul](https://github.com/sidp24)

## Built with

* python  
* fastapi  
* react  
* firebase  
* gemini 2.5 flash  
* tailwindcss  
* typescript  
* opencv  
* mediapipe