# 🧠 NEURAL VISION

> **See. Understand. Experience.**

An AI-powered computer vision application that analyzes live camera input, detects objects and faces, interprets visual and emotional context, and transforms the detected scene into an interactive AI experience.

Built with React, TypeScript, TensorFlow.js, MediaPipe, and Gemini AI.

![React](https://img.shields.io/badge/React-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-blue)
![Vite](https://img.shields.io/badge/Vite-purple)
![Gemini%20AI](https://img.shields.io/badge/Gemini%20AI-orange)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-blue)

---

## 🌐 Live Demo

🔗 [https://yantano.github.io/neural-vision/](https://yantano.github.io/neural-vision/)

---

## 📸 Preview

![cat-run Preview](https://i.gyazo.com/ff7fcef97e53ef5169ef3c4ffd259f7a.jpg)

---

# ✨ Features

- 📷 Real-time camera analysis
- 👁️ Face detection and landmark tracking
- 🧠 Facial expression analysis
- 🎯 Real-time object detection
- 🔍 AI-powered scene interpretation
- 🤖 Gemini AI integration
- 📊 Live visual metrics and charts
- 🎨 Multiple camera visualization modes
- 🎵 Adaptive procedural soundscapes
- ⚡ Real-time visual overlays
- 📱 Responsive interface
- 🎛️ Interactive camera controls
- 🌈 Cyber, thermal, monochrome, and other visual modes

---

# 🧠 AI & Computer Vision

Neural Vision combines multiple AI and computer vision technologies to interpret live camera input.

### Object Detection

Uses **TensorFlow.js + COCO-SSD** to identify objects in the camera feed.

Detected objects can influence the application's visual and audio experience.

### Face & Expression Analysis

Uses **MediaPipe Face Landmarker** to track facial landmarks and analyze facial expressions such as:

- Happy
- Sad
- Surprised
- Angry
- Fear
- Disgust
- Neutral

### Gemini AI

Gemini analyzes the detected scene and generates contextual descriptions used by the application to determine an appropriate visual/audio experience.

---

# 🎵 Adaptive Sound Engine

One of the core features of Neural Vision is its **scene-aware procedural audio system**.

Detected objects and emotional states influence the generated soundscape.

| Detected Context | Sound Experience |
| ---------------- | ---------------- |
| Person | Ambient |
| Phone / Laptop | Cyberpunk |
| Cat / Dog | Playful |
| Car | Driving / Rock |
| Cup / Bottle | Acoustic / Jazz |
| Plant | Ambient Nature |
| Book | Classical |
| Angry / Fear | Tense / Dissonant |

The application dynamically transitions between soundscapes instead of simply playing a fixed audio track.

---

# 🎨 Camera Modes

Neural Vision includes multiple visual processing modes:

- ⚡ Cyber Neon
- 🌈 Vivid Natural
- 🌡️ Thermal Heat
- 🌅 Warm Sunset
- 🟢 Matrix Emerald
- ⚫ Cyber Monochrome

Each mode applies a different visual treatment to the camera feed.

---

# 🛠️ Built With

- React
- TypeScript
- Vite
- Gemini AI
- TensorFlow.js
- COCO-SSD
- MediaPipe
- Tailwind CSS
- Motion
- Recharts
- Tone.js
- Lucide React

---

# 🏗️ Architecture

```text
Camera Input
     │
     ▼
┌──────────────────────┐
│  Computer Vision     │
│                      │
│  MediaPipe           │
│  TensorFlow.js       │
│  COCO-SSD            │
└──────────┬───────────┘
           │
           ▼
   Detection Pipeline
           │
     ┌─────┴─────┐
     ▼           ▼
 Objects       Face /
 Detected      Emotion
     │           │
     └─────┬─────┘
           ▼
      Gemini AI
           │
           ▼
   Scene Interpretation
           │
     ┌─────┴─────┐
     ▼           ▼
 Visual UI    Audio Engine
     │           │
     ▼           ▼
Camera FX    Dynamic Sound
```

---

# 📂 Project Structure

```text
neural-vision/
│
├── src/
│   ├── App.tsx
│   └── ...
│
├── .github/
│   └── workflows/
│
├── index.css
├── index.html
├── server.ts
├── vite.config.ts
├── tsconfig.json
├── package.json
├── package-lock.json
├── bun.lock
├── .env.example
└── README.md
```

---

# 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/YanTano/neural-vision.git
```

### Open the project

```bash
cd neural-vision
```

### Install dependencies

```bash
npm install
```

### Configure the API key

Create a local environment file and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

> Never commit your real API key to GitHub.

### Start the development server

```bash
npm run dev
```

The application runs on:

```text
http://localhost:3000
```

---
<!--
# 📷 Screenshots

## 🧠 Neural Vision Dashboard

> Add dashboard screenshot.

## 📷 Live Camera Detection

> Add camera detection screenshot.

## 🎯 Object Detection

> Add object detection screenshot.

## 🎵 Adaptive Soundscape

> Add audio visualization screenshot.

---
-->

# 🎯 Technologies Used

| Technology | Purpose |
| ---------- | ------- |
| React | Application UI |
| TypeScript | Type-safe development |
| Vite | Development and build tooling |
| TensorFlow.js | Browser-based machine learning |
| COCO-SSD | Object detection |
| MediaPipe | Face landmark detection |
| Gemini AI | Scene interpretation |
| Recharts | Data visualization |
| Tone.js | Procedural audio |
| Motion | UI animations |
| Tailwind CSS | Interface styling |

---

# ⚙️ Technical Highlights

- Real-time browser-based computer vision
- Client-side object detection with TensorFlow.js
- Face landmark tracking using MediaPipe
- AI-assisted scene interpretation
- Real-time camera visualization
- Dynamic emotion classification
- Procedural audio generation
- Context-aware soundscape transitions
- Interactive data visualization
- Modular React architecture
- TypeScript-based application logic

---

# 🔐 Security

API credentials should be provided through environment variables and should never be committed to the repository.

```env
GEMINI_API_KEY=your_api_key_here
```

Use `.env.example` to document required configuration without exposing secrets.

---

# 📈 Development Goals

- Maintain responsive real-time camera processing
- Keep AI inference efficient in the browser
- Provide clear visual feedback for detected objects
- Keep computer vision and UI logic modular
- Provide an immersive AI interaction experience
- Maintain responsive performance across supported devices

---

# 👨‍💻 Author

**Carlo Tano**

Software QA Engineer | Computer Engineer | Front-End Developer

- GitHub: https://github.com/YanTano
- LinkedIn: https://www.linkedin.com/in/carlo-tano-7375bb1bb/
- Portfolio: https://yantano.github.io/carlo-tano-portfolio/

---

# ⭐ Support

If you found Neural Vision interesting, consider giving the repository a ⭐ on GitHub!
