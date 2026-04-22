# ✨ AI Air Canvas — Real-Time Writing Using Hand Gestures

> Draw in the air with your finger. No mouse. No keyboard. Just your hand and a webcam.

---

## 🧠 How It Works

```
Webcam  →  MediaPipe Hand Detection  →  Index Finger Tracking
                                              ↓
                                   Draw on Virtual Canvas (OpenCV)
                                              ↓
                                   Composite onto Live Video Feed
```

### Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ **1 finger (index only)** | Draw on canvas |
| ✌️ **2+ fingers raised** | Stop drawing / hover mode |
| ✌️ **Hover over colour button** | Switch active colour |
| ✌️ **Hover over Clear** | Wipe entire canvas |
| **C key** | Clear canvas (keyboard shortcut) |
| **Q or ESC** | Quit the application |

---

## 📁 Project Structure

```
AI-Air-Canvas/
│
├── air_canvas.py          ← Main application (run this)
├── requirements.txt       ← Python dependencies
├── README.md              ← This file
│
├── utils/
│   ├── __init__.py
│   └── hand_tracking.py   ← MediaPipe wrapper
│
└── assets/
    └── icons/             ← (reserved for future icons)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python **3.8 – 3.11** (MediaPipe does not support 3.12 yet)
- A working **webcam**
- **VS Code** (recommended)

---

### Step-by-step (VS Code)

#### 1. Clone / download the project

```bash
# If using git
git clone <repo-url>
cd AI-Air-Canvas

# Or just unzip the folder and open it in VS Code
```

#### 2. Open the folder in VS Code

```
File → Open Folder → select AI-Air-Canvas
```

#### 3. Create a virtual environment

Open the **VS Code Terminal** (`Ctrl + `` ` ```) and run:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> ⏳ MediaPipe is ~50 MB — this may take a minute.

#### 5. Run the project

```bash
python air_canvas.py
```

A window titled **"✨ AI Air Canvas"** will open showing your webcam feed.

---

## 🎨 Colour Palette

| Button | Colour |
|--------|--------|
| 🔴 Red | Draw in red |
| 🟢 Green | Draw in green |
| 🔵 Blue | Draw in blue |
| 🟡 Yellow | Draw in yellow |
| 🗑️ Clear | Wipe canvas |

**Switch colours:** Raise 2+ fingers and hover the fingertip over a button at the top of the screen.

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Python 3.8+** | Core language |
| **OpenCV** | Webcam capture, canvas rendering, UI |
| **MediaPipe** | Real-time hand landmark detection (21 points) |
| **NumPy** | Fast array operations for canvas compositing |

---

## 🧩 Key Concepts

- **MediaPipe Hands** detects 21 3D landmarks per hand at ~30 FPS
- **Landmark 8** (index fingertip) is tracked as the drawing cursor
- **Deque smoothing** averages the last 8 positions to eliminate jitter
- **Canvas compositing** blends drawing layer with live video using bitwise masks
- **Finger counting** compares tip vs PIP joint Y-coordinates

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open webcam` | Check webcam is connected; try changing `cv2.VideoCapture(0)` to `(1)` |
| Poor detection | Improve lighting; ensure hand is clearly visible |
| Laggy drawing | Lower webcam resolution in `air_canvas.py` (change 1280×720 to 640×480) |
| `ModuleNotFoundError` | Make sure you activated the virtual environment before running |
| MediaPipe install fails | Use Python 3.8–3.11 (not 3.12) |

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built with ❤️ using Python, OpenCV & MediaPipe*
