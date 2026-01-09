# 🧠 Echo – Local Voice-Controlled AI Assistant

**Echo** is a voice-activated, conversational AI assistant powered by a local LLM (Qwen via Ollama). It listens for a wake word, processes spoken commands using a local language model with LangChain, and responds out loud via TTS. It supports tool-calling for dynamic functions like checking the current time.

---

## 🚀 Features

- 🗣 Voice-activated with wake word **"Echo"**
- 🧠 Local language model (Qwen 3 via Ollama)
- 🔧 Tool-calling with LangChain
- 🔊 Text-to-speech responses via `pyttsx3`
- 🌍 Example tool: Get the current time in a given city
- 🔐 Optional support for OpenAI API integration

---


## ▶️ How It Works (`main.py`)

1. **Startup & local LLM Setup**
   - Initializes a local Ollama model (`qwen3:1.7b`) via `ChatOllama`
   - Registers tools (`get_time`) using LangChain

2. **Wake Word Listening**
   - Listens via microphone (e.g., `device_index=0`)
   - If it hears the word **"Echo"**, it enters "conversation mode"

3. **Voice Command Handling**
   - Records the user’s spoken command
   - Passes the command to the LLM, which may invoke tools
   - Responds using `pyttsx3` text-to-speech (with optional custom voice)

4. **Timeout**
   - If the user is inactive for more than 30 seconds in conversation mode, it resets to wait for the wake word again.

---

## 🤖 How To Start Echo

1. **Install System Dependencies**  
   Run these commands to install required system libraries:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-venv portaudio19-dev libasound2-dev libcap-dev libespeak1 libcamera-dev python3-opencv -y
   ```

2. **Set Up a Virtual Environment (Recommended for Pi)**  
   Since modern Pi OS protects system packages, create a virtual environment:
   ```bash
   # Create the environment
   python3 -m venv venv
   
   # Activate it
   source venv/bin/activate
   ```

3. **Install Python Libraries**  
   With the environment active, install the project requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up the Local Model**  
   Ensure you have the `llama3.2:3b` model available in Ollama (run `ollama pull llama3.2:3b`).

5. **Run Echo**  
   ```bash
   # Make sure venv is active (you see (venv) in your prompt)
   python3 main.py
   ```
---

