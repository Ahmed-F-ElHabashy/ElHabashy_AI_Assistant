import sys
import os
import re
import requests
import time
import json
from datetime import datetime
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QLabel, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor, QSyntaxHighlighter, QTextDocument

class CodeHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.highlighting_rules = []
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(0, 0, 255))
        keywords = [
            "def", "class", "return", "if", "else", "elif", "for", "while",
            "import", "from", "as", "try", "except", "finally", "with", "pass",
            "break", "continue", "raise", "yield", "lambda", "None", "True", "False"
        ]
        for keyword in keywords:
            pattern = r'\b' + keyword + r'\b'
            self.highlighting_rules.append((re.compile(pattern), keyword_format))
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(0, 128, 0))
        self.highlighting_rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), string_format))
        self.highlighting_rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), string_format))
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(128, 128, 128))
        self.highlighting_rules.append((re.compile(r'#.*'), comment_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            for match in pattern.finditer(text):
                start, end = match.span()
                self.setFormat(start, end - start, format)

class OllamaWorker(QThread):
    response_chunk = pyqtSignal(str, str)
    finished_signal = pyqtSignal(float)
    error_signal = pyqtSignal(str)

    def __init__(self, prompt, model_name, use_gpu, parent_window, mode="normal", continue_context=False):
        super().__init__()
        self.prompt = prompt
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.parent = parent_window
        self.mode = mode
        self.continue_context = continue_context
        self._is_running = True
        self.retry_count = 0
        self.max_retries = 2

    def run(self):
        start_time = time.time()
        try:
            final_prompt = self.build_final_prompt()
            payload = {
                "model": self.model_name,
                "prompt": final_prompt,
                "stream": True,
                "options": {
                    "num_gpu": 99999 if self.use_gpu else 0,  # Offload all layers to GPU if use_gpu is True
                    "main_gpu": 0,
                    "low_vram": False,
                    "temperature": 0.7 if self.mode == "deepthink" else 0.5,
                    "max_tokens": -1 if self.continue_context else 2048
                }
            }
            print(f"Sending payload: {json.dumps(payload, indent=2)}")

            with requests.post(
                "http://127.0.0.1:11434/api/generate",
                json=payload,
                stream=True,
                timeout=3000
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not self._is_running:
                        break
                    if line:
                        chunk = line.decode("utf-8")
                        data = self.parse_chunk(chunk)
                        if data:
                            self.response_chunk.emit(data["content"], data["timestamp"])

            elapsed = time.time() - start_time
            self.finished_signal.emit(elapsed)

        except requests.exceptions.RequestException as e:
            if "CUDA out of memory" in str(e) and self.retry_count < self.max_retries:
                self.retry_count += 1
                self.parent.update_status(f"GPU memory full - retrying with CPU (attempt {self.retry_count})")
                self.use_gpu = False
                self.run()
            else:
                self.error_signal.emit(f"Error: {str(e)}")
                elapsed = time.time() - start_time
                self.finished_signal.emit(elapsed)
        except Exception as e:
            self.error_signal.emit(f"Error: {str(e)}")
            elapsed = time.time() - start_time
            self.finished_signal.emit(elapsed)

    def build_final_prompt(self):
        if self.continue_context:
            return "\n".join([msg["raw"] for msg in self.parent.message_history])
        if self.mode == "deepthink":
            return f"[DeepThink Mode Enabled]\n{self.prompt}\nPlease analyze this thoroughly, considering multiple perspectives and potential edge cases."
        return self.prompt

    def parse_chunk(self, chunk):
        try:
            data = json.loads(chunk)
            if "response" in data:
                return {"content": data["response"], "timestamp": datetime.now().strftime("%H:%M:%S")}
            elif "error" in data:
                print(f"Error from Ollama: {data['error']}")
                return None
            return None
        except Exception:
            return None

    def stop(self):
        self._is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ElHabashy AI Assistant")
        self.setGeometry(100, 100, 1000, 800)
        self.models = [
            "ALIENTELLIGENCE/pythonconsultantv2:latest",
            "qwen2.5-coder:latest",
            "deepseek-r1:latest",
            "qwen2.5-coder:3b",
            "qwen2.5:7b",
            "nomic-embed-text:latest",
            "qwen2.5-coder:1.5b-base",
            "llama3.1:8b",
            "deepseek-r1:8b",
            "deepseek-coder-v2:latest",
            "deepseek-coder:6.7b"
        ]
        self.message_history = []
        self.typing_queue = deque()
        self.current_mode = "normal"
        self.init_ui()
        self.check_gpu_support()

        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self.process_typing_queue)
        self.typing_delay = 5

        self.highlighter = CodeHighlighter(self.chat_display.document())

    def get_installed_models(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
        except Exception as e:
            print(f"Error fetching models: {e}")
        return []

    def init_ui(self):
        main_layout = QVBoxLayout()
        model_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(sorted(self.models))
        self.model_selector.setEditable(True)
        self.model_selector.setInsertPolicy(QComboBox.NoInsert)
        self.refresh_models_btn = QPushButton("🔄")
        self.refresh_models_btn.setToolTip("Refresh model list")
        self.refresh_models_btn.clicked.connect(self.refresh_models)
        model_layout.addWidget(QLabel("Select Model:"))
        model_layout.addWidget(self.model_selector)
        model_layout.addWidget(self.refresh_models_btn)
        main_layout.addLayout(model_layout)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(""" 
            QTextEdit { 
                background-color: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Segoe UI', Arial;
                font-size: 14px;
            }
            QScrollBar:vertical {
                width: 10px;
            }
        """)
        main_layout.addWidget(self.chat_display)

        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your message here... (Use Shift+Enter for new line)")
        self.input_field.setMaximumHeight(100)
        self.input_field.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', monospace;
                font-size: 13px;
            }
        """)
        main_layout.addWidget(self.input_field)

        control_layout = QHBoxLayout()
        self.send_btn = QPushButton("Send (Ctrl+Enter)")
        self.send_btn.clicked.connect(self.send_message)
        self.stop_btn = QPushButton("Stop (Esc)")
        self.stop_btn.clicked.connect(self.stop_generation)
        self.gpu_check = QCheckBox("Use GPU")
        self.gpu_check.setChecked(True)
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Normal Mode", "DeepThink Mode"])
        self.mode_selector.currentTextChanged.connect(self.set_mode)
        self.continue_btn = QPushButton("Continue")
        self.continue_btn.clicked.connect(self.continue_response)
        self.regenerate_btn = QPushButton("Regenerate")
        self.regenerate_btn.clicked.connect(self.regenerate_response)
        control_layout.addWidget(self.send_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.gpu_check)
        control_layout.addWidget(self.mode_selector)
        control_layout.addWidget(self.continue_btn)
        control_layout.addWidget(self.regenerate_btn)
        self.status_bar = QLabel()
        self.status_bar.setAlignment(Qt.AlignRight)
        self.status_bar.setStyleSheet("color: #666; font-style: italic;")
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.status_bar)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.send_btn.setShortcut("Ctrl+Return")
        self.stop_btn.setShortcut("Esc")

    def refresh_models(self):
        current = self.model_selector.currentText()
        self.models = self.get_installed_models()
        if not self.models:
            self.models = [
                "ALIENTELLIGENCE/pythonconsultantv2:latest",
                "qwen2.5-coder:latest",
                "deepseek-r1:latest",
                "qwen2.5-coder:3b",
                "qwen2.5:7b",
                "nomic-embed-text:latest",
                "qwen2.5-coder:1.5b-base",
                "llama3.1:8b",
                "deepseek-r1:8b",
                "deepseek-coder-v2:latest",
                "deepseek-coder:6.7b"
            ]
        self.model_selector.clear()
        self.model_selector.addItems(sorted(self.models))
        if current in self.models:
            self.model_selector.setCurrentText(current)
        self.update_status(f"Loaded {len(self.models)} models")

    def set_mode(self):
        self.current_mode = "deepthink" if self.mode_selector.currentText() == "DeepThink Mode" else "normal"
        self.update_status(f"Mode changed to {self.mode_selector.currentText()}")

    def continue_response(self):
        if self.message_history and self.message_history[-1]["role"] == "assistant":
            self.send_message(continue_context=True)

    def regenerate_response(self):
        if self.message_history:
            while self.message_history and self.message_history[-1]["role"] == "assistant":
                self.message_history.pop()
            self.chat_display.clear()
            self.render_messages()
            self.send_message(regenerate=True)

    def check_gpu_support(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                self.update_status("GPU: GTX 1660 Super detected")
            else:
                self.gpu_check.setEnabled(False)
                self.gpu_check.setChecked(False)
                self.update_status("Ollama server not responding - GPU unavailable")
        except Exception as e:
            self.update_status(f"GPU check failed: {str(e)}")
            print(f"GPU Check Error: {str(e)}")

    def update_status(self, message):
        self.status_bar.setText(message)

    def send_message(self, continue_context=False, regenerate=False):
        prompt = self.input_field.toPlainText().strip()
        if not continue_context and not regenerate:
            if not prompt:
                return
            self.message_history.append({
                "role": "Habashy",
                "content": prompt,
                "raw": prompt,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            self.render_messages()
            self.input_field.clear()

        worker_mode = "deepthink" if self.mode_selector.currentText() == "DeepThink Mode" else "normal"
        self.worker = OllamaWorker(
            prompt=prompt,
            model_name=self.model_selector.currentText(),
            use_gpu=self.gpu_check.isChecked(),
            parent_window=self,
            mode=worker_mode,
            continue_context=continue_context
        )
        self.worker.response_chunk.connect(self.handle_response_chunk)
        self.worker.finished_signal.connect(self.handle_response_end)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.start()
        self.send_btn.setEnabled(False)
        self.update_status("Processing request...")

    def append_message(self, role, content):
        self.message_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        self.render_messages()

    def render_messages(self):
        self.chat_display.clear()
        for msg in self.message_history:
            if msg["role"] == "Habashy":
                bubble_style = """
                    background: #e3f2fd;
                    border: 1px solid #bbdefb;
                    margin-left: 20%;
                    margin-right: 5px;
                """
            else:
                bubble_style = """
                    background: #f5f5f5;
                    border: 1px solid #e0e0e0;
                    margin-right: 20%;
                    margin-left: 5px;
                """
            content = msg["content"]
            # Escape HTML special characters to prevent rendering issues
            content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if '```' in content:
                content = self.highlight_code_blocks(content)
            message_html = f"""
            <div style="margin: 8px 0;">
                <div style="{bubble_style}
                            border-radius: 12px;
                            padding: 8px 12px;
                            font-family: 'Segoe UI';
                            font-size: 14px;">
                    <div style="font-weight: bold; color: #0d47a1;">
                        {msg['role'].capitalize()}
                        <span style="color: #666; font-size: 0.8em;">
                            {msg['timestamp']}
                        </span>
                    </div>
                    <div style="margin-top: 4px; white-space: pre-wrap;">
                        {content}
                    </div>
                </div>
            </div>
            """
            self.chat_display.append(message_html)
        self.chat_display.moveCursor(QTextCursor.End)

    def highlight_code_blocks(self, text):
        def replacement(match):
            language = match.group(1) or "text"
            code = match.group(2).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Preserve indentation by replacing spaces with non-breaking spaces
            code = code.replace(" ", "&nbsp;")
            return (
                f'<div style="background: #263238; color: #f8f8f2; padding: 8px; border-radius: 4px; '
                f'margin: 5px 0; font-family: Consolas, monospace; white-space: pre;">'
                f'<span style="color: #7fdbff;">{language}</span><br>{code}</div>'
            )
        # Split text into parts and apply formatting only to code blocks
        parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
        for i, part in enumerate(parts):
            if part.startswith('```') and part.endswith('```'):
                parts[i] = re.sub(r'```(\w+)?\n(.*?)```', replacement, part, flags=re.DOTALL)
        return ''.join(parts)

    def handle_response_chunk(self, content, timestamp):
        for char in content:
            self.typing_queue.append((char, timestamp))
        if not self.typing_timer.isActive():
            self.typing_timer.start(self.typing_delay)

    def process_typing_queue(self):
        if self.typing_queue:
            char, timestamp = self.typing_queue.popleft()
            if self.message_history and self.message_history[-1]["role"] == "assistant":
                self.message_history[-1]["content"] += char
                self.message_history[-1]["timestamp"] = timestamp
            else:
                self.message_history.append({
                    "role": "assistant",
                    "content": char,
                    "raw": char,
                    "timestamp": timestamp
                })
            self.render_messages()

    def handle_response_end(self, elapsed_time):
        self.typing_timer.stop()
        self.send_btn.setEnabled(True)
        self.update_status(f"Response generated in {elapsed_time:.2f}s")

    def handle_error(self, error_msg):
        self.append_message("system", f"<span style='color:red;'>{error_msg}</span>")
        self.send_btn.setEnabled(True)
        self.update_status("Error occurred - check logs")

    def stop_generation(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.update_status("Generation stopped")
            self.send_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
