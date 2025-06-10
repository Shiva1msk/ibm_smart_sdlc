# Paste the complete smartsdlc_app.py code here
# SmartSDLC with IBM Granite Integration for Google Colab
# Install required packages first by running:
# !pip install transformers torch accelerate gradio pandas plotly numpy streamlit

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import re
import ast
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

class GraniteAICodeReviewer:
    """AI-powered code review assistant using IBM Granite model"""
    
    def __init__(self):
        self.model_name = "ibm-granite/granite-3.3-2b-instruct"
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.is_loaded = False
        
        self.coding_standards = {
            'python': {
                'max_line_length': 88,
                'naming_convention': 'snake_case',
                'docstring_required': True
            },
            'javascript': {
                'max_line_length': 120,
                'naming_convention': 'camelCase',
                'docstring_required': False
            }
        }
    
    def load_model(self):
        """Load the IBM Granite model"""
        try:
            print("Loading IBM Granite model... This may take a few minutes.")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings for Colab
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=2048,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            self.is_loaded = True
            print("✅ IBM Granite model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def analyze_code_with_granite(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """Analyze code using IBM Granite model"""
        if not self.is_loaded:
            return {"error": "Model not loaded. Please load the model first."}
        
        # Create prompt for code review
        prompt = f"""<|system|>
You are an expert code reviewer. Analyze the following {language} code and provide a detailed review focusing on:
1. Code quality and best practices
2. Potential bugs and security issues
3. Performance improvements
4. Maintainability concerns
5. Documentation and readability

Provide your response in a structured format with specific line numbers where applicable.

<|user|>
Please review this {language} code:

```{language}
{code}
```

<|assistant|>"""

        try:
            # Generate review using Granite
            response = self.generator(
                prompt,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            granite_review = response[0]['generated_text'].split('<|assistant|>')[-1].strip()
            
            # Also perform traditional static analysis
            static_issues = self._static_analysis(code, language)
            
            return {
                'granite_review': granite_review,
                'static_analysis': static_issues,
                'summary': {
                    'ai_review_length': len(granite_review),
                    'static_issues_count': len(static_issues),
                    'total_analysis': 'completed'
                }
            }
            
        except Exception as e:
            return {"error": f"Error during analysis: {str(e)}"}
    
    def _static_analysis(self, code: str, language: str = 'python') -> List[Dict[str, Any]]:
        """Perform static code analysis"""
        issues = []
        lines = code.split('\n')
        
        # Check line length
        for i, line in enumerate(lines, 1):
            if len(line) > self.coding_standards.get(language, {}).get('max_line_length', 120):
                issues.append({
                    'type': 'style',
                    'severity': 'medium',
                    'line': i,
                    'message': f'Line too long ({len(line)} characters)',
                    'suggestion': 'Consider breaking long lines for better readability'
                })
        
        if language == 'python':
            issues.extend(self._analyze_python_code(code))
        
        return issues
    
    def _analyze_python_code(self, code: str) -> List[Dict[str, Any]]:
        """Analyze Python-specific issues"""
        issues = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check function length
                    if len(node.body) > 20:
                        issues.append({
                            'type': 'complexity',
                            'severity': 'high',
                            'line': node.lineno,
                            'message': f'Function "{node.name}" is too long ({len(node.body)} statements)',
                            'suggestion': 'Consider breaking this function into smaller functions'
                        })
                    
                    # Check for missing docstrings
                    if not ast.get_docstring(node):
                        issues.append({
                            'type': 'documentation',
                            'severity': 'medium',
                            'line': node.lineno,
                            'message': f'Function "{node.name}" missing docstring',
                            'suggestion': 'Add a docstring to explain the function purpose'
                        })
        
        except SyntaxError as e:
            issues.append({
                'type': 'syntax',
                'severity': 'critical',
                'line': e.lineno or 0,
                'message': f'Syntax error: {e.msg}',
                'suggestion': 'Fix the syntax error before proceeding'
            })
        
        # Check for security patterns
        security_patterns = [
            (r'password\s*=\s*["\'].*["\']', 'Hardcoded password detected'),
            (r'api_key\s*=\s*["\'].*["\']', 'Hardcoded API key detected'),
            (r'exec\s*\(', 'Use of exec() function - potential security risk'),
            (r'eval\s*\(', 'Use of eval() function - potential security risk')
        ]
        
        for pattern, message in security_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    'type': 'security',
                    'severity': 'critical',
                    'line': line_num,
                    'message': message,
                    'suggestion': 'Use secure alternatives or environment variables'
                })
        
        return issues

class GraniteDocumentationGenerator:
    """AI-powered documentation generator using IBM Granite"""
    
    def __init__(self, granite_reviewer):
        self.granite_reviewer = granite_reviewer
    
    def generate_documentation(self, code: str, language: str = 'python') -> str:
        """Generate documentation using IBM Granite model"""
        if not self.granite_reviewer.is_loaded:
            return "Error: Model not loaded. Please load the model first."
        
        prompt = f"""<|system|>
You are a technical documentation expert. Generate comprehensive documentation for the provided code including:
1. Overview and purpose
2. Function/class descriptions
3. Parameters and return values
4. Usage examples
5. Dependencies and requirements

Format the response in clean Markdown.

<|user|>
Generate documentation for this {language} code:

```{language}
{code}
```

<|assistant|>"""

        try:
            response = self.granite_reviewer.generator(
                prompt,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True
            )
            
            documentation = response[0]['generated_text'].split('<|assistant|>')[-1].strip()
            return documentation
            
        except Exception as e:
            return f"Error generating documentation: {str(e)}"

class GraniteBugDetector:
    """AI-powered bug detection using IBM Granite"""
    
    def __init__(self, granite_reviewer):
        self.granite_reviewer = granite_reviewer
    
    def detect_bugs(self, code: str, language: str = 'python') -> List[Dict[str, Any]]:
        """Detect bugs using IBM Granite model"""
        if not self.granite_reviewer.is_loaded:
            return [{"error": "Model not loaded"}]
        
        prompt = f"""<|system|>
You are an expert bug detection specialist. Analyze the following code for potential bugs, vulnerabilities, and runtime issues. Focus on:
1. Logic errors
2. Memory leaks
3. Security vulnerabilities
4. Performance issues
5. Edge cases that might cause failures

Provide specific line numbers and detailed explanations for each issue found.

<|user|>
Analyze this {language} code for bugs:

```{language}
{code}
```

<|assistant|>"""

        try:
            response = self.granite_reviewer.generator(
                prompt,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True
            )
            
            bug_analysis = response[0]['generated_text'].split('<|assistant|>')[-1].strip()
            
            # Parse the analysis into structured format
            bugs = self._parse_bug_analysis(bug_analysis)
            return bugs
            
        except Exception as e:
            return [{"error": f"Error detecting bugs: {str(e)}"}]
    
    def _parse_bug_analysis(self, analysis: str) -> List[Dict[str, Any]]:
        """Parse the AI analysis into structured bug reports"""
        bugs = []
        
        # Simple parsing - in production, you might want more sophisticated parsing
        lines = analysis.split('\n')
        current_bug = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_bug:
                    bugs.append(current_bug)
                    current_bug = {}
                continue
            
            # Look for common patterns in bug descriptions
            if any(keyword in line.lower() for keyword in ['bug', 'issue', 'problem', 'error', 'vulnerability']):
                current_bug = {
                    'type': 'ai_detected',
                    'severity': 'medium',
                    'description': line,
                    'line': self._extract_line_number(line),
                    'suggestion': 'Review and fix as suggested by AI analysis'
                }
        
        if current_bug:
            bugs.append(current_bug)
        
        return bugs if bugs else [{'type': 'analysis', 'description': analysis, 'severity': 'info', 'line': 0}]
    
    def _extract_line_number(self, text: str) -> int:
        """Extract line number from text"""
        import re
        match = re.search(r'line\s+(\d+)', text, re.IGNORECASE)
        return int(match.group(1)) if match else 0

class SmartSDLCGradioApp:
    """Gradio interface for SmartSDLC with IBM Granite integration"""
    
    def __init__(self):
        self.granite_reviewer = GraniteAICodeReviewer()
        self.doc_generator = GraniteDocumentationGenerator(self.granite_reviewer)
        self.bug_detector = GraniteBugDetector(self.granite_reviewer)
        self.analysis_history = []
        
        # Sample codes
        self.sample_codes = {
            "Sample Python Function": '''def calculate_user_score(user_data):
    # Missing docstring
    score = 0
    if user_data != None:  # Should use 'is not None'
        for item in user_data:
            score += item['points']
            if score > 1000000000000000000000:  # Magic number
                break
    return score

def process_query(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"  # SQL injection
    return execute_query(query)''',
            
            "Sample Class": '''class UserManager:
    def __init__(self):
        self.api_key = "sk-1234567890"  # Hardcoded key
        self.users = []
    
    def add_user(self, name, email):
        user = {"name": name, "email": email}
        self.users.append(user)
        
    def find_user(self, email):
        for user in self.users:
            if user["email"] == email:
                return user
        return None'''
        }
    
    def load_model_interface(self):
        """Load the IBM Granite model"""
        success = self.granite_reviewer.load_model()
        if success:
            return "✅ IBM Granite model loaded successfully! You can now use all AI-powered features."
        else:
            return "❌ Failed to load model. Please check the error messages above."
    
    def analyze_code_interface(self, code, language, enable_review, enable_docs, enable_bugs):
        """Main analysis interface"""
        if not code.strip():
            return "Please enter some code to analyze.", "", "", ""
        
        if not self.granite_reviewer.is_loaded:
            return "Please load the IBM Granite model first using the 'Load Model' button.", "", "", ""
        
        results = {"timestamp": datetime.now(), "language": language}
        
        # Code Review
        review_result = ""
        if enable_review:
            review_data = self.granite_reviewer.analyze_code_with_granite(code, language)
            if "error" not in review_data:
                review_result = f"## AI Code Review (IBM Granite)\n\n{review_data['granite_review']}\n\n"
                if review_data['static_analysis']:
                    review_result += "## Static Analysis Issues\n\n"
                    for issue in review_data['static_analysis']:
                        review_result += f"- **Line {issue['line']}** ({issue['severity']}): {issue['message']}\n"
                        review_result += f"  *Suggestion: {issue['suggestion']}*\n\n"
            else:
                review_result = f"Error in code review: {review_data['error']}"
        
        # Documentation
        documentation = ""
        if enable_docs:
            documentation = self.doc_generator.generate_documentation(code, language)
        
        # Bug Detection
        bug_report = ""
        if enable_bugs:
            bugs = self.bug_detector.detect_bugs(code, language)
            if bugs and "error" not in bugs[0]:
                bug_report = "## Bug Detection Results\n\n"
                for bug in bugs:
                    if "error" not in bug:
                        line_info = f" (Line {bug['line']})" if bug.get('line', 0) > 0 else ""
                        bug_report += f"### {bug['type'].replace('_', ' ').title()}{line_info}\n"
                        bug_report += f"**Severity:** {bug.get('severity', 'Unknown')}\n\n"
                        bug_report += f"{bug.get('description', 'No description')}\n\n"
                        if bug.get('suggestion'):
                            bug_report += f"*Suggestion: {bug['suggestion']}*\n\n"
            else:
                bug_report = "No bugs detected or error in analysis."
        
        # Summary
        summary = f"""## Analysis Summary
- **Language:** {language}
- **Lines of Code:** {len(code.split())}
- **Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Features Used:** {', '.join([f for f, enabled in [('Code Review', enable_review), ('Documentation', enable_docs), ('Bug Detection', enable_bugs)] if enabled])}
"""
        
        # Store in history
        self.analysis_history.append({
            'code': code[:200] + "..." if len(code) > 200 else code,
            'language': language,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': [f for f, enabled in [('Review', enable_review), ('Docs', enable_docs), ('Bugs', enable_bugs)] if enabled]
        })
        
        return summary, review_result, documentation, bug_report
    
    def get_analysis_history(self):
        """Get analysis history as a dataframe"""
        if not self.analysis_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.analysis_history)
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
        
        with gr.Blocks(css=css, title="SmartSDLC - AI Code Analysis") as app:
            
            # Header
            gr.HTML("""
            <div class="header">
                <h1>🚀 SmartSDLC - AI-Enhanced Code Analysis</h1>
                <p>Powered by IBM Granite AI Model | Code Review • Documentation • Bug Detection</p>
            </div>
            """)
            
            # Model loading section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 🤖 Model Setup")
                    load_btn = gr.Button("🔄 Load IBM Granite Model", variant="primary", size="lg")
                    model_status = gr.Textbox(
                        label="Model Status",
                        value="Click 'Load IBM Granite Model' to initialize the AI system.",
                        interactive=False
                    )
            
            load_btn.click(self.load_model_interface, outputs=model_status)
            
            gr.Markdown("---")
            
            # Main interface
            with gr.Row():
                # Input column
                with gr.Column(scale=1):
                    gr.Markdown("## 📝 Code Input")
                    
                    # Sample code selector
                    sample_dropdown = gr.Dropdown(
                        choices=["Custom Code"] + list(self.sample_codes.keys()),
                        value="Custom Code",
                        label="Quick Start - Load Sample Code"
                    )
                    
                    # Code input
                    code_input = gr.Code(
                        label="Enter your code here",
                        language="python",
                        lines=15,
                    )
                    
                    # Language selection
                    language_select = gr.Dropdown(
                        choices=["python", "javascript", "java", "cpp"],
                        value="python",
                        label="Programming Language"
                    )
                    
                    # Analysis options
                    gr.Markdown("### Analysis Options")
                    with gr.Row():
                        enable_review = gr.Checkbox(label="AI Code Review", value=True)
                        enable_docs = gr.Checkbox(label="Auto Documentation", value=True)
                        enable_bugs = gr.Checkbox(label="Bug Detection", value=True)
                    
                    # Analyze button
                    analyze_btn = gr.Button("🔍 Analyze Code", variant="primary", size="lg")
                
                # Results column
                with gr.Column(scale=1):
                    gr.Markdown("## 📊 Analysis Results")
                    
                    with gr.Tabs():
                        with gr.TabItem("📋 Summary"):
                            summary_output = gr.Markdown()
                        
                        with gr.TabItem("🔍 Code Review"):
                            review_output = gr.Markdown()
                        
                        with gr.TabItem("📚 Documentation"):
                            docs_output = gr.Markdown()
                        
                        with gr.TabItem("🐛 Bug Detection"):
                            bugs_output = gr.Markdown()
            
            # History section
            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📈 Analysis History")
                    history_btn = gr.Button("🔄 Refresh History")
                    history_display = gr.Dataframe()
            
            # Event handlers
            def load_sample_code(sample_name):
                if sample_name in self.sample_codes:
                    return self.sample_codes[sample_name]
                return ""
            
            sample_dropdown.change(
                load_sample_code,
                inputs=sample_dropdown,
                outputs=code_input
            )
            
            analyze_btn.click(
                self.analyze_code_interface,
                inputs=[code_input, language_select, enable_review, enable_docs, enable_bugs],
                outputs=[summary_output, review_output, docs_output, bugs_output]
            )
            
            history_btn.click(
                self.get_analysis_history,
                outputs=history_display
            )
        
        return app

def main():
    """Main function to run the application"""
    print("🚀 Starting SmartSDLC with IBM Granite Integration...")
    print("📦 Make sure you have installed the required packages:")
    print("   !pip install transformers torch accelerate gradio pandas plotly numpy")
    print()
    
    # Create and launch the application
    app_instance = SmartSDLCGradioApp()
    app = app_instance.create_interface()
    
    # Launch with public sharing for Colab
    app.launch(
        share=True,  # Creates a public link for Colab
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()

