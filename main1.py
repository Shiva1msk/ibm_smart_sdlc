# SmartSDLC - AI-Enhanced Software Development Lifecycle
# A comprehensive toolkit for automating code reviews, documentation, and bug detection

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import re
import ast
import json
from datetime import datetime
import base64
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go

# Simulated AI models (in production, replace with actual IBM Granite LLM/Hugging Face models)
class AICodeReviewer:
    """AI-powered code review assistant"""
    
    def __init__(self):
        self.coding_standards = {
            'python': {
                'max_line_length': 88,
                'naming_convention': 'snake_case',
                'docstring_required': True
            }
        }
        self.common_issues = [
            'unused_imports', 'long_functions', 'complex_conditions',
            'missing_docstrings', 'hardcoded_values', 'security_issues'
        ]
    
    def analyze_code(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """Analyze code for issues and provide suggestions"""
        issues = []
        suggestions = []
        
        lines = code.split('\n')
        
        # Check line length
        for i, line in enumerate(lines, 1):
            if len(line) > self.coding_standards[language]['max_line_length']:
                issues.append({
                    'type': 'style',
                    'severity': 'medium',
                    'line': i,
                    'message': f'Line too long ({len(line)} > {self.coding_standards[language]["max_line_length"]})',
                    'suggestion': 'Consider breaking long lines for better readability'
                })
        
        # Check for common Python issues
        if language == 'python':
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if len(node.body) > 20:
                            issues.append({
                                'type': 'complexity',
                                'severity': 'high',
                                'line': node.lineno,
                                'message': f'Function "{node.name}" is too long ({len(node.body)} statements)',
                                'suggestion': 'Consider breaking this function into smaller, more focused functions'
                            })
                        
                        if not ast.get_docstring(node):
                            issues.append({
                                'type': 'documentation',
                                'severity': 'medium',
                                'line': node.lineno,
                                'message': f'Function "{node.name}" missing docstring',
                                'suggestion': 'Add a docstring to explain the function purpose and parameters'
                            })
            except SyntaxError as e:
                issues.append({
                    'type': 'syntax',
                    'severity': 'critical',
                    'line': e.lineno or 0,
                    'message': f'Syntax error: {e.msg}',
                    'suggestion': 'Fix the syntax error before proceeding'
                })
        
        # Check for hardcoded values
        hardcoded_patterns = [r'password\s*=\s*["\'].*["\']', r'api_key\s*=\s*["\'].*["\']']
        for pattern in hardcoded_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    'type': 'security',
                    'severity': 'critical',
                    'line': line_num,
                    'message': 'Hardcoded sensitive information detected',
                    'suggestion': 'Use environment variables or secure configuration files'
                })
        
        return {
            'issues': issues,
            'summary': {
                'total_issues': len(issues),
                'critical': len([i for i in issues if i['severity'] == 'critical']),
                'high': len([i for i in issues if i['severity'] == 'high']),
                'medium': len([i for i in issues if i['severity'] == 'medium']),
                'low': len([i for i in issues if i['severity'] == 'low'])
            }
        }

class DocumentationGenerator:
    """AI-powered documentation generator"""
    
    def generate_documentation(self, code: str, language: str = 'python') -> str:
        """Generate documentation from code"""
        doc_sections = []
        
        if language == 'python':
            try:
                tree = ast.parse(code)
                
                # Extract classes
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                if classes:
                    doc_sections.append("## Classes\n")
                    for cls in classes:
                        doc_sections.append(f"### {cls.name}")
                        docstring = ast.get_docstring(cls)
                        if docstring:
                            doc_sections.append(f"{docstring}\n")
                        else:
                            doc_sections.append(f"Class definition for {cls.name}\n")
                
                # Extract functions
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                if functions:
                    doc_sections.append("\n## Functions\n")
                    for func in functions:
                        doc_sections.append(f"### {func.name}")
                        docstring = ast.get_docstring(func)
                        if docstring:
                            doc_sections.append(f"{docstring}")
                        else:
                            doc_sections.append(f"Function: {func.name}")
                        
                        # Extract parameters
                        if func.args.args:
                            doc_sections.append("\n**Parameters:**")
                            for arg in func.args.args:
                                if arg.arg != 'self':
                                    doc_sections.append(f"- `{arg.arg}`: Parameter description needed")
                        doc_sections.append("\n")
                
            except SyntaxError:
                doc_sections.append("## Code Documentation\n")
                doc_sections.append("Unable to parse code for automatic documentation generation due to syntax errors.\n")
        
        return '\n'.join(doc_sections) if doc_sections else "No documentation generated."

class BugDetector:
    """Early bug detection system"""
    
    def __init__(self):
        self.bug_patterns = {
            'null_pointer': [r'\.(\w+)\s*\(\s*\)\s*\.', r'(\w+)\.(\w+)\s*=\s*None'],
            'resource_leak': [r'open\s*\(.*\)\s*(?!.*with)', r'\.connect\s*\(.*\)\s*(?!.*close)'],
            'infinite_loop': [r'while\s+True\s*:', r'for.*while.*:'],
            'sql_injection': [r'query\s*=.*\+.*input', r'execute\s*\(.*\+.*input'],
            'buffer_overflow': [r'strcpy\s*\(', r'gets\s*\('],
            'race_condition': [r'threading\.Thread.*shared_var', r'multiprocessing.*global']
        }
    
    def detect_bugs(self, code: str, language: str = 'python') -> List[Dict[str, Any]]:
        """Detect potential bugs in code"""
        bugs = []
        
        for bug_type, patterns in self.bug_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    bugs.append({
                        'type': bug_type,
                        'severity': self._get_severity(bug_type),
                        'line': line_num,
                        'pattern': match.group(),
                        'description': self._get_bug_description(bug_type),
                        'suggestion': self._get_bug_suggestion(bug_type)
                    })
        
        return bugs
    
    def _get_severity(self, bug_type: str) -> str:
        severity_map = {
            'null_pointer': 'high',
            'resource_leak': 'medium',
            'infinite_loop': 'critical',
            'sql_injection': 'critical',
            'buffer_overflow': 'critical',
            'race_condition': 'high'
        }
        return severity_map.get(bug_type, 'medium')
    
    def _get_bug_description(self, bug_type: str) -> str:
        descriptions = {
            'null_pointer': 'Potential null pointer dereference',
            'resource_leak': 'Resource may not be properly closed',
            'infinite_loop': 'Potential infinite loop detected',
            'sql_injection': 'Potential SQL injection vulnerability',
            'buffer_overflow': 'Potential buffer overflow vulnerability',
            'race_condition': 'Potential race condition in concurrent code'
        }
        return descriptions.get(bug_type, 'Unknown bug pattern')
    
    def _get_bug_suggestion(self, bug_type: str) -> str:
        suggestions = {
            'null_pointer': 'Add null checks before dereferencing',
            'resource_leak': 'Use context managers (with statement) for resource management',
            'infinite_loop': 'Add proper exit conditions to loops',
            'sql_injection': 'Use parameterized queries or prepared statements',
            'buffer_overflow': 'Use safe string functions with bounds checking',
            'race_condition': 'Use proper synchronization mechanisms'
        }
        return suggestions.get(bug_type, 'Review code for potential issues')

class SmartSDLCDashboard:
    """Main dashboard for SmartSDLC system"""
    
    def __init__(self):
        self.code_reviewer = AICodeReviewer()
        self.doc_generator = DocumentationGenerator()
        self.bug_detector = BugDetector()
        
        # Initialize session state
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'code_samples' not in st.session_state:
            st.session_state.code_samples = self._load_sample_codes()
    
    def _load_sample_codes(self) -> Dict[str, str]:
        """Load sample codes for demonstration"""
        return {
            "Sample Python Function": '''def calculate_user_score(user_data):
    # Missing docstring
    score = 0
    if user_data != None:  # Potential null pointer issue
        for item in user_data:
            score += item['points']
            if score > 1000000000000000000000:  # Hardcoded value
                break
    return score

def process_database_query(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"  # SQL injection risk
    # Resource leak - no proper connection handling
    connection = database.connect()
    result = connection.execute(query)
    return result''',
            
            "Sample Class Definition": '''class UserManager:
    def __init__(self):
        self.users = []
        self.api_key = "sk-1234567890abcdef"  # Hardcoded API key
    
    def add_user(self, name, email):  # Missing docstring
        user = {'name': name, 'email': email}
        self.users.append(user)
        
    def find_user_by_email(self, email):
        # Long function that should be broken down
        for user in self.users:
            if user['email'] == email:
                if user['status'] == 'active' and user['verified'] == True and user['premium'] == True and user['age'] > 18:
                    return user
        return None
        
    def infinite_process(self):
        while True:  # Potential infinite loop
            self.process_data()''',
            
            "Sample Configuration": '''import os
import threading

# Global variable with potential race condition
shared_counter = 0

def increment_counter():
    global shared_counter
    for i in range(1000):
        shared_counter += 1  # Race condition

class ConfigManager:
    def __init__(self):
        self.password = "admin123"  # Hardcoded password
        self.db_connection = None
        
    def connect_to_database(self):
        self.db_connection = open_connection()  # Resource leak
        
    def process_file(self, filename):
        file_handle = open(filename, 'r')  # Resource leak - no with statement
        content = file_handle.read()
        return content'''
        }
    
    def run(self):
        """Main dashboard interface"""
        st.set_page_config(
            page_title="SmartSDLC - AI-Enhanced SDLC",
            page_icon="🚀",
            layout="wide"
        )
        
        # Header
        st.title("🚀 SmartSDLC - AI-Enhanced Software Development Lifecycle")
        st.markdown("**Automated Code Review • Documentation Generation • Bug Detection**")
        st.divider()
        
        # Sidebar
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # Language selection
            language = st.selectbox("Programming Language", ["python", "javascript", "java", "cpp"])
            
            # Analysis options
            st.subheader("Analysis Options")
            enable_code_review = st.checkbox("Code Review", value=True)
            enable_documentation = st.checkbox("Auto Documentation", value=True)
            enable_bug_detection = st.checkbox("Bug Detection", value=True)
            
            # Sample code selector
            st.subheader("Quick Start")
            selected_sample = st.selectbox("Load Sample Code", 
                                         ["Custom Code"] + list(st.session_state.code_samples.keys()))
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📝 Code Input")
            
            # Code input
            if selected_sample != "Custom Code":
                default_code = st.session_state.code_samples[selected_sample]
            else:
                default_code = ""
            
            code_input = st.text_area(
                "Enter your code here:",
                value=default_code,
                height=400,
                placeholder="Paste your code here or select a sample from the sidebar..."
            )
            
            # File upload option
            uploaded_file = st.file_uploader("Or upload a code file", type=['py', 'js', 'java', 'cpp', 'txt'])
            if uploaded_file:
                code_input = str(uploaded_file.read(), "utf-8")
                st.text_area("Uploaded code:", value=code_input, height=200, disabled=True)
            
            # Analyze button
            if st.button("🔍 Analyze Code", type="primary", use_container_width=True):
                if code_input.strip():
                    self._analyze_code(code_input, language, enable_code_review, 
                                     enable_documentation, enable_bug_detection)
                else:
                    st.error("Please enter some code to analyze!")
        
        with col2:
            st.header("📊 Analysis Results")
            
            if st.session_state.analysis_history:
                latest_analysis = st.session_state.analysis_history[-1]
                self._display_analysis_results(latest_analysis)
            else:
                st.info("👈 Enter code and click 'Analyze Code' to see results here")
        
        # History section
        if st.session_state.analysis_history:
            st.header("📈 Analysis History")
            self._display_analysis_history()
    
    def _analyze_code(self, code: str, language: str, enable_review: bool, 
                     enable_docs: bool, enable_bugs: bool):
        """Perform comprehensive code analysis"""
        
        with st.spinner("🤖 AI is analyzing your code..."):
            analysis_result = {
                'timestamp': datetime.now(),
                'language': language,
                'code_length': len(code),
                'lines_of_code': len(code.split('\n'))
            }
            
            # Code Review
            if enable_review:
                review_result = self.code_reviewer.analyze_code(code, language)
                analysis_result['code_review'] = review_result
            
            # Documentation Generation
            if enable_docs:
                documentation = self.doc_generator.generate_documentation(code, language)
                analysis_result['documentation'] = documentation
            
            # Bug Detection
            if enable_bugs:
                bugs = self.bug_detector.detect_bugs(code, language)
                analysis_result['bugs'] = bugs
            
            # Store analysis
            st.session_state.analysis_history.append(analysis_result)
            
        st.success("✅ Analysis completed!")
    
    def _display_analysis_results(self, analysis: Dict[str, Any]):
        """Display analysis results in tabs"""
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Lines of Code", analysis['lines_of_code'])
        
        with col2:
            if 'code_review' in analysis:
                total_issues = analysis['code_review']['summary']['total_issues']
                st.metric("Total Issues", total_issues, 
                         delta=f"-{total_issues}" if total_issues > 0 else "Clean")
        
        with col3:
            if 'bugs' in analysis:
                bug_count = len(analysis['bugs'])
                critical_bugs = len([b for b in analysis['bugs'] if b['severity'] == 'critical'])
                st.metric("Potential Bugs", bug_count, 
                         delta=f"{critical_bugs} critical" if critical_bugs > 0 else "")
        
        with col4:
            if 'documentation' in analysis:
                doc_length = len(analysis['documentation'])
                st.metric("Documentation", f"{doc_length} chars", 
                         delta="Generated" if doc_length > 0 else "")
        
        # Detailed results in tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Code Review", "📚 Documentation", "🐛 Bug Detection"])
        
        with tab1:
            if 'code_review' in analysis:
                self._display_code_review(analysis['code_review'])
            else:
                st.info("Code review was not enabled for this analysis.")
        
        with tab2:
            if 'documentation' in analysis:
                self._display_documentation(analysis['documentation'])
            else:
                st.info("Documentation generation was not enabled for this analysis.")
        
        with tab3:
            if 'bugs' in analysis:
                self._display_bug_detection(analysis['bugs'])
            else:
                st.info("Bug detection was not enabled for this analysis.")
    
    def _display_code_review(self, review_result: Dict[str, Any]):
        """Display code review results"""
        
        # Summary
        summary = review_result['summary']
        if summary['total_issues'] == 0:
            st.success("🎉 No issues found! Your code follows best practices.")
            return
        
        # Issues breakdown
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Critical", summary['critical'], delta="🔴" if summary['critical'] > 0 else "")
        with col2:
            st.metric("High", summary['high'], delta="🟠" if summary['high'] > 0 else "")
        with col3:
            st.metric("Medium", summary['medium'], delta="🟡" if summary['medium'] > 0 else "")
        with col4:
            st.metric("Low", summary['low'], delta="🟢" if summary['low'] > 0 else "")
        
        # Issues list
        st.subheader("Issues Found")
        issues_df = pd.DataFrame(review_result['issues'])
        if not issues_df.empty:
            # Color coding for severity
            def highlight_severity(row):
                colors = {
                    'critical': 'background-color: #ffebee',
                    'high': 'background-color: #fff3e0',
                    'medium': 'background-color: #fffde7',
                    'low': 'background-color: #f1f8e9'
                }
                return [colors.get(row['severity'], '')] * len(row)
            
            styled_df = issues_df.style.apply(highlight_severity, axis=1)
            st.dataframe(styled_df, use_container_width=True)
    
    def _display_documentation(self, documentation: str):
        """Display generated documentation"""
        
        if documentation and documentation.strip():
            st.subheader("Generated Documentation")
            st.markdown(documentation)
            
            # Download option
            st.download_button(
                label="📥 Download Documentation",
                data=documentation,
                file_name=f"documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("No documentation could be generated from the provided code.")
    
    def _display_bug_detection(self, bugs: List[Dict[str, Any]]):
        """Display bug detection results"""
        
        if not bugs:
            st.success("🎉 No potential bugs detected!")
            return
        
        # Bugs by severity
        bug_df = pd.DataFrame(bugs)
        severity_counts = bug_df['severity'].value_counts()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Severity pie chart
            fig = px.pie(values=severity_counts.values, names=severity_counts.index,
                        title="Bugs by Severity", color_discrete_map={
                            'critical': '#f44336',
                            'high': '#ff9800',
                            'medium': '#ffeb3b',
                            'low': '#4caf50'
                        })
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bug types
            type_counts = bug_df['type'].value_counts()
            fig = px.bar(x=type_counts.index, y=type_counts.values,
                        title="Bug Types", labels={'x': 'Bug Type', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed bugs list
        st.subheader("Detected Issues")
        for i, bug in enumerate(bugs):
            with st.expander(f"🐛 {bug['type'].replace('_', ' ').title()} - Line {bug['line']} ({bug['severity']})"):
                st.write(f"**Description:** {bug['description']}")
                st.write(f"**Pattern:** `{bug['pattern']}`")
                st.write(f"**Suggestion:** {bug['suggestion']}")
    
    def _display_analysis_history(self):
        """Display analysis history and trends"""
        
        if len(st.session_state.analysis_history) < 2:
            st.info("Run more analyses to see trends and history.")
            return
        
        # Prepare data for visualization
        history_data = []
        for i, analysis in enumerate(st.session_state.analysis_history):
            row = {
                'Analysis': i + 1,
                'Timestamp': analysis['timestamp'],
                'Lines of Code': analysis['lines_of_code'],
                'Total Issues': analysis.get('code_review', {}).get('summary', {}).get('total_issues', 0),
                'Bugs Found': len(analysis.get('bugs', [])),
                'Language': analysis['language']
            }
            history_data.append(row)
        
        history_df = pd.DataFrame(history_data)
        
        # Trends visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(history_df, x='Analysis', y=['Total Issues', 'Bugs Found'],
                         title="Code Quality Trends", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(history_df, x='Lines of Code', y='Total Issues',
                           color='Language', title="Issues vs Code Size")
            st.plotly_chart(fig, use_container_width=True)
        
        # History table
        st.subheader("Analysis History")
        st.dataframe(history_df, use_container_width=True)

# Main application
def main():
    dashboard = SmartSDLCDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()