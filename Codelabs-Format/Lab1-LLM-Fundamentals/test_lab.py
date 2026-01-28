#!/usr/bin/env python3
"""
Comprehensive Test Suite for Lab1-LLM-Fundamentals
Tests all exercises without requiring API keys
"""

import sys
import os
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ️  {text}{RESET}")

class LabTester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def test_python_version(self):
        """Test Python version"""
        print_header("Testing Python Environment")

        version = sys.version_info
        print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")

        if version.major == 3 and version.minor >= 8:
            print_success("Python version is compatible (3.8+)")
            self.passed += 1
        else:
            print_error("Python 3.8+ required")
            self.failed += 1

    def test_dependencies(self):
        """Test required packages"""
        print_header("Testing Dependencies")

        required_packages = [
            ('openai', 'openai'),
            ('anthropic', 'anthropic'),
            ('google.generativeai', 'google-generativeai'),
            ('tiktoken', 'tiktoken'),
            ('dotenv', 'python-dotenv')
        ]

        for import_name, package_name in required_packages:
            try:
                __import__(import_name)
                print_success(f"{package_name} installed")
                self.passed += 1
            except ImportError:
                print_error(f"{package_name} not installed")
                print_info(f"   Install with: pip install {package_name}")
                self.failed += 1

    def test_file_syntax(self):
        """Test Python file syntax"""
        print_header("Testing File Syntax")

        solutions_dir = Path(__file__).parent / "solutions"
        python_files = list(solutions_dir.glob("*.py"))

        if not python_files:
            print_error("No Python files found in solutions/")
            self.failed += 1
            return

        for py_file in sorted(python_files):
            try:
                with open(py_file, 'r') as f:
                    compile(f.read(), py_file.name, 'exec')
                print_success(f"{py_file.name}")
                self.passed += 1
            except SyntaxError as e:
                print_error(f"{py_file.name}: Syntax error on line {e.lineno}")
                self.failed += 1

    def test_env_setup(self):
        """Test .env configuration"""
        print_header("Testing Environment Configuration")

        env_file = Path(__file__).parent / ".env"
        env_example = Path(__file__).parent / ".env.example"

        if env_file.exists():
            print_success(".env file exists")
            self.passed += 1

            # Check if it has content
            with open(env_file, 'r') as f:
                content = f.read().strip()
                if content:
                    print_success(".env file has content")
                    self.passed += 1
                else:
                    print_warning(".env file is empty")
                    self.warnings += 1
        else:
            print_warning(".env file not found (required for API calls)")
            self.warnings += 1

        if env_example.exists():
            print_success(".env.example exists")
            self.passed += 1
        else:
            print_warning(".env.example not found (recommended)")
            self.warnings += 1

    def test_notebook_exists(self):
        """Test notebook file"""
        print_header("Testing Notebook")

        notebook = Path(__file__).parent / "Lab1-LLM-Fundamentals.ipynb"

        if notebook.exists():
            print_success("Lab1-LLM-Fundamentals.ipynb exists")
            self.passed += 1

            # Check file size (should be > 1KB for real content)
            size = notebook.stat().st_size
            if size > 1024:
                print_success(f"Notebook has content ({size:,} bytes)")
                self.passed += 1
            else:
                print_warning("Notebook seems empty")
                self.warnings += 1
        else:
            print_error("Notebook file not found")
            self.failed += 1

    def test_solutions_structure(self):
        """Test solutions directory structure"""
        print_header("Testing Solutions Structure")

        solutions_dir = Path(__file__).parent / "solutions"

        if not solutions_dir.exists():
            print_error("solutions/ directory not found")
            self.failed += 1
            return

        print_success("solutions/ directory exists")
        self.passed += 1

        expected_files = [
            'test_setup.py',
            'exercise1_openai.py',
            'exercise1_claude.py',
            'exercise1_gemini.py',
            'exercise2_tokens.py',
            'exercise3_temperature.py',
            'exercise4_parameters.py',
            'exercise5_streaming.py',
            'exercise6_cost_calculator.py',
            'exercise7_chatbot.py',
            'capstone_supportgenie_v01.py'
        ]

        for filename in expected_files:
            filepath = solutions_dir / filename
            if filepath.exists():
                print_success(f"{filename}")
                self.passed += 1
            else:
                print_error(f"{filename} missing")
                self.failed += 1

    def test_imports_in_files(self):
        """Test that files have correct imports"""
        print_header("Testing File Imports")

        solutions_dir = Path(__file__).parent / "solutions"

        import_tests = {
            'exercise1_openai.py': ['openai', 'os', 'dotenv'],
            'exercise1_claude.py': ['anthropic', 'os', 'dotenv'],
            'exercise2_tokens.py': ['tiktoken'],
            'exercise5_streaming.py': ['openai'],
        }

        for filename, required_imports in import_tests.items():
            filepath = solutions_dir / filename
            if not filepath.exists():
                continue

            with open(filepath, 'r') as f:
                content = f.read()

            missing = []
            for imp in required_imports:
                if f'import {imp}' not in content:
                    missing.append(imp)

            if not missing:
                print_success(f"{filename} has all required imports")
                self.passed += 1
            else:
                print_error(f"{filename} missing imports: {', '.join(missing)}")
                self.failed += 1

    def test_documentation(self):
        """Test documentation files"""
        print_header("Testing Documentation")

        doc_files = [
            ('README.md', 'solutions/'),
            ('codelab.md', ''),
            ('lab.md', ''),
            ('learning.md', '')
        ]

        base_dir = Path(__file__).parent

        for filename, subdir in doc_files:
            filepath = base_dir / subdir / filename if subdir else base_dir / filename
            if filepath.exists():
                print_success(f"{subdir}{filename}")
                self.passed += 1
            else:
                print_warning(f"{subdir}{filename} not found")
                self.warnings += 1

    def test_gitignore(self):
        """Test .gitignore configuration"""
        print_header("Testing Git Configuration")

        gitignore = Path(__file__).parent / ".gitignore"

        if gitignore.exists():
            print_success(".gitignore exists")
            self.passed += 1

            with open(gitignore, 'r') as f:
                content = f.read()

            critical_entries = ['.env', '__pycache__', '*.pyc']
            for entry in critical_entries:
                if entry in content:
                    print_success(f".gitignore contains '{entry}'")
                    self.passed += 1
                else:
                    print_warning(f".gitignore missing '{entry}'")
                    self.warnings += 1
        else:
            print_warning(".gitignore not found")
            self.warnings += 1

    def print_summary(self):
        """Print test summary"""
        print_header("Test Summary")

        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0

        print(f"\n{GREEN}Passed: {self.passed}{RESET}")
        print(f"{RED}Failed: {self.failed}{RESET}")
        print(f"{YELLOW}Warnings: {self.warnings}{RESET}")
        print(f"\n{BLUE}Pass Rate: {pass_rate:.1f}%{RESET}\n")

        if self.failed == 0:
            print(f"{GREEN}{'🎉 ALL TESTS PASSED! 🎉':^60}{RESET}")
            print(f"{GREEN}{'Your lab is ready to use!':^60}{RESET}\n")
        elif self.failed <= 5:
            print(f"{YELLOW}{'⚠️  MINOR ISSUES FOUND':^60}{RESET}")
            print(f"{YELLOW}{'Fix the errors above':^60}{RESET}\n")
        else:
            print(f"{RED}{'❌ CRITICAL ISSUES FOUND':^60}{RESET}")
            print(f"{RED}{'Please fix the errors above':^60}{RESET}\n")

    def run_all_tests(self):
        """Run all tests"""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}{'Lab1-LLM-Fundamentals Test Suite':^60}{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")

        self.test_python_version()
        self.test_dependencies()
        self.test_file_syntax()
        self.test_solutions_structure()
        self.test_imports_in_files()
        self.test_env_setup()
        self.test_notebook_exists()
        self.test_documentation()
        self.test_gitignore()

        self.print_summary()

        return self.failed == 0

def main():
    """Main test runner"""
    tester = LabTester()
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
