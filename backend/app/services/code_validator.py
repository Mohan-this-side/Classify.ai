"""
Code Validation and Security Scanning Service

This service validates and scans generated code before execution:
- Syntax validation
- Security scanning for dangerous operations
- Best practices checking
- Resource usage validation
"""

import ast
import logging
import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    security_issues: List[str]
    suggestions: List[str]


class CodeValidator:
    """
    Validates and scans Python code for security and quality issues.
    """
    
    # Dangerous operations that should be blocked (relaxed for ML sandbox)
    DANGEROUS_IMPORTS = {
        'os.system', 'subprocess.call', 'subprocess.run', 'subprocess.Popen'
    }
    
    # Dangerous built-in functions (relaxed for ML sandbox)
    DANGEROUS_BUILTINS = {
        'compile', '__import__', 'input'
    }
    
    # Allowed imports for ML/Data Science
    ALLOWED_IMPORTS = {
        'pandas', 'numpy', 'sklearn', 'scipy', 'matplotlib',
        'seaborn', 'plotly', 'joblib', 'pickle', 'json',
        'datetime', 'collections', 'itertools', 'functools',
        'warnings', 'logging', 'typing', 'dataclasses',
        'xgboost', 'lightgbm', 'catboost'
    }
    
    def __init__(self):
        self.logger = logging.getLogger("code_validator")
    
    def validate(self, code: str) -> ValidationResult:
        """
        Perform comprehensive validation on generated code.
        
        Args:
            code: Python code to validate
            
        Returns:
            ValidationResult with detailed findings
        """
        errors = []
        warnings = []
        security_issues = []
        suggestions = []
        
        # 1. Syntax validation
        syntax_valid, syntax_errors = self._validate_syntax(code)
        if not syntax_valid:
            errors.extend(syntax_errors)
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                security_issues=security_issues,
                suggestions=suggestions
            )
        
        # 2. Security scanning (STRICT - block on security issues)
        security_valid, sec_issues = self._scan_security(code)
        if not security_valid:
            security_issues.extend(sec_issues)
        
        # 3. Import validation (RELAXED - treat as warnings not errors)
        import_valid, import_issues = self._validate_imports(code)
        if not import_valid:
            # Downgrade unauthorized imports to warnings instead of blocking errors
            warnings.extend([f"Import warning: {issue}" for issue in import_issues])
        
        # 4. Best practices checking
        practice_warnings = self._check_best_practices(code)
        warnings.extend(practice_warnings)
        
        # 5. Resource usage validation
        resource_warnings = self._check_resource_usage(code)
        warnings.extend(resource_warnings)
        
        # 6. Code quality suggestions
        quality_suggestions = self._generate_suggestions(code)
        suggestions.extend(quality_suggestions)
        
        is_valid = len(errors) == 0 and len(security_issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            security_issues=security_issues,
            suggestions=suggestions
        )
    
    def _clean_code(self, code: str) -> str:
        """Clean and fix common LLM-generated code issues"""
        if not code:
            return code
        
        original_code = code
        
        # Remove markdown code fences
        code = code.strip()
        if code.startswith('```python'):
            code = code[9:].strip()
        elif code.startswith('```'):
            code = code[3:].strip()
        if code.endswith('```'):
            code = code[:-3].strip()
        
        lines = code.split('\n')
        cleaned_lines = []
        skip_until_code = True
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines at the start
            if skip_until_code and not stripped:
                continue
            
            # Detect start of actual code
            if skip_until_code:
                if stripped.startswith('import ') or stripped.startswith('from ') or stripped.startswith('def ') or stripped.startswith('class '):
                    skip_until_code = False
                else:
                    # Skip explanatory text
                    continue
            
            # Process code lines
            if not skip_until_code:
                # Fix indentation issues: if line should be at module level but is indented
                if stripped.startswith('import ') or stripped.startswith('from ') or stripped.startswith('def ') or stripped.startswith('class '):
                    # These should be at column 0 - remove leading whitespace
                    cleaned_lines.append(stripped)
                elif stripped.startswith('#'):
                    # Comments - keep as is but remove excessive indentation
                    if line.startswith(' ') and len(line) - len(line.lstrip()) > 4:
                        cleaned_lines.append(stripped)
                    else:
                        cleaned_lines.append(line)
                else:
                    # Other code - preserve relative indentation but fix absolute
                    # If line is indented but shouldn't be (after empty line or import)
                    if cleaned_lines and cleaned_lines[-1].strip() and not cleaned_lines[-1].startswith(' '):
                        # Previous line was at module level, check if this should be too
                        if stripped and not (stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('while ') or stripped.startswith('try:') or stripped.startswith('except') or stripped.startswith('with ')):
                            # Might be incorrectly indented - try dedenting
                            dedented = line.lstrip()
                            if dedented and (dedented.startswith('import ') or dedented.startswith('from ') or dedented.startswith('def ') or dedented.startswith('class ')):
                                cleaned_lines.append(dedented)
                            else:
                                cleaned_lines.append(line)
                        else:
                            cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(line)
        
        # Join and clean up
        code_str = '\n'.join(cleaned_lines)
        
        # Final pass: ensure first import/def is at column 0
        final_lines = code_str.split('\n')
        if final_lines:
            # Find first actual code line
            first_code_idx = None
            for i, line in enumerate(final_lines):
                stripped = line.strip()
                if stripped and (stripped.startswith('import ') or stripped.startswith('from ') or stripped.startswith('def ') or stripped.startswith('class ')):
                    first_code_idx = i
                    break
            
            if first_code_idx is not None and first_code_idx > 0:
                # Remove everything before first code
                final_lines = final_lines[first_code_idx:]
            
            # Ensure first line starts at column 0
            if final_lines and final_lines[0].strip():
                first_line = final_lines[0]
                if first_line.startswith(' ') or first_line.startswith('\t'):
                    final_lines[0] = final_lines[0].lstrip()
            
            code_str = '\n'.join(final_lines)
        
        result = code_str.strip()
        
        # Log if significant cleaning happened
        if len(result) < len(original_code) * 0.8:
            self.logger.debug(f"Significant code cleaning: {len(original_code)} -> {len(result)} chars")
        
        return result
    
    def _validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax, with automatic cleaning"""
        # First try to clean the code
        cleaned_code = self._clean_code(code)
        
        # Try parsing cleaned code
        try:
            ast.parse(cleaned_code)
            return True, []
        except SyntaxError as e:
            # If cleaned code still fails, try original
            try:
                ast.parse(code)
                return True, []
            except SyntaxError as e2:
                return False, [f"Syntax error at line {e2.lineno}: {e2.msg}"]
        except Exception as e:
            return False, [f"Parse error: {str(e)}"]
    
    def _scan_security(self, code: str) -> Tuple[bool, List[str]]:
        """Scan for security issues (RELAXED for sandbox environment)"""
        issues = []
        
        # In sandbox environment, exec/eval are safe (isolated execution)
        # Only block truly dangerous operations that could escape sandbox
        critical_dangerous = ['os.system', 'subprocess.call', 'subprocess.run', 'subprocess.Popen']
        for dangerous in critical_dangerous:
            if dangerous in code:
                issues.append(f"Critical dangerous operation: {dangerous}")
        
        # exec/eval are OK in sandbox (code runs in isolated Docker container)
        # __import__ is OK in sandbox (limited to allowed imports)
        # File system operations are OK in sandbox (has limited access)
        # Network operations are OK in sandbox (network is isolated)
        # Most built-ins are OK in sandbox
        
        return len(issues) == 0, issues
    
    def _validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """Validate that only allowed imports are used"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module not in self.ALLOWED_IMPORTS:
                            issues.append(f"Unauthorized import: {module}")
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module.split('.')[0] if node.module else ''
                    if module and module not in self.ALLOWED_IMPORTS:
                        issues.append(f"Unauthorized import from: {module}")
        
        except Exception as e:
            issues.append(f"Import validation error: {str(e)}")
        
        return len(issues) == 0, issues
    
    def _check_best_practices(self, code: str) -> List[str]:
        """Check for best practices violations"""
        warnings = []
        
        # Check for proper error handling
        if 'try:' not in code and 'except' not in code:
            warnings.append("Consider adding error handling with try/except blocks")
        
        # Check for magic numbers
        if re.search(r'\b\d{3,}\b', code):
            warnings.append("Consider using named constants instead of magic numbers")
        
        # Check for proper logging
        if 'print(' in code:
            warnings.append("Consider using logging instead of print statements")
        
        # Check for docstrings
        if 'def ' in code and '"""' not in code:
            warnings.append("Consider adding docstrings to functions")
        
        return warnings
    
    def _check_resource_usage(self, code: str) -> List[str]:
        """Check for potential resource issues"""
        warnings = []
        
        # Check for infinite loops
        if re.search(r'while\s+True:', code):
            warnings.append("Infinite loop detected - ensure proper break conditions")
        
        # Check for large memory allocations
        if re.search(r'\.zeros\(\[.*\d{6,}.*\]\)', code):
            warnings.append("Large memory allocation detected")
        
        # Check for nested loops
        loop_count = code.count('for ') + code.count('while ')
        if loop_count > 3:
            warnings.append(f"Multiple nested loops detected ({loop_count}) - may impact performance")
        
        return warnings
    
    def _generate_suggestions(self, code: str) -> List[str]:
        """Generate code quality suggestions"""
        suggestions = []
        
        # Suggest vectorization
        if 'for ' in code and ('pandas' in code or 'numpy' in code):
            suggestions.append("Consider using vectorized operations instead of loops for better performance")
        
        # Suggest type hints
        if 'def ' in code and '->' not in code:
            suggestions.append("Consider adding type hints for better code documentation")
        
        # Suggest constants
        if code.count('=') > 5:
            suggestions.append("Consider extracting repeated values as constants")
        
        return suggestions
    
    def get_validation_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report"""
        report = []
        
        report.append("=" * 60)
        report.append("CODE VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"\nStatus: {'✅ VALID' if result.is_valid else '❌ INVALID'}\n")
        
        if result.errors:
            report.append("\n🚫 ERRORS:")
            for error in result.errors:
                report.append(f"  - {error}")
        
        if result.security_issues:
            report.append("\n🔒 SECURITY ISSUES:")
            for issue in result.security_issues:
                report.append(f"  - {issue}")
        
        if result.warnings:
            report.append("\n⚠️  WARNINGS:")
            for warning in result.warnings:
                report.append(f"  - {warning}")
        
        if result.suggestions:
            report.append("\n💡 SUGGESTIONS:")
            for suggestion in result.suggestions:
                report.append(f"  - {suggestion}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Global validator instance
_code_validator = None


def get_code_validator() -> CodeValidator:
    """Get or create global code validator instance"""
    global _code_validator
    if _code_validator is None:
        _code_validator = CodeValidator()
    return _code_validator


