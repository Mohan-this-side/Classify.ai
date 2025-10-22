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
    
    # Dangerous operations that should be blocked
    DANGEROUS_IMPORTS = {
        'os.system', 'subprocess', 'eval', 'exec', 'compile',
        '__import__', 'open', 'file', 'input', 'raw_input'
    }
    
    # Dangerous built-in functions
    DANGEROUS_BUILTINS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'input'
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
        
        # 2. Security scanning
        security_valid, sec_issues = self._scan_security(code)
        if not security_valid:
            security_issues.extend(sec_issues)
        
        # 3. Import validation
        import_valid, import_issues = self._validate_imports(code)
        if not import_valid:
            errors.extend(import_issues)
        
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
    
    def _validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error at line {e.lineno}: {e.msg}"]
        except Exception as e:
            return False, [f"Parse error: {str(e)}"]
    
    def _scan_security(self, code: str) -> Tuple[bool, List[str]]:
        """Scan for security issues"""
        issues = []
        
        # Check for dangerous imports
        for dangerous in self.DANGEROUS_IMPORTS:
            if dangerous in code:
                issues.append(f"Dangerous operation detected: {dangerous}")
        
        # Check for dangerous built-ins
        for builtin in self.DANGEROUS_BUILTINS:
            pattern = rf'\b{builtin}\s*\('
            if re.search(pattern, code):
                issues.append(f"Dangerous built-in function: {builtin}")
        
        # Check for file system operations
        if re.search(r'open\s*\(', code):
            issues.append("File system access detected")
        
        # Check for network operations
        network_patterns = ['socket', 'urllib', 'requests', 'http']
        for pattern in network_patterns:
            if pattern in code.lower():
                issues.append(f"Network operation detected: {pattern}")
        
        # Check for system commands
        if 'os.' in code or 'subprocess' in code:
            issues.append("System command execution detected")
        
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


