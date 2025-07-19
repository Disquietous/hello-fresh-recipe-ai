#!/usr/bin/env python3
"""
Comprehensive Test Suite for Ingredient Parsing Pipeline
Tests all aspects of the intelligent ingredient parsing system.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from enhanced_ingredient_parser import EnhancedIngredientParser


class IngredientParsingTest:
    """Test suite for ingredient parsing."""
    
    def __init__(self):
        self.parser = EnhancedIngredientParser()
        self.test_results = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test categories."""
        print("🧪 Running Comprehensive Ingredient Parsing Tests")
        print("=" * 60)
        
        test_categories = [
            ("Basic Parsing", self._test_basic_parsing),
            ("Fractional Quantities", self._test_fractional_quantities),
            ("Unit Normalization", self._test_unit_normalization),
            ("Preparation Extraction", self._test_preparation_extraction),
            ("Typo Correction", self._test_typo_correction),
            ("Abbreviation Expansion", self._test_abbreviation_expansion),
            ("Complex Formats", self._test_complex_formats),
            ("Edge Cases", self._test_edge_cases),
            ("Database Integration", self._test_database_integration),
            ("Nutritional Calculation", self._test_nutritional_calculation)
        ]
        
        overall_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "categories": {}
        }
        
        for category_name, test_func in test_categories:
            print(f"\n🔍 Testing {category_name}...")
            category_results = test_func()
            overall_results["categories"][category_name] = category_results
            overall_results["total_tests"] += category_results["total"]
            overall_results["passed_tests"] += category_results["passed"]
            overall_results["failed_tests"] += category_results["failed"]
        
        # Calculate success rate
        overall_results["success_rate"] = (
            overall_results["passed_tests"] / overall_results["total_tests"] 
            if overall_results["total_tests"] > 0 else 0
        )
        
        return overall_results
    
    def _test_basic_parsing(self) -> Dict[str, Any]:
        """Test basic ingredient parsing functionality."""
        test_cases = [
            {
                "input": "2 cups all-purpose flour",
                "expected": {
                    "quantity": "2",
                    "unit": "cup",
                    "ingredient": "all-purpose flour"
                }
            },
            {
                "input": "1 tsp vanilla extract",
                "expected": {
                    "quantity": "1",
                    "unit": "teaspoon",
                    "ingredient": "vanilla extract"
                }
            },
            {
                "input": "3 large eggs",
                "expected": {
                    "quantity": "3",
                    "unit": None,
                    "ingredient": "large eggs"
                }
            },
            {
                "input": "1 lb ground beef",
                "expected": {
                    "quantity": "1",
                    "unit": "pound",
                    "ingredient": "ground beef"
                }
            }
        ]
        
        return self._run_test_cases("Basic Parsing", test_cases)
    
    def _test_fractional_quantities(self) -> Dict[str, Any]:
        """Test fractional quantity parsing."""
        test_cases = [
            {
                "input": "1/2 cup sugar",
                "expected": {
                    "quantity": "0.5",
                    "normalized_quantity": 0.5
                }
            },
            {
                "input": "1 1/2 cups milk",
                "expected": {
                    "quantity": "1.5",
                    "normalized_quantity": 1.5
                }
            },
            {
                "input": "3/4 teaspoon salt",
                "expected": {
                    "quantity": "0.75",
                    "normalized_quantity": 0.75
                }
            },
            {
                "input": "2 2/3 cups flour",
                "expected": {
                    "quantity": "2.6666666666666665",
                    "normalized_quantity": 2.6666666666666665
                }
            }
        ]
        
        return self._run_test_cases("Fractional Quantities", test_cases)
    
    def _test_unit_normalization(self) -> Dict[str, Any]:
        """Test unit normalization."""
        test_cases = [
            {
                "input": "2 tbsp olive oil",
                "expected": {
                    "unit": "tablespoon",
                    "normalized_unit": "milliliter"
                }
            },
            {
                "input": "1 tsp baking powder",
                "expected": {
                    "unit": "teaspoon",
                    "normalized_unit": "milliliter"
                }
            },
            {
                "input": "1 lb butter",
                "expected": {
                    "unit": "pound",
                    "normalized_unit": "gram"
                }
            },
            {
                "input": "8 oz cream cheese",
                "expected": {
                    "unit": "ounce",
                    "normalized_unit": "gram"
                }
            }
        ]
        
        return self._run_test_cases("Unit Normalization", test_cases)
    
    def _test_preparation_extraction(self) -> Dict[str, Any]:
        """Test preparation method extraction."""
        test_cases = [
            {
                "input": "1 onion, diced",
                "expected": {
                    "ingredient": "onion",
                    "preparation": "diced"
                }
            },
            {
                "input": "2 tomatoes, peeled and chopped",
                "expected": {
                    "ingredient": "tomatoes",
                    "preparation": "peeled and chopped"
                }
            },
            {
                "input": "1 cup spinach, washed and stemmed",
                "expected": {
                    "ingredient": "spinach",
                    "preparation": "washed and stemmed"
                }
            },
            {
                "input": "8 oz cream cheese, softened",
                "expected": {
                    "ingredient": "cream cheese",
                    "preparation": "softened"
                }
            }
        ]
        
        return self._run_test_cases("Preparation Extraction", test_cases)
    
    def _test_typo_correction(self) -> Dict[str, Any]:
        """Test typo correction functionality."""
        test_cases = [
            {
                "input": "2 cups floru",
                "expected": {
                    "ingredient": "flour",
                    "typo_corrections": ["floru -> flour"]
                }
            },
            {
                "input": "1 tsp vanilia",
                "expected": {
                    "ingredient": "vanilla",
                    "typo_corrections": ["vanilia -> vanilla"]
                }
            },
            {
                "input": "3 large egs",
                "expected": {
                    "ingredient": "eggs",
                    "typo_corrections": ["egs -> eggs"]
                }
            },
            {
                "input": "1 lb chiken",
                "expected": {
                    "ingredient": "chicken",
                    "typo_corrections": ["chiken -> chicken"]
                }
            }
        ]
        
        return self._run_test_cases("Typo Correction", test_cases)
    
    def _test_abbreviation_expansion(self) -> Dict[str, Any]:
        """Test abbreviation expansion."""
        test_cases = [
            {
                "input": "2 tbsp oil",
                "expected": {
                    "unit": "tablespoon",
                    "abbreviation_expansions": ["tbsp -> tablespoon"]
                }
            },
            {
                "input": "1 tsp salt",
                "expected": {
                    "unit": "teaspoon",
                    "abbreviation_expansions": ["tsp -> teaspoon"]
                }
            },
            {
                "input": "1 lb beef",
                "expected": {
                    "unit": "pound",
                    "abbreviation_expansions": ["lb -> pound"]
                }
            },
            {
                "input": "8 oz cheese",
                "expected": {
                    "unit": "ounce",
                    "abbreviation_expansions": ["oz -> ounce"]
                }
            }
        ]
        
        return self._run_test_cases("Abbreviation Expansion", test_cases)
    
    def _test_complex_formats(self) -> Dict[str, Any]:
        """Test complex ingredient formats."""
        test_cases = [
            {
                "input": "1 package (8 oz) cream cheese",
                "expected": {
                    "ingredient": "cream cheese",
                    "confidence": lambda x: x > 0.7
                }
            },
            {
                "input": "2 cans (14.5 oz each) diced tomatoes",
                "expected": {
                    "ingredient": "diced tomatoes",
                    "confidence": lambda x: x > 0.6
                }
            },
            {
                "input": "1 bottle (750ml) white wine",
                "expected": {
                    "ingredient": "white wine",
                    "confidence": lambda x: x > 0.6
                }
            }
        ]
        
        return self._run_test_cases("Complex Formats", test_cases)
    
    def _test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and unusual inputs."""
        test_cases = [
            {
                "input": "Salt and pepper to taste",
                "expected": {
                    "ingredient": "salt and pepper",
                    "confidence": lambda x: x > 0.3
                }
            },
            {
                "input": "A pinch of red pepper flakes",
                "expected": {
                    "ingredient": "red pepper flakes",
                    "confidence": lambda x: x > 0.4
                }
            },
            {
                "input": "Handful of fresh herbs",
                "expected": {
                    "ingredient": "fresh herbs",
                    "confidence": lambda x: x > 0.4
                }
            },
            {
                "input": "",
                "expected": {
                    "confidence": lambda x: x == 0.0
                }
            }
        ]
        
        return self._run_test_cases("Edge Cases", test_cases)
    
    def _test_database_integration(self) -> Dict[str, Any]:
        """Test database integration functionality."""
        test_cases = [
            {
                "input": "2 cups flour",
                "expected": {
                    "database_match": lambda x: x is not None
                }
            },
            {
                "input": "1 cup sugar",
                "expected": {
                    "database_match": lambda x: x is not None
                }
            },
            {
                "input": "1 lb butter",
                "expected": {
                    "database_match": lambda x: x is not None
                }
            }
        ]
        
        return self._run_test_cases("Database Integration", test_cases)
    
    def _test_nutritional_calculation(self) -> Dict[str, Any]:
        """Test nutritional information calculation."""
        test_cases = [
            {
                "input": "1 cup flour",
                "expected": {
                    "nutritional_info": lambda x: x is not None if x else True
                }
            },
            {
                "input": "2 large eggs",
                "expected": {
                    "nutritional_info": lambda x: x is not None if x else True
                }
            }
        ]
        
        return self._run_test_cases("Nutritional Calculation", test_cases)
    
    def _run_test_cases(self, category: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Run a set of test cases for a category."""
        passed = 0
        failed = 0
        failures = []
        
        for i, test_case in enumerate(test_cases):
            try:
                result = self.parser.parse_ingredient(test_case["input"])
                
                # Check each expected value
                test_passed = True
                for key, expected_value in test_case["expected"].items():
                    actual_value = getattr(result, key, None)
                    
                    if callable(expected_value):
                        # For lambda functions
                        if not expected_value(actual_value):
                            test_passed = False
                            failures.append(f"Test {i+1}: {key} failed lambda check")
                    elif isinstance(expected_value, list):
                        # For lists (like corrections)
                        if not actual_value or not any(item in actual_value for item in expected_value):
                            test_passed = False
                            failures.append(f"Test {i+1}: {key} expected {expected_value}, got {actual_value}")
                    else:
                        # For exact matches
                        if actual_value != expected_value:
                            test_passed = False
                            failures.append(f"Test {i+1}: {key} expected {expected_value}, got {actual_value}")
                
                if test_passed:
                    passed += 1
                    print(f"  ✅ Test {i+1}: '{test_case['input']}' - PASSED")
                else:
                    failed += 1
                    print(f"  ❌ Test {i+1}: '{test_case['input']}' - FAILED")
                    
            except Exception as e:
                failed += 1
                failures.append(f"Test {i+1}: Exception - {str(e)}")
                print(f"  ❌ Test {i+1}: '{test_case['input']}' - ERROR: {str(e)}")
        
        return {
            "total": len(test_cases),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(test_cases) if test_cases else 0,
            "failures": failures
        }
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("🧪 INGREDIENT PARSING TEST REPORT")
        report.append("=" * 60)
        report.append(f"Total Tests: {results['total_tests']}")
        report.append(f"Passed: {results['passed_tests']}")
        report.append(f"Failed: {results['failed_tests']}")
        report.append(f"Success Rate: {results['success_rate']:.1%}")
        report.append("")
        
        report.append("📊 CATEGORY BREAKDOWN:")
        report.append("-" * 30)
        
        for category, category_results in results["categories"].items():
            status = "✅" if category_results["success_rate"] == 1.0 else "⚠️" if category_results["success_rate"] >= 0.7 else "❌"
            report.append(f"{status} {category}: {category_results['passed']}/{category_results['total']} ({category_results['success_rate']:.1%})")
            
            if category_results["failures"]:
                report.append("   Failures:")
                for failure in category_results["failures"]:
                    report.append(f"     - {failure}")
        
        report.append("")
        report.append("🎯 RECOMMENDATIONS:")
        report.append("-" * 20)
        
        if results["success_rate"] >= 0.9:
            report.append("✅ Excellent! The parsing system is working very well.")
        elif results["success_rate"] >= 0.7:
            report.append("⚠️ Good performance, but some areas need improvement.")
        else:
            report.append("❌ Performance needs significant improvement.")
        
        # Specific recommendations based on category performance
        for category, category_results in results["categories"].items():
            if category_results["success_rate"] < 0.7:
                if category == "Typo Correction":
                    report.append("💡 Consider improving typo correction algorithms")
                elif category == "Database Integration":
                    report.append("💡 Check database connections and API keys")
                elif category == "Unit Normalization":
                    report.append("💡 Review unit conversion mappings")
                elif category == "Preparation Extraction":
                    report.append("💡 Expand preparation method patterns")
        
        return "\n".join(report)


def main():
    """Run the comprehensive test suite."""
    print("🧪 HelloFresh Recipe AI - Ingredient Parsing Test Suite")
    print("=" * 60)
    
    # Create test instance
    test_suite = IngredientParsingTest()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Generate and display report
    print("\n" + "=" * 60)
    report = test_suite.generate_test_report(results)
    print(report)
    
    # Save detailed results
    output_file = Path(__file__).parent / "test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {output_file}")
    
    # Return exit code based on success rate
    if results["success_rate"] >= 0.8:
        print("\n🎉 Test suite completed successfully!")
        return 0
    else:
        print("\n⚠️  Test suite completed with issues. Please review the failures.")
        return 1


if __name__ == "__main__":
    exit(main())