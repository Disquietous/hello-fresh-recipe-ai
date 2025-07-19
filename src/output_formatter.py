#!/usr/bin/env python3
"""
Output formatter for recipe OCR pipeline results.
Formats structured ingredient data into various output formats with confidence scores.
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import time
from datetime import datetime
import logging


@dataclass
class FormattedOutput:
    """Formatted output result."""
    format_type: str
    content: Union[str, Dict, List]
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = None


class OutputFormatter:
    """Formatter for OCR pipeline results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize output formatter.
        
        Args:
            config: Formatter configuration
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default formatter configuration."""
        return {
            "include_metadata": True,
            "include_confidence_scores": True,
            "include_bounding_boxes": False,
            "include_processing_details": False,
            "round_coordinates": True,
            "round_confidences": 3,
            "sort_by_confidence": True,
            "filter_low_confidence": True,
            "min_confidence_threshold": 0.3,
            "timestamp_format": "%Y-%m-%d %H:%M:%S"
        }
    
    def format_results(self, pipeline_result: Any, output_format: str = "json") -> FormattedOutput:
        """
        Format pipeline results into specified format.
        
        Args:
            pipeline_result: Pipeline result object
            output_format: Output format ('json', 'csv', 'xml', 'yaml', 'txt')
            
        Returns:
            Formatted output
        """
        # Extract and structure data
        structured_data = self._structure_data(pipeline_result)
        
        # Format according to requested type
        if output_format.lower() == "json":
            return self._format_json(structured_data)
        elif output_format.lower() == "csv":
            return self._format_csv(structured_data)
        elif output_format.lower() == "xml":
            return self._format_xml(structured_data)
        elif output_format.lower() == "yaml":
            return self._format_yaml(structured_data)
        elif output_format.lower() == "txt":
            return self._format_text(structured_data)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _structure_data(self, pipeline_result: Any) -> Dict[str, Any]:
        """Structure pipeline result data for formatting."""
        
        # Extract ingredients with filtering and sorting
        ingredients = self._process_ingredients(pipeline_result.ingredients)
        
        # Create structured output
        structured = {
            "recipe_extraction_result": {
                "source_image": pipeline_result.source_image_path,
                "extraction_timestamp": datetime.now().strftime(self.config["timestamp_format"]),
                "processing_summary": {
                    "processing_time_seconds": round(pipeline_result.processing_time, 2),
                    "text_regions_detected": pipeline_result.text_regions_detected,
                    "ingredients_extracted": len(ingredients),
                    "extraction_success_rate": len(ingredients) / max(pipeline_result.text_regions_detected, 1)
                },
                "ingredients": ingredients
            }
        }
        
        # Add optional sections based on configuration
        if self.config["include_metadata"]:
            structured["recipe_extraction_result"]["metadata"] = self._format_metadata(pipeline_result)
        
        if self.config["include_confidence_scores"]:
            structured["recipe_extraction_result"]["confidence_summary"] = self._format_confidence_summary(pipeline_result)
        
        if self.config["include_processing_details"]:
            structured["recipe_extraction_result"]["processing_details"] = self._format_processing_details(pipeline_result)
        
        if self.config["include_bounding_boxes"]:
            structured["recipe_extraction_result"]["text_regions"] = self._format_text_regions(pipeline_result.text_regions)
        
        return structured
    
    def _process_ingredients(self, raw_ingredients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean ingredient data."""
        ingredients = []
        
        for ingredient in raw_ingredients:
            # Filter by confidence if enabled
            confidence = ingredient.get("confidence", 0)
            if self.config["filter_low_confidence"] and confidence < self.config["min_confidence_threshold"]:
                continue
            
            # Create clean ingredient entry
            processed_ingredient = {
                "ingredient_name": ingredient.get("ingredient_name", "").strip(),
                "quantity": ingredient.get("quantity"),
                "unit": ingredient.get("unit"),
                "preparation": ingredient.get("preparation"),
                "raw_text": ingredient.get("raw_text", "").strip()
            }
            
            # Add confidence scores if enabled
            if self.config["include_confidence_scores"]:
                processed_ingredient["confidence"] = round(confidence, self.config["round_confidences"])
            
            # Remove empty fields
            processed_ingredient = {k: v for k, v in processed_ingredient.items() if v is not None and v != ""}
            
            ingredients.append(processed_ingredient)
        
        # Sort by confidence if enabled
        if self.config["sort_by_confidence"] and self.config["include_confidence_scores"]:
            ingredients.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return ingredients
    
    def _format_metadata(self, pipeline_result: Any) -> Dict[str, Any]:
        """Format metadata information."""
        return {
            "pipeline_version": pipeline_result.pipeline_metadata.get("pipeline_version", "unknown"),
            "text_detection_model": pipeline_result.pipeline_metadata.get("text_detector_model", "unknown"),
            "ocr_engines": pipeline_result.pipeline_metadata.get("ocr_engines", []),
            "text_cleaning_enabled": pipeline_result.pipeline_metadata.get("text_cleaning_enabled", False),
            "processing_timestamp": pipeline_result.pipeline_metadata.get("processing_timestamp")
        }
    
    def _format_confidence_summary(self, pipeline_result: Any) -> Dict[str, Any]:
        """Format confidence summary."""
        summary = pipeline_result.confidence_summary.copy()
        
        # Round confidence values
        for key, value in summary.items():
            if isinstance(value, float):
                summary[key] = round(value, self.config["round_confidences"])
        
        return summary
    
    def _format_processing_details(self, pipeline_result: Any) -> Dict[str, Any]:
        """Format processing details."""
        return {
            "error_log": pipeline_result.error_log,
            "pipeline_config": pipeline_result.pipeline_config if hasattr(pipeline_result, 'pipeline_config') else {},
            "text_regions_processed": len(pipeline_result.text_regions),
            "successful_ocr_extractions": sum(1 for r in pipeline_result.text_regions 
                                            if r.get("ocr_result", {}).get("confidence", 0) > 0.1)
        }
    
    def _format_text_regions(self, text_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format text regions data."""
        formatted_regions = []
        
        for region in text_regions:
            formatted_region = {
                "region_id": region.get("region_id"),
                "class_name": region.get("class_name"),
                "detection_confidence": round(region.get("detection_confidence", 0), self.config["round_confidences"]),
                "bounding_box": self._format_bounding_box(region.get("bbox", [])),
                "ocr_text": region.get("ocr_result", {}).get("raw_text", ""),
                "ocr_confidence": round(region.get("ocr_result", {}).get("confidence", 0), self.config["round_confidences"])
            }
            
            formatted_regions.append(formatted_region)
        
        return formatted_regions
    
    def _format_bounding_box(self, bbox: List[int]) -> Dict[str, int]:
        """Format bounding box coordinates."""
        if len(bbox) != 4:
            return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        
        coords = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
        
        if self.config["round_coordinates"]:
            coords = {k: int(v) for k, v in coords.items()}
        
        return coords
    
    def _format_json(self, data: Dict[str, Any]) -> FormattedOutput:
        """Format as JSON."""
        json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        
        return FormattedOutput(
            format_type="json",
            content=json_str,
            metadata={"size_bytes": len(json_str.encode('utf-8'))}
        )
    
    def _format_csv(self, data: Dict[str, Any]) -> FormattedOutput:
        """Format as CSV."""
        import io
        
        output = io.StringIO()
        ingredients = data["recipe_extraction_result"]["ingredients"]
        
        if not ingredients:
            return FormattedOutput(format_type="csv", content="")
        
        # Define CSV columns
        fieldnames = ["ingredient_name", "quantity", "unit", "preparation", "raw_text"]
        if self.config["include_confidence_scores"]:
            fieldnames.append("confidence")
        
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for ingredient in ingredients:
            writer.writerow(ingredient)
        
        csv_content = output.getvalue()
        output.close()
        
        return FormattedOutput(
            format_type="csv",
            content=csv_content,
            metadata={"rows": len(ingredients) + 1}  # +1 for header
        )
    
    def _format_xml(self, data: Dict[str, Any]) -> FormattedOutput:
        """Format as XML."""
        root = ET.Element("recipe_extraction_result")
        
        result_data = data["recipe_extraction_result"]
        
        # Add metadata
        metadata_elem = ET.SubElement(root, "metadata")
        ET.SubElement(metadata_elem, "source_image").text = result_data["source_image"]
        ET.SubElement(metadata_elem, "extraction_timestamp").text = result_data["extraction_timestamp"]
        
        # Add processing summary
        summary_elem = ET.SubElement(root, "processing_summary")
        for key, value in result_data["processing_summary"].items():
            ET.SubElement(summary_elem, key).text = str(value)
        
        # Add ingredients
        ingredients_elem = ET.SubElement(root, "ingredients")
        for ingredient in result_data["ingredients"]:
            ingredient_elem = ET.SubElement(ingredients_elem, "ingredient")
            
            for key, value in ingredient.items():
                if value is not None:
                    elem = ET.SubElement(ingredient_elem, key)
                    elem.text = str(value)
        
        # Add optional sections
        if "confidence_summary" in result_data:
            confidence_elem = ET.SubElement(root, "confidence_summary")
            for key, value in result_data["confidence_summary"].items():
                ET.SubElement(confidence_elem, key).text = str(value)
        
        # Convert to string
        ET.indent(root, space="  ")
        xml_str = ET.tostring(root, encoding='unicode')
        
        return FormattedOutput(
            format_type="xml",
            content=xml_str,
            metadata={"elements": len(list(root.iter()))}
        )
    
    def _format_yaml(self, data: Dict[str, Any]) -> FormattedOutput:
        """Format as YAML."""
        try:
            import yaml
            yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            return FormattedOutput(
                format_type="yaml",
                content=yaml_str,
                metadata={"size_bytes": len(yaml_str.encode('utf-8'))}
            )
        except ImportError:
            # Fallback to simple YAML-like format
            yaml_str = self._simple_yaml_format(data)
            return FormattedOutput(
                format_type="yaml",
                content=yaml_str,
                metadata={"format": "simple", "size_bytes": len(yaml_str.encode('utf-8'))}
            )
    
    def _simple_yaml_format(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Simple YAML-like formatting without PyYAML dependency."""
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._simple_yaml_format(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  -")
                        for k, v in item.items():
                            lines.append(f"{prefix}    {k}: {v}")
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return "\n".join(lines)
    
    def _format_text(self, data: Dict[str, Any]) -> FormattedOutput:
        """Format as human-readable text."""
        lines = []
        result_data = data["recipe_extraction_result"]
        
        # Header
        lines.append("RECIPE INGREDIENT EXTRACTION RESULTS")
        lines.append("=" * 40)
        lines.append("")
        
        # Basic info
        lines.append(f"Source Image: {Path(result_data['source_image']).name}")
        lines.append(f"Processed: {result_data['extraction_timestamp']}")
        lines.append(f"Processing Time: {result_data['processing_summary']['processing_time_seconds']}s")
        lines.append("")
        
        # Summary
        summary = result_data['processing_summary']
        lines.append("EXTRACTION SUMMARY:")
        lines.append(f"  Text regions detected: {summary['text_regions_detected']}")
        lines.append(f"  Ingredients extracted: {summary['ingredients_extracted']}")
        lines.append(f"  Success rate: {summary['extraction_success_rate']:.1%}")
        lines.append("")
        
        # Ingredients
        ingredients = result_data['ingredients']
        if ingredients:
            lines.append("EXTRACTED INGREDIENTS:")
            lines.append("-" * 20)
            
            for i, ingredient in enumerate(ingredients, 1):
                # Format ingredient line
                parts = []
                if ingredient.get('quantity'):
                    parts.append(str(ingredient['quantity']))
                if ingredient.get('unit'):
                    parts.append(str(ingredient['unit']))
                if ingredient.get('ingredient_name'):
                    parts.append(str(ingredient['ingredient_name']))
                if ingredient.get('preparation'):
                    parts.append(f"({ingredient['preparation']})")
                
                ingredient_text = " ".join(parts) if parts else ingredient.get('raw_text', 'Unknown')
                
                if self.config["include_confidence_scores"] and 'confidence' in ingredient:
                    lines.append(f"{i:2d}. {ingredient_text} [confidence: {ingredient['confidence']:.2f}]")
                else:
                    lines.append(f"{i:2d}. {ingredient_text}")
        else:
            lines.append("No ingredients extracted.")
        
        # Confidence summary
        if self.config["include_confidence_scores"] and "confidence_summary" in result_data:
            lines.append("")
            lines.append("CONFIDENCE SUMMARY:")
            summary = result_data["confidence_summary"]
            lines.append(f"  Average detection confidence: {summary.get('avg_detection_confidence', 0):.2f}")
            lines.append(f"  Average OCR confidence: {summary.get('avg_ocr_confidence', 0):.2f}")
            lines.append(f"  Average parsing confidence: {summary.get('avg_parsing_confidence', 0):.2f}")
        
        text_content = "\n".join(lines)
        
        return FormattedOutput(
            format_type="text",
            content=text_content,
            metadata={"lines": len(lines), "characters": len(text_content)}
        )
    
    def save_formatted_output(self, formatted_output: FormattedOutput, 
                            output_path: str) -> str:
        """
        Save formatted output to file.
        
        Args:
            formatted_output: Formatted output object
            output_path: Output file path
            
        Returns:
            Actual saved file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure correct file extension
        if not output_file.suffix:
            output_file = output_file.with_suffix(f".{formatted_output.format_type}")
        
        # Write content
        if isinstance(formatted_output.content, str):
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_output.content)
        else:
            # For binary content or complex objects
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_output.content, f, indent=2, default=str)
        
        self.logger.info(f"Saved {formatted_output.format_type} output to {output_file}")
        return str(output_file)
    
    def create_summary_report(self, results: List[Any], output_path: str) -> str:
        """
        Create summary report for multiple pipeline results.
        
        Args:
            results: List of pipeline results
            output_path: Output file path for summary
            
        Returns:
            Path to saved summary report
        """
        if not results:
            return ""
        
        # Aggregate statistics
        total_images = len(results)
        total_processing_time = sum(r.processing_time for r in results)
        total_regions = sum(r.text_regions_detected for r in results)
        total_ingredients = sum(r.ingredients_extracted for r in results)
        successful_extractions = sum(1 for r in results if r.ingredients_extracted > 0)
        
        # Create summary data
        summary_data = {
            "batch_summary": {
                "total_images_processed": total_images,
                "successful_extractions": successful_extractions,
                "success_rate": successful_extractions / total_images if total_images > 0 else 0,
                "total_processing_time": round(total_processing_time, 2),
                "average_processing_time": round(total_processing_time / total_images, 2) if total_images > 0 else 0,
                "total_text_regions_detected": total_regions,
                "total_ingredients_extracted": total_ingredients,
                "average_ingredients_per_image": round(total_ingredients / total_images, 2) if total_images > 0 else 0,
                "processing_timestamp": datetime.now().strftime(self.config["timestamp_format"])
            },
            "individual_results": []
        }
        
        # Add individual results
        for i, result in enumerate(results):
            summary_data["individual_results"].append({
                "image_index": i + 1,
                "source_image": Path(result.source_image_path).name,
                "processing_time": round(result.processing_time, 2),
                "regions_detected": result.text_regions_detected,
                "ingredients_extracted": result.ingredients_extracted,
                "errors": len(result.error_log)
            })
        
        # Format and save
        formatted_output = self._format_json(summary_data)
        return self.save_formatted_output(formatted_output, output_path)


def main():
    """Example usage of output formatter."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize formatter
    formatter = OutputFormatter()
    
    print("Output Formatter")
    print("================")
    print("Supported formats: JSON, CSV, XML, YAML, TXT")
    print("\nUsage examples:")
    print("1. Format pipeline result:")
    print("   formatted = formatter.format_results(pipeline_result, 'json')")
    print("2. Save formatted output:")
    print("   path = formatter.save_formatted_output(formatted, 'output.json')")
    print("3. Create batch summary:")
    print("   summary_path = formatter.create_summary_report(results, 'summary.json')")


if __name__ == "__main__":
    main()