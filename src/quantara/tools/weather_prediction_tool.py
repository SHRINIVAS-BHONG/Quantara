"""
Weather Prediction Tool for LangGraph integration.

This tool wraps the WeatherPredictionAPI for use in the trading agent system,
enabling weather-based market analysis and opportunity detection.
"""

import json
import logging
from typing import Dict, Any, List

from quantara.core.weather_prediction_api import WeatherPredictionAPI, WeatherPrediction

logger = logging.getLogger(__name__)


def tool_result(data: Any) -> str:
    """Format successful tool result."""
    return json.dumps({
        "status": "success",
        "data": data if isinstance(data, (dict, list)) else str(data)
    })


def tool_error(message: str) -> str:
    """Format tool error result."""
    return json.dumps({
        "status": "error",
        "message": message
    })


WEATHER_PREDICTION_SCHEMA = {
    "name": "weather_prediction_fetch",
    "description": "Fetch weather predictions for a location to identify weather-related prediction market opportunities. Returns detailed forecasts including temperature, precipitation, wind, and confidence scores.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location name or coordinates (e.g., 'New York', 'London', '40.7128,-74.0060')"
            },
            "days": {
                "type": "integer",
                "description": "Number of days to forecast (1-14, default: 7)",
                "minimum": 1,
                "maximum": 14
            },
            "include_alerts": {
                "type": "boolean",
                "description": "Include active weather alerts (default: true)"
            }
        },
        "required": ["location"]
    }
}

WEATHER_ALERTS_SCHEMA = {
    "name": "weather_alerts_fetch",
    "description": "Fetch active weather alerts for a location. Useful for identifying high-impact weather events that may affect prediction markets.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location name or coordinates"
            }
        },
        "required": ["location"]
    }
}

WEATHER_HISTORICAL_SCHEMA = {
    "name": "weather_historical_fetch",
    "description": "Fetch historical weather data for a location. Useful for analyzing patterns and validating prediction accuracy.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location name or coordinates"
            },
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format"
            },
            "end_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format"
            }
        },
        "required": ["location", "start_date", "end_date"]
    }
}


class WeatherPredictionTool:
    """Tool wrapper for weather prediction API."""
    
    def __init__(self, primary_source: str = "openweathermap"):
        """Initialize the weather prediction tool."""
        self.api = WeatherPredictionAPI(primary_source=primary_source)
        logger.info(f"Initialized WeatherPredictionTool with source: {primary_source}")
    
    def fetch_forecast(self, args: Dict[str, Any]) -> str:
        """
        Fetch weather forecast for a location.
        
        Args:
            args: Dictionary with 'location' and optional 'days' and 'include_alerts'
            
        Returns:
            JSON string with forecast data
        """
        try:
            location = args.get("location")
            if not location:
                return tool_error("Location is required")
            
            days = args.get("days", 7)
            include_alerts = args.get("include_alerts", True)
            
            # Validate inputs
            if not isinstance(days, int) or not (1 <= days <= 14):
                return tool_error("Days must be an integer between 1 and 14")
            
            logger.info(f"Fetching weather forecast for {location} ({days} days)")
            
            # Get forecast
            predictions = self.api.get_forecast(location, days)
            
            # Convert predictions to dictionaries
            forecast_data = []
            for pred in predictions:
                pred_dict = {
                    "location": pred.location,
                    "date": pred.date,
                    "temperature_high": pred.temperature_high,
                    "temperature_low": pred.temperature_low,
                    "precipitation_probability": pred.precipitation_probability,
                    "wind_speed": pred.wind_speed,
                    "humidity": pred.humidity,
                    "condition": pred.condition,
                    "confidence": pred.confidence,
                    "source": pred.source
                }
                forecast_data.append(pred_dict)
            
            result = {
                "forecast": forecast_data,
                "location": location,
                "days": days,
                "total_predictions": len(forecast_data)
            }
            
            # Add alerts if requested
            if include_alerts:
                try:
                    alerts = self.api.get_alerts(location)
                    result["alerts"] = [
                        {
                            "type": alert.alert_type,
                            "severity": alert.severity,
                            "description": alert.description,
                            "effective_time": alert.effective_time,
                            "expires_time": alert.expires_time
                        }
                        for alert in alerts
                    ]
                except Exception as e:
                    logger.warning(f"Failed to fetch alerts: {e}")
                    result["alerts"] = []
            
            logger.info(f"Successfully fetched forecast for {location}")
            return tool_result(result)
        
        except ValueError as e:
            return tool_error(f"Invalid input: {e}")
        except Exception as e:
            logger.error(f"Forecast fetch failed: {e}")
            return tool_error(f"Failed to fetch forecast: {e}")
    
    def fetch_alerts(self, args: Dict[str, Any]) -> str:
        """
        Fetch weather alerts for a location.
        
        Args:
            args: Dictionary with 'location'
            
        Returns:
            JSON string with alerts data
        """
        try:
            location = args.get("location")
            if not location:
                return tool_error("Location is required")
            
            logger.info(f"Fetching weather alerts for {location}")
            
            alerts = self.api.get_alerts(location)
            
            alerts_data = [
                {
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "description": alert.description,
                    "effective_time": alert.effective_time,
                    "expires_time": alert.expires_time,
                    "source": alert.source
                }
                for alert in alerts
            ]
            
            result = {
                "location": location,
                "alerts": alerts_data,
                "total_alerts": len(alerts_data)
            }
            
            logger.info(f"Successfully fetched {len(alerts)} alerts for {location}")
            return tool_result(result)
        
        except Exception as e:
            logger.error(f"Alert fetch failed: {e}")
            return tool_error(f"Failed to fetch alerts: {e}")
    
    def fetch_historical(self, args: Dict[str, Any]) -> str:
        """
        Fetch historical weather data.
        
        Args:
            args: Dictionary with 'location', 'start_date', 'end_date'
            
        Returns:
            JSON string with historical data
        """
        try:
            location = args.get("location")
            start_date = args.get("start_date")
            end_date = args.get("end_date")
            
            if not all([location, start_date, end_date]):
                return tool_error("Location, start_date, and end_date are required")
            
            logger.info(f"Fetching historical weather for {location} ({start_date} to {end_date})")
            
            historical_data = self.api.get_historical_data(location, start_date, end_date)
            
            result = {
                "location": location,
                "start_date": start_date,
                "end_date": end_date,
                "historical_data": historical_data,
                "total_records": len(historical_data)
            }
            
            logger.info(f"Successfully fetched {len(historical_data)} historical records")
            return tool_result(result)
        
        except ValueError as e:
            return tool_error(f"Invalid input: {e}")
        except Exception as e:
            logger.error(f"Historical data fetch failed: {e}")
            return tool_error(f"Failed to fetch historical data: {e}")


# Global tool instance
_weather_tool = None


def get_weather_tool() -> WeatherPredictionTool:
    """Get or create the global weather prediction tool instance."""
    global _weather_tool
    if _weather_tool is None:
        _weather_tool = WeatherPredictionTool()
    return _weather_tool


def weather_prediction_fetch_handler(args: Dict[str, Any], **kwargs) -> str:
    """Handler for weather prediction fetch tool."""
    tool = get_weather_tool()
    return tool.fetch_forecast(args)


def weather_alerts_fetch_handler(args: Dict[str, Any], **kwargs) -> str:
    """Handler for weather alerts fetch tool."""
    tool = get_weather_tool()
    return tool.fetch_alerts(args)


def weather_historical_fetch_handler(args: Dict[str, Any], **kwargs) -> str:
    """Handler for weather historical data fetch tool."""
    tool = get_weather_tool()
    return tool.fetch_historical(args)


# Tool registration (if using a registry system)
try:
    from quantara.tools.registry import registry
    
    registry.register(
        name="weather_prediction_fetch",
        toolset="quantara",
        schema=WEATHER_PREDICTION_SCHEMA,
        handler=weather_prediction_fetch_handler,
    )
    
    registry.register(
        name="weather_alerts_fetch",
        toolset="quantara",
        schema=WEATHER_ALERTS_SCHEMA,
        handler=weather_alerts_fetch_handler,
    )
    
    registry.register(
        name="weather_historical_fetch",
        toolset="quantara",
        schema=WEATHER_HISTORICAL_SCHEMA,
        handler=weather_historical_fetch_handler,
    )
    
    logger.info("Weather prediction tools registered successfully")
except (ImportError, ModuleNotFoundError):
    logger.debug("Tool registry not available, tools can be used directly")
