"""
Tests for agent tool functions (Blog 6.1).

Validates pure-Python tool function logic without calling any LLM.
"""


# ── Tool definitions (from blog 6.1) ─────────────────────────────────


def get_weather(city: str) -> str:
    """Get the current weather for a city. Returns temperature and conditions."""
    weather_data = {
        "paris": "18°C, partly cloudy",
        "london": "14°C, rainy",
        "tokyo": "25°C, sunny",
        "new york": "22°C, clear skies",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic."""
    try:
        allowed_chars = set("0123456789.+-*/() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic arithmetic operations are supported."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# ── Tests ─────────────────────────────────────────────────────────────


class TestGetWeather:
    def test_known_city_paris(self):
        assert get_weather("paris") == "18°C, partly cloudy"

    def test_known_city_london(self):
        assert get_weather("london") == "14°C, rainy"

    def test_known_city_tokyo(self):
        assert get_weather("tokyo") == "25°C, sunny"

    def test_known_city_new_york(self):
        assert get_weather("new york") == "22°C, clear skies"

    def test_case_insensitive(self):
        assert get_weather("PARIS") == "18°C, partly cloudy"
        assert get_weather("Paris") == "18°C, partly cloudy"

    def test_unknown_city(self):
        result = get_weather("mars")
        assert "not available" in result

    def test_unknown_city_preserves_name(self):
        result = get_weather("Berlin")
        assert "Berlin" in result


class TestCalculate:
    def test_addition(self):
        assert calculate("2 + 3") == "5"

    def test_subtraction(self):
        assert calculate("10 - 4") == "6"

    def test_multiplication(self):
        assert calculate("6 * 7") == "42"

    def test_division(self):
        assert calculate("15 / 3") == "5.0"

    def test_complex_expression(self):
        assert calculate("(2 + 3) * 4") == "20"

    def test_decimal(self):
        assert calculate("3.14 * 2") == "6.28"

    def test_rejects_letters(self):
        result = calculate("abc")
        assert "Error" in result

    def test_rejects_imports(self):
        result = calculate("__import__('os')")
        assert "Error" in result

    def test_rejects_semicolons(self):
        result = calculate("1; print('hacked')")
        assert "Error" in result
