import sys
from chatgpt import parse_disaster_alert
from rainy import create_rain_effect
from snowy import create_snow_effect
from background_inpainting import background_inpainting
from inpainting import apply_inpainting

def disaster_output(alert_text):
    parsed_alert = parse_disaster_alert(alert_text)
    return parsed_alert